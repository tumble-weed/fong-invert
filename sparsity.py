
# from hooks import register_backward_hook,register_forward_hook
from hooks import get_pytorch_module
import torchvision.models as models
import torch
from torch import nn
from utils import MyNamespace
import utils
from collections import defaultdict
opts = MyNamespace()
hook_dict ={}
"""
def add_to_activations(activations,output):
    # assert len(activations) in [0,1]
    # # if len(activations) == 1:
    # #     del activations[0]
    # # activations.append(output)
    # if len(activations) == 0:
    #     activations.append(output)
    # activations[0] = output
    opts.__dict__['activations'] = output
"""

def add_to_activations(activations,output):
    assert len(activations) in [0,1]
    if len(activations) == 1:
        del activations[0]
    if len(activations) == 0:
        activations.append(output)

def register_forward_noising_hook(model,layer,
                                    noisy_activations,
                                    activations,  
                                    avg_angle,
                                    avg_mag,
                                  noise_mag=0.1,
                                #   noise_mags = None,
                                  angular_simplicity=True):
    # noisy_activations = []
    # activations = []    
    # avg_angle = []
    # avg_mag = []
    model.noise_flag = True
    layer.noise_mag = noise_mag
    if 'noisy_layers' not in model.__dict__:
        model.noisy_layers = []    
    model.noisy_layers.append(layer)
    # model.noise_mags.append(noise_mag)
    # if noise_mags is not None:
    #     model.noise_mags = noise_mags
    # else:
    #     model.noise_mags = []

    def get_angular_simplicity(module, input, output):
        # calculate the angle between the convolutional module weights and the input
        # import ipdb; ipdb.set_trace()
        assert isinstance(module,nn.Conv2d)
        w0 = module.weight
        w_norm = w0.permute(0,2,3,1).flatten(start_dim=1,end_dim=-1).norm(dim=-1)
        w = w0 / w_norm[:,None,None,None]
        #w.shape = (out_channels,in_channels,h,w)
        padding_mode = module.padding_mode
        if padding_mode == 'zeros':
            padding_mode = 'constant'
            pad_val = 0
            padding = module.padding
            padding = padding[0],padding[0],padding[1],padding[1]
            padded = nn.functional.pad(input[0],padding,padding_mode,pad_val)
        else:
            import ipdb; ipdb.set_trace()
        # unfold the input
        unfolded = padded.unfold(2, w.shape[2], module.stride[0]).unfold(3, w.shape[3], module.stride[1])
        

        assert unfolded.shape[2:3] == output.shape[2:3]
        # normalize the input along the window and the channel dimensions
        unfolded_norm  = unfolded.permute(0,2,3,1,4,5).flatten(start_dim=3,end_dim=-1).norm(dim=-1)
        unfolded_unit = unfolded / unfolded_norm[:,None,:,:,None,None]
        # calculate the dot product with the weights
        dot = torch.einsum('bcHWhw,ochw->boHW',unfolded_unit,w)
        assert dot.shape == output.shape
        # dot.shape = (batch_size,out_channels,h,w)
        # only average the dot product that survive the relu
        kept_dot = dot * (output > 0).float()
        avg_mag_ = (unfolded_norm[:,None,...] * w_norm[None,:,None,None] * (output > 0).float()).mean()
        avg_mag0_ = (unfolded_norm[:,None,...] * (output > 0).float()).mean()
        opts.ORTHOGONALIZE_TO_MAG = True
        if opts.ORTHOGONALIZE_TO_MAG:
            # import ipdb; ipdb.set_trace()
            if input[0].requires_grad:
                grad_of_mag = torch.autograd.grad(avg_mag0_, unfolded, retain_graph=True,create_graph=False)[0]
                # grad_of_mag = torch.autograd.grad(avg_mag_, unfolded, retain_graph=True,create_graph=False)[0]
                grad_of_mag_norm = grad_of_mag.permute(0,2,3,1,4,5).flatten(start_dim=3,end_dim=-1).norm(dim=-1)
                unit_grad_of_mag = grad_of_mag / grad_of_mag_norm[:,None,:,:,None,None]
                def orthogonalize_hook(g):
                    g_along = (g*unit_grad_of_mag)
                    g_along_norm =g_along.permute(0,2,3,1,4,5).flatten(start_dim=-3,end_dim=-1).norm(dim=-1)
                    if (g_along_norm.max() > 1e-5):                        
                        gortho = g - g_along
                        assert ((gortho * unit_grad_of_mag).sum(dim=(1,-2,-1)) <  g_along_norm).all()
                        return gortho
                    else: 
                        return g
                unfolded.register_hook(orthogonalize_hook)
        add_to_activations(avg_angle,kept_dot.mean())
        add_to_activations(avg_mag,avg_mag_)
    def add_noise(module, input, output):
        add_to_activations(activations,output)

        if output.ndim == 4:
            if module.noise_mag > 0:
                if model.noise_flag:
                    # import ipdb; ipdb.set_trace()
                    output_std = output.flatten(start_dim=-2,end_dim=-1).std(dim=-1)
                    noise = module.noise_mag*output_std[...,None,None] * torch.randn_like(output)
                    noisy_output = output + noise
                    noisy_std = noisy_output.std(dim=(-1,-2))
                    # if noise_mag > 0:
                    #     import ipdb; ipdb.set_trace()
                else:
                    noisy_output = output
            else:
                noisy_output = output
        else:
            import ipdb; ipdb.set_trace()
        add_to_activations(noisy_activations,noisy_output)
        return noisy_output
    # _ = get_pytorch_module(model, layer).register_forward_hook(add_noise)    
    # if ('add_noise',layer) in hook_dict:
    #     hook_dict['add_noise',layer].remove()
    if angular_simplicity:
        # if ('angular_simplicity',layer) in hook_dict:
        #     hook_dict['angular_simplicity',layer].remove()        
        hook_dict['angular_simplicity',layer] = layer.register_forward_hook(get_angular_simplicity)
    hook_dict['add_noise',layer] = layer.register_forward_hook(add_noise)
def setup_network_for_comprehensive_sparsity(model,
                                             
                                            activations,
                                            noisy_activations,
                                            avg_angles,
                                            avg_mags,
                                             
                                             layer_type=nn.Conv2d,noise_mag=0.1,
                                             noise_mags  = None,
                                             layer_ixs=None,angular_simplicity=True):
    for mod in model.modules():
        if isinstance(mod,nn.ReLU):
            mod.inplace = False    
    # import ipdb; ipdb.set_trace()
    for il,layer in enumerate(model.modules()):
        
        # import ipdb; ipdb.set_trace()
        if any([
            # isinstance(layer,nn.Linear),
            isinstance(layer,layer_type)
            ]):
            if layer_ixs is not None:
                if il not in layer_ixs:
                    # import ipdb; ipdb.set_trace()
                    continue
            utils.cipdb('DBG_ABLATION')
            activations.append([])
            noisy_activations.append([])
            avg_angles.append([])
            avg_mags.append([])
            activations_layer,noisy_activations_layer,avg_angle,avg_mag = activations[0],noisy_activations[0],avg_angles[0],avg_mags[0]
            if noise_mags is not None:
                noise_mag = noise_mags[il]
            register_forward_noising_hook(model,layer,
                                          activations_layer,noisy_activations_layer,avg_angle,avg_mag,
                                          noise_mag=noise_mag,angular_simplicity=angular_simplicity)    
            # import ipdb; ipdb.set_trace()
        prev_layer = layer

def recursively_delete_tensors(container):
    if isinstance(container,list):
        # for ix,inner in enumerate(container):
        for ix in reversed(range(len(container))):
            inner = container[ix]
            if isinstance(inner,list):
                recursively_delete_tensors(inner)
            else:
                container[ix].grad = None
                del container[ix]

def clear_hooks():
    for k in list(hook_dict.keys()):
        del hook_dict[k]
    # for k in list(assets.keys()):        
    #     if k in assets:
    #         for item in assets[k]:
    #             recursively_delete_tensors(item)
    #             # del assets[k]
    torch.cuda.empty_cache()

if __name__ == '__main__':
    
    network = 'alexnet'
    device = 'cpu'
    def get_model(network,device):
        model = models.__dict__[network](pretrained=True)
        model.eval()
        model.to(device)
        return model
    model = get_model(network,device)
    activations,noisy_activations = setup_network_for_comprehensive_sparsity(model)
    import ipdb; ipdb.set_trace()