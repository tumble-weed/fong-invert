import torch
from utils import MyNamespace
from collections import defaultdict
hook_dict = {}
opts = MyNamespace()
opts.T = 0.1
assets = defaultdict(list)
def get_pytorch_module(net, blob):
    modules = blob.split('.')
    if len(modules) == 1:
        return net._modules.get(blob)
    else:
        curr_m = net
        for m in modules:
            curr_m = curr_m._modules.get(m)
        return curr_m
    

def register_backward_hook(layer):
    # import ipdb; ipdb.set_trace()
    def hook_grad(module, grad_in, grad_out):
        module.grads = grad_out[0]
    layer.register_backward_hook(hook_grad)   

# activations = []    
def register_forward_hook(model,layer,activations,backward=False):
    
    def hook_acts(module, input, output):
        assert len(activations) in [0,1]
        if len(activations) == 1:
            del activations[0]
        # output = output.clone()
        if backward and output.requires_grad:
            output.retain_grad()
        activations.append(output)
        # if len(activations) == 0:
        #     activations.append(output)
        # activations[0] = output
    layer.register_forward_hook(hook_acts)        
    if False:
        if ('forward',layer) in hook_dict:
            hook_dict['forward',layer].remove()
        hook_dict['forward',layer] = get_pytorch_module(model, layer).register_forward_hook(hook_acts)
    # if True:
    #     if ('forward',layer) in hook_dict:
    #         return assets[('forward',layer)]
    #     else:
    #         hook_dict['forward',layer] = get_pytorch_module(model, layer).register_forward_hook(hook_acts)
    #         assets[('forward',layer)] = activations
    # return activations
#============================================================================================
from torch import nn
"""
def add_to_activations(activations,output):
    assert len(activations) in [0,1]
    # if len(activations) == 1:
    #     del activations[0]
    # activations.append(output)
    if len(activations) == 0:
        activations.append(output)
    activations[0] = output

"""
def add_to_activations(activations,output):
    assert len(activations) in [0,1]
    if len(activations) == 1:
        del activations[0]
    if len(activations) == 0:
        activations.append(output)


def register_conv_hook(model,layer,activations):

    # def get_act(module, input, output):
    #     add_to_activations(activations,output)
    
    # if ('conv',layer) in hook_dict:
    #     hook_dict['conv',layer].remove()
    # hook_dict['conv',layer] = layer.register_forward_hook(get_act)
    register_forward_hook(model,layer,activations)

def setup_network_for_conv_acts(model,activations,layer_type=nn.Conv2d):
    for mod in model.modules():
        if isinstance(mod,nn.ReLU):
            mod.inplace = False    
    for layer in model.modules():
        # import ipdb; ipdb.set_trace()
        if any([
            # isinstance(layer,nn.Linear),
            isinstance(layer,layer_type)
            ]):
            activations.append([])
            # import ipdb; ipdb.set_trace()
            register_conv_hook(model,layer,activations[-1])    
        prev_layer = layer



class SoftMaxPool2d(nn.Module):
    def __init__(self,T=opts.T,kernel_size=2,stride=2):
        super(SoftMaxPool2d,self).__init__()
        self.T = T
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self,x):
        # import ipdb; ipdb.set_trace()
        n,c,h,w = x.size()
        unfolded = x.unfold(2,self.kernel_size,self.stride).unfold(3,self.kernel_size,self.stride).flatten(start_dim=-2,end_dim=-1)
        p = nn.functional.softmax(unfolded/self.T,dim=-1)
        x = (p*unfolded).sum(dim=-1)
        #x.shape = (n,c,h,w)
        return x



def setup_network(model,layer,activations,grads=None,backward=False):
    # import ipdb;ipdb.set_trace()
    for mod in model.modules():
        if isinstance(mod,nn.ReLU):
            mod.inplace = False
    # replace MaxPool in features with SoftMaxPool
    # features is a sequential module, replace max_pool within it
    if False and 'replace with softmaxpool':
        old_features = model.features
        new_features = nn.ModuleList()
        for mod in old_features:
            if isinstance(mod,nn.MaxPool2d):
                # import ipdb; ipdb.set_trace()
                mod = SoftMaxPool2d(kernel_size=mod.kernel_size,stride=mod.stride)
            new_features.append(mod)
        new_features = nn.Sequential(*new_features)
        model.features = new_features
    """
    if False:
        '''Sets up the network to extract activations from a given layer.'''
        activations = []    
        def hook_acts(module, input, output):
            assert len(activations) in [0,1]
            if len(activations) == 1:
                del activations[0]
            activations.append(output)
        _ = get_pytorch_module(model, layer).register_forward_hook(hook_acts)
        if backward:
            grads = []
            # import ipdb; ipdb.set_trace()
            def hook_grad(module, grad_in, grad_out):
                assert len(grads) in [0,1]
                if len(grads) == 1:
                    del grads[0]
                grads.append(grad_out[0])
            _ = get_pytorch_module(model, layer).register_full_backward_hook(hook_grad)
            
            return activations,grads
        return activations
    """

    register_forward_hook(model,layer,activations,backward=backward)
    # assets['forward',layer].append(activations)
    if False and backward:
        assert grads is not None
        register_backward_hook(model,layer,grads)
        # assets['backward',layer].append(grads)
        # return activations,grads
        return
    # return activations
    return

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
                
#==========================================================
# prints currently alive Tensors and Variables
import torch
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass
#==========================================================