import torch
from hooks import get_pytorch_module, register_backward_hook, register_forward_hook
import utils
from elp_masking import get_masked_input
# if 'get region importances':
def get_region_importances(x__, 
                           model, 
                           diversity_layer = 'classifier.4',
                           window_size=91, stride=1, batch_size=64):
    # import ipdb;ipdb.set_trace()
    # model.classifier['4']
    device = x__.device
    diversity_activations = register_forward_hook(model,diversity_layer)
    '''
    diversity_activations = []    
    def hook_acts_diversity(module, input, output,diversity_activations=diversity_activations):
        assert len(diversity_activations) in [0,1]
        if len(diversity_activations) == 1:
            del diversity_activations[0]
        diversity_activations.append(output)        
    _ = get_pytorch_module(model, diversity_layer).register_forward_hook(hook_acts_diversity)
    window_size = 91
    stride = 1
    '''

    # num_masks = ((input_size - window_size) // stride) + 1
    I1 = range(window_size//2, x__.shape[-2] - window_size//2,stride)
    I2 = range(window_size//2,x__.shape[-1] - window_size//2,stride)
    masks = torch.zeros((len(I1)*len(I2), 1, *x__.shape[-2:]),device=device)
    
    for ii1,i1 in enumerate(I1):
        for ii2,i2 in enumerate(I2):            
            # mask_= torch.zeros_like(x__[:,:1])
            masks[ii1*len(I2) + ii2,:,i1-(window_size//2):i1+(window_size//2)+1,i2-(window_size//2):i2+(window_size//2)+1] = 1
    x_= (x__ - 0.5)*2
    _ = model(x_)
    base_activation = diversity_activations[0].detach().clone()
    batch_size = 64
    act_diff = torch.zeros((len(I1)*len(I2)),device=device)
    for bi in range(0,(len(masks) + batch_size -1)//batch_size):
        # masks = torch.zeros((len(I1)*len(I2), 1, *x__.shape[-2:]),device=device)
        if 'elp masking':
            masked,_ = get_masked_input(
                            x__,
                            1-masks[(bi)*batch_size:(bi+1)*batch_size],
                            # perturbation=BLUR_PERTURBATION,
                            num_levels=8,
                            # variant=PRESERVE_VARIANT,
                            smooth=0)
        elif False:
            masked = x__ * (1 - masks)
        masked = (masked- 0.5)*2
        _ = model(masked)
        masked_activation = diversity_activations[0].detach().clone()
        act_diff[bi*batch_size:(bi+1)*batch_size] = (base_activation - masked_activation).flatten(start_dim=1,end_dim=-1).norm(2,dim=-1)
        
        # act_diff = (base_activation[bi:bi+batch_size] - masked_activation).flatten(start_dim=1,end_dim=-1).norm(2,dim=-1)
        # act_diff = act_diff.reshape(len(I1),len(I2))
        # utils.img_save2(act_diff, f'act_diff_{bi}.png')
        
    # _ = model(masked)
    # masked_activation = diversity_activations[0].detach().clone()
    # act_diff = (base_activation - masked_activation).flatten(start_dim=1,end_dim=-1).norm(2,dim=-1)
    act_diff = act_diff.reshape(len(I1),len(I2))
    utils.img_save2(act_diff, 'act_diff.png')
    return act_diff