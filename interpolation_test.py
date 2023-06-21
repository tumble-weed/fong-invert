import torch
def interpolate(image1,image2,model,n):
    device = image1.device
    ratio = torch.linspace(1.,0.,n,device=device)[:,None,None,None]
    # import ipdb; ipdb.set_trace()
    mix = image1 * ratio + (1-ratio) * image2
    mix_scores = model(mix)
    return mix_scores,mix

"""
_ = model(ref)
ref_acts2 = activations[0]
(ref_acts - ref_acts2).norm()
"""