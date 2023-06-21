import torch
import colorful
def sample_scale_and_crop(full,crop_size,n,min_scale = 3/4.,max_scale = 4/3.):
    device =full.device
    # sample random affine matrices for torch affine grid    
    theta = torch.zeros(n,2, 3,device=device)
    theta[:,0,0] = theta[:,0,0] + min_scale + (max_scale - min_scale)*torch.rand(n,device=device)
    theta[:,1,1] = theta[:,1,1] + min_scale + (max_scale - min_scale)*torch.rand(n,device=device)
    crop_ratio_y,crop_ratio_x = (crop_size[0]//2)/full.shape[2],(crop_size[1]//2)/full.shape[3]
    tx = (torch.rand(n,device=device) - 0.5 )*2
    ty = (torch.rand(n,device=device) - 0.5 )*2
    # correct for the scale
    tx = tx * (1 - crop_ratio_x/theta[:,0,0])
    ty = ty * (1 - crop_ratio_y/theta[:,1,1])
    theta[:,0,2] = tx
    theta[:,1,2] = ty
    print(colorful.red("check if y and x are in the correct rows"))
    # import ipdb; ipdb.set_trace()
    return theta
theta = sample_scale_and_crop(torch.ones(1,3,336,336),(224,224),10,min_scale = 0.5,max_scale = 2)