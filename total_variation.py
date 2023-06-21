import colorful
import torch
def tv_norm(x, beta=2.):
    # assert(x.size(0) == 1)
    # img = x[0]
    img = x
    dy = img - img # set size of derivative and set border = 0
    dx = img - img
    dy[:,:,1:,:] = -img[:,:,:-1,:] + img[:,:,1:,:]
    dx[:,:,:,1:] = -img[:,:,:,:-1] + img[:,:,:,1:]
    map = ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum(dim=1)
    # import ipdb; ipdb.set_trace()
    return map.sum(),map

#========================================================
def tv_norm_trunc(x, beta=2.,T = 0.5,mode='penalize-smaller'):
    
    # assert(x.size(0) == 1)
    # img = x[0]
    img = x
    dy = img - img # set size of derivative and set border = 0
    dx = img - img
    dy[:,:,1:,:] = -img[:,:,:-1,:] + img[:,:,1:,:]
    dx[:,:,:,1:] = -img[:,:,:,:-1] + img[:,:,:,1:]
    map = ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum(dim=1)
    # import ipdb;ipdb.set_trace()
    if mode == 'penalize-smaller':
        keep = (map <T*3*(2**(beta/2.))).float()
    else:
        print(colorful.red("trunc tv norm to penalize larger"))
        keep = (map >T*3*(2**(beta/2.))).float()
    print(colorful.red(keep.mean().item()))
    # if keep.mean() == 1:
    #     import ipdb; ipdb.set_trace()
    map = map* keep
    return map.sum(),map
#========================================================
def tv_norm_trunc2(x, beta=2.,tv_lambda=1,ratio = 1):
    assert x.max() <= 1., 'computation for the thresholds is for 0 to 1 range'
    assert x.min() >= 0., 'computation for the thresholds is for 0 to 1 range'
    x1 = x.clone()
    def limit_grads(g,ratio=ratio):
        
        assert beta  == 2
        max_diff = 1
        dim_factor = 2
        leftright_factor = 2
        pow_factor = 2
        max_g = (beta/2) * pow_factor * (max_diff * dim_factor) * leftright_factor * tv_lambda
        mag_g = ratio * max_g
        print(colorful.yellow(f'{g.abs().max().item()},{max_g}'))
        g1 = torch.clamp(g,-mag_g,mag_g)
        return g1
    x1.register_hook(limit_grads)
    assert(x.size(0) == 1)
    img = x1[0]
    dy = img - img # set size of derivative and set border = 0
    dx = img - img
    dy[:,1:,:] = -img[:,:-1,:] + img[:,1:,:]
    dx[:,:,1:] = -img[:,:,:-1] + img[:,:,1:]
    map = ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum(dim=0)
    return map.sum(),map



