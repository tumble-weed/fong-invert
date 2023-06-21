# CUBLAS_WORKSPACE_CONFIG=:4096:8 python invert.py
import register_ipdb
import torch
import torch.nn as nn
import colorful
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

import os
import utils
import matplotlib.pyplot as plt

from PIL import Image
#==================================================================
from dip.common_utils import get_noise#,get_params
from dip.common_utils import tv_loss
from dip.models import skip
from collections import defaultdict
#==================================================================
from hooks import get_pytorch_module,register_backward_hook,register_forward_hook
from diversity import get_region_importances
import random
from elp_masking import get_masked_input
import torch
from total_variation import tv_norm,tv_norm_trunc,tv_norm_trunc2
import torchvision.transforms
DISABLED_FOR_MULTI = False
torch.manual_seed(0)
torch.use_deterministic_algorithms(True,warn_only=True)
np.random.seed(0)
random.seed(0)

# losses
def alpha_prior(x, alpha=2.):
    return torch.abs(x.view(-1)**alpha).sum()




def norm_loss(input, target):
    return torch.div(alpha_prior(input - target, alpha=2.), alpha_prior(target, alpha=2.))

#==================================================================
class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Clip(object):
    def __init__(self):
        return

    def __call__(self, tensor):
        t = tensor.clone()
        t[t>1] = 1
        t[t<0] = 0
        return t

#==================================================================
#function to decay the learning rate
def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor 

def get_edge_alignment(edge_mags):
    normalizer = edge_mags.flatten(start_dim=-2,end_dim=-1).max(dim=-1)[0][...,None,None]
    normalized_edge_mags = edge_mags/normalizer
    # import ipdb;ipdb.set_trace()
    mean_edge_mag = normalized_edge_mags.mean(dim=1,keepdim=True)
    misalignment = (mean_edge_mag-normalized_edge_mags).norm(p=2,dim=(0,-1,-2))
    return misalignment
def channelwise_edge_gradients(tensor):
    # Compute the Sobel operator kernels to detect edges
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=tensor.device)
    sobel_y = sobel_x.t()

    # Compute the horizontal and vertical gradient components for each channel
    grad_x = F.conv2d(tensor.flatten(start_dim=0,end_dim=1).unsqueeze(1), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(tensor.flatten(start_dim=0,end_dim=1).unsqueeze(1), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    grad_x = grad_x.view(tensor.shape[0],-1,*tensor.shape[2:])
    grad_y = grad_y.view(tensor.shape[0],-1,*tensor.shape[2:])
    # Compute the magnitude and direction of the gradient for each channel
    magnitudes = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    directions = torch.atan2(grad_y, grad_x)

    return magnitudes, directions

def main(image, network='alexnet', size=227, layer='features.4', alpha=6, beta=2, 
        alpha_lambda=1e-5,  tv_lambda=1e-5, epochs=200, learning_rate=1e2, 
        momentum=0.9, decay_iter=100, decay_factor=1e-1, print_iter=25, 
        device='cpu',method=None,cam_weight=None):


    trends = defaultdict(list)
    transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.CenterCrop(size=size),
        transforms.ToTensor(),
        transforms.Normalize(utils.mu, utils.sigma),
    ])

    detransform = transforms.Compose([
        Denormalize(utils.mu, utils.sigma),
        Clip(),
        transforms.ToPILImage(),
    ])

    
    ref = transform(Image.open(image)).unsqueeze(0)
    print(ref.size())
    ref = ref.to(device)
    def get_model(network,device):
        model = models.__dict__[network](pretrained=True)
        model.eval()
        model.to(device)
        return model
    model = get_model(network,device)


    
        
    def setup_network(model,layer,backward=False):
        # import ipdb;ipdb.set_trace()
        for mod in model.modules():
            if isinstance(mod,nn.ReLU):
                mod.inplace = False
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

        activations = register_forward_hook(model,layer)
        if backward:
            grads = register_backward_hook(model,layer)
            return activations,grads
        return activations
    
    activations = setup_network(model,layer)
    # setup for gradcam
    # default gradcam layer for alexnet
    gradcam_layer = 'features.11'
    activations_gradcam,grads_gradcam = setup_network(model,gradcam_layer,backward=True)

    # def invert_(ref,model,alpha,beta,learning_rate,momentum,epochs):
    
    # ref_acts = get_acts(model, ref).detach()
    # def get_acts(model, input): 
    #     del activations[:]
    #     _ = model(input)
    #     assert(len(activations) == 1)
    #     return activations[0]
    # import ipdb; ipdb.set_trace()
    ref_edge_mags,ref_edge_dirs = channelwise_edge_gradients(ref)
    ref_edge_alignment =  get_edge_alignment(ref_edge_mags)
    # import ipdb;ipdb.set_trace()
    for chan in range(3):
        utils.img_save2(ref_edge_mags[:,chan:chan+1].cpu().numpy(), f'ref_edge_mags_{chan}.png')
    ref_scores = model(ref)
    target_class = ref_scores.argmax(dim=-1)
    target_class = 153
    print('target class: ',target_class)
    assert len(activations) == 1
    ref_acts = activations[0].detach()
    if method == 'fong':
        prng = np.random.RandomState(1)
        if False:
            x_ = (torch.tensor( 1e-3 * 
                prng.randn(*ref.size())
                )).float().to(device).requires_grad_(True)
        else:
            n_noise = 100
            # noise_sigma = 0.1            
            noise_sigma = 1
            x__ = (torch.tensor( 
                prng.uniform(size = ref.size()) + noise_sigma * prng.randn(n_noise, *ref.size()[1:])
                )).float().to(device).requires_grad_(True)
        optimizer = torch.optim.SGD([x__], lr=learning_rate, momentum=momentum)
    elif method == 'dip':
        ############################################################
        imsize = 224
        # Something divisible by a power of two
        imsize_net = 256
        # VGG and Alexnet need input to be correctly normalized
        ############################################################
        ############################################################
        INPUT = 'noise'
        input_depth = 32
        net_input = get_noise(input_depth, INPUT, imsize_net).type(torch.cuda.FloatTensor).detach()
        pad='reflection'
        PLOT = True
        ############################################################
        tv_weight=0.0
        reg_noise_std = 0*0.03
        param_noise = False
        # num_iter = 2000
        LR = 0.001    
        dtype = torch.cuda.FloatTensor
        ############################################################
        
        net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                                num_channels_up =   [16, 32, 64, 128, 128, 128],
                                num_channels_skip = [0, 4, 4, 4, 4, 4],   
                                filter_size_down = [5, 3, 5, 5, 3, 5], filter_size_up = [5, 3, 5, 3, 5, 3], 
                                upsample_mode='bilinear', downsample_mode='avg',
                                need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(dtype)
        #------------------------------------------------
        # Compute number of parameters
        s  = sum(np.prod(list(p.size())) for p in net.parameters())
        print ('Number of params: %d' % s)

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        ############################################################                
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        ############################################################                


#         alpha_f = lambda x: alpha_prior(x, alpha=alpha)
#         tv_f = lambda x: tv_norm(x, beta=beta)
#         loss_f = lambda x: norm_loss(x, ref_acts)

    if False:
        from elp import extremal_perturbation
        extremal_perturbation(model,
                            input,
                            target,
                            areas=[0.1],
                            perturbation=BLUR_PERTURBATION,
                            max_iter=800,
                            num_levels=8,
                            step=7,
                            sigma=21,
                            jitter=True,
                            variant=PRESERVE_VARIANT,
                            print_iter=None,
                            debug=False,
                            reward_func=simple_reward,
                            resize=False,
                            resize_mode='bilinear',
                            smooth=0)

    for i in range(epochs):
#             acts = get_acts(model, x_)
        del activations[:]
        del activations_gradcam[:]
        del grads_gradcam[:]
        if method == 'dip':
            if param_noise:
                for n in [x for x in net.parameters() if len(x.size()) == 4]:
                    n = n + n.detach().clone().normal_() * n.std()/50
            net_input = net_input_saved
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            x_ = net(net_input)[:, :, :imsize, :imsize]
            x_ = (x_- 0.5)*2
        elif method == 'fong':
            x_ = (x__- 0.5)*2
            
        if False and 'color jitter':
            color_jitter = transforms.ColorJitter(
                                                    # brightness=0.5*1,
                                                    # contrast=0.5*1,
                                                    # saturation=0.5*1,
                                                    hue=0.1
                                                    )            
            x_ = color_jitter(x_)
            x_ = x_.contiguous()
        # import ipdb; ipdb.set_trace()                 
        # create noisy copies of x_

        #==================================================
        # median filter grads
        if False:
            def median_filter_grads(g):
                g1 = g.median(dim=0,keepdim=True)[0]
                return g1
            x__.register_hook(median_filter_grads)
        #==================================================
        # suppress high std grads
        if False:
            def std_filter_grads(g):
                g = g
                g_std = g.std(dim=0,keepdim=True)
                g_std = g_std.mean(dim=1,keepdim=True)
                return g*(g_std < 0.95*g_std.max()).float()

            x_.register_hook(std_filter_grads)
        
        mix_ratio = 0.9 + 0.1* torch.rand(size=(n_noise,),device=device)
        x_for_forward_ = (mix_ratio[:,None,None,None]*x_[np.random.randint(0,n_noise,n_noise)] + (1-mix_ratio[:,None,None,None])*x_[np.random.randint(0,n_noise,n_noise)].detach())
        # import ipdb; ipdb.set_trace()
        x_scores = model(x_for_forward_)
        x_probs = torch.softmax(x_scores, dim=-1)
        acts = activations[0]
        gradcam_acts = activations_gradcam[0]
        alpha_term = alpha_prior(x_, alpha=alpha)
        #---------------------------------------------------------------
        print(colorful.red("add tv norm with gradient clipping"))
        print(colorful.red("add gradient clipping to the network gradients"))
        #*****************************************************************
        tv_norm_x,tv_map = tv_norm(x__, beta=beta)
        tv_norm_trunc_x,tv_trunc_map = tv_norm_trunc(x__, beta=beta, 
                                                    #  T = 0.15
                                                    T = 0.9
                                                     )
        # tv_norm_trunc2_x,tv_trunc2_map = tv_norm_trunc2(
        #     # x__.detach().clone().requires_grad_(True), 
        #     x__,
        #     beta=beta, tv_lambda=tv_lambda,ratio = 0.5)
        #*****************************************************************
        tv_term = tv_norm_x
        # tv_term = tv_norm_trunc_x
        # tv_term = tv_norm_trunc2_x
        #---------------------------------------------------------------
        x_edge_mags,x_edge_dirs = channelwise_edge_gradients(x__)
        x_edge_alignment =  get_edge_alignment(x_edge_mags)        
        if (i+1)%print_iter == 0:
            print(x_edge_alignment)
        #---------------------------------------------------------------
        assert x_scores.ndim == 2
        feat_diversity = x_scores.std(dim=0).sum()
        # assert not x_probs.log().isinf().any()
        # assert not x_probs.log().isnan().any()
        # kl_1v1 = (F.kl_div(x_probs.log()[:,None,:],x_probs[None,:,:],log_target=True,reduction='none') - F.kl_div(x_probs.log(),x_probs,log_target=True,reduction='none')[None,None]).sum()
        kl_1v1 = (F.kl_div(x_probs[:,None,:],x_probs[None,:,:],log_target=False,reduction='none') - F.kl_div(x_probs,x_probs,log_target=False,reduction='none')[None,None]).sum()

        kl_diversity = kl_1v1
        # kl_from_original = kl_from_original.view(probs.shape[0],probs.shape[1])
        #---------------------------------------------------------------
        # loss_term = norm_loss(acts, ref_acts)
        if os.environ.get('USE_CROSS_ENTROPY',False) == '1':
            loss_term = - torch.log(x_probs[:,target_class]).sum()
        else:
            loss_term = -x_scores[:,target_class].sum()
        if False:
            act_diff = get_region_importances(x__,model)
            trends['act_diff'].append(act_diff.item())
        if method == 'fong':
            # if os.environ.get('USE_CROSS_ENTROPY',False) == '1':
            #     tv_lambda =  1e-2
                
            diversity_lambda = 1.
            loss_lambda = 0.5
            if target_class == 153:
                if os.environ.get('USE_CROSS_ENTROPY',False) == '1':
                    loss_lambda = 1
                else:
                    loss_lambda = 0.3
                # loss_lambda = 0.3
                diversity_lambda = 0.5
            
            print(colorful.yellow(f"multiplying loss term by factor {loss_lambda}"))
            tot_loss = alpha_lambda*alpha_term + tv_lambda*tv_term + loss_lambda*loss_term - diversity_lambda*feat_diversity                                
        else:
            tot_loss = loss_term
        # tot_loss = loss_term
        trends['tv_term'].append(tv_term.item())
        trends['loss_term'].append(loss_term.item())
        
        optimizer.zero_grad()
        tot_loss.backward(retain_graph=True)
        #---------------------------------------------------------------
        # feature diversity

        # import ipdb; ipdb.set_trace()
        
        trends['particle_feat_diversity'].append(feat_diversity.item())        
        trends['particle_kl_diversity'].append(kl_diversity.item())        
        
        #---------------------------------------------------------------
            
        if method == 'fong':
            if x__.grad.isnan().any() or x__.grad.isinf().any():
                import ipdb; ipdb.set_trace()    
        if 'gradcam':
            weights = torch.mean(grads_gradcam[0], dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * gradcam_acts[0], dim=1, keepdim=True)
            '''
            the cam will need to be ultiplied by -1, as the gradient is for negative class score
            '''
            if False:
                cam0 = torch.nn.functional.relu(-cam)         
                cam = torch.nn.functional.relu(-cam)        
            else:
                softplus = nn.Softplus(beta=1, threshold=20) 
                cam0 = torch.nn.functional.relu(-cam)
                cam = softplus(-cam)
                         
                
            tv_cam,tv_cam_map = tv_norm(cam, beta=2)

        if True:
            if True:
                cam_norm = cam.sum()
                cam0_norm = cam0.sum()
                cam0_sparsity = (cam0>0).float().sum()
                cam_loss = cam_weight*cam_norm + 0*10*tv_cam
            else:
                print(colorful.red("using squared sum for cam loss"))
                cam_loss = cam_weight*(cam**2).sum()
            cam_loss.backward()
        
        trends['cam0_norm'].append(cam0_norm.item())
        trends['cam0_sparsity'].append(cam0_sparsity.item())
        trends['cam_norm'].append(cam_norm.item())
        trends['cam_tv'].append(tv_cam.item())
        optimizer.step()

        if (i+1) % print_iter == 0:
            if False:
                print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f\tCAM: %f\tprob/max_prob: %f/%f' % (i+1,
                    alpha_term.item(), tv_term.data.item(),
                    loss_term.item(), tot_loss.item(),
                    cam.sum().item(),
                    x_probs[:,target_class].item(),x_probs.max(dim=-1)[0].item()
                    ))
            try:
                for j in [0,n_noise//4,2*n_noise//4,3*n_noise//4]:
                    utils.img_save2(x__[j:j+1],f'multi_particle_x_{j}_{i}.png')
                
            except Exception as e:
                import ipdb;ipdb.set_trace()
            # import ipdb;ipdb.set_trace()
            
            cam_up = torch.nn.functional.interpolate(cam, size=ref.size()[-2:], mode='bilinear', align_corners=False)
            cam_up = cam_up/( cam_up.max() + (cam_up.max() == 0).float())
            # cam_up = (cam_up>0).float()
            utils.img_save2(cam_up,f'gradcam_{i}.png')
            
            
            cam0_up = torch.nn.functional.interpolate(cam0, size=ref.size()[-2:], mode='bilinear', align_corners=False)
            cam0_up = cam0_up/( cam0_up.max() + (cam0_up.max() == 0).float())
            # cam_up = (cam_up>0).float()
            utils.img_save2(cam0_up,f'gradcam0_{i}.png')
            # import ipdb; ipdb.set_trace()
            utils.img_save2((tv_map/tv_map.max())[:1],f'tv_map_{i}.png')
            if DISABLED_FOR_MULTI:
                try:
                    utils.img_save2(tv_trunc_map/tv_trunc_map.max(),f'tv_trunc_map_{i}.png')
                except ValueError:
                    pass
            for lname in ['cam0_norm','cam0_sparsity','cam_norm','cam_tv','tv_term','loss_term','particle_feat_diversity','particle_kl_diversity']:
                utils.save_plot2(trends[lname],lname,f'{lname}.png')

        if decay_iter is not None:
            if (i+1) % decay_iter == 0:
                decay_lr(optimizer, decay_factor)
        if True and method == 'fong':
            x__.data.copy_(torch.clamp(x__, 0, 1))
    # f, ax = plt.subplots(1,2)
    # ax[0].imshow(detransform(img_[0]))
    # ax[1].imshow(detransform(x_[0].data.cpu()))
    # for a in ax:
    #     a.set_xticks([])
    #     a.set_yticks([])
    # plt.show()
    x_edge_mags,x_edge_dirs = channelwise_edge_gradients(x__.detach())
    for chan in range(3):
        utils.img_save2(x_edge_mags[:,chan:chan+1].cpu().numpy(), f'x_edge_mags_{chan}.png')    
    # if DISABLED_FOR_MULTI:
    #     act_diff = get_region_importances(x__,model)

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        import sys;
        if False:
            parser = argparse.ArgumentParser()
            parser.add_argument('--image', type=str,
                    default='grace_hopper.jpg')
            parser.add_argument('--network', type=str, default='alexnet')
            parser.add_argument('--size', type=int, default=227)
            parser.add_argument('--layer', type=str, default='features.4')
            parser.add_argument('--alpha', type=float, default=6.)
            parser.add_argument('--beta', type=float, default=2.)
            parser.add_argument('--alpha_lambda', type=float, default=1e-5)
            parser.add_argument('--tv_lambda', type=float, default=1e-5)
            parser.add_argument('--epochs', type=int, default=200)
            parser.add_argument('--learning_rate', type=int, default=1e2)
            parser.add_argument('--momentum', type=float, default=0.9)
            parser.add_argument('--print_iter', type=int, default=25)
            parser.add_argument('--decay_iter', type=int, default=100)
            parser.add_argument('--decay_factor', type=float, default=1e-1)
            parser.add_argument('--gpu', type=int, nargs='*', default=None)
            parser.add_argument('--method', type=str,  default='fong')
            args = parser.parse_args()
        else:
            args = argparse.Namespace()
            args.image = 'grace_hopper.jpg'
            args.network = 'alexnet'
            args.size = 227
            # args.layer = 'features.4'
            args.layer = 'classifier.6'
            args.alpha = 6.
            args.beta = 2.
            # args.alpha_lambda = 1e-5
            args.alpha_lambda = 0e-1
            # args.tv_lambda = 1e-5
            # args.tv_lambda = 1e-1
            args.tv_lambda = 1e-1
            # args.tv_lambda = 1
            
            
            # args.learning_rate = 1e2
            args.momentum = 0.9
            args.print_iter = 25
            
            args.decay_factor = 1e-1
            args.device = 'cuda'
            args.method = 'fong'
            
            if args.method == 'fong':
                args.decay_iter = None
                args.learning_rate = 1e-1
                args.epochs = 4000
                # args.cam_weight = 5*1
                # args.cam_weight = 4
                # args.cam_weight = 1
                args.cam_weight = 0
            elif args.method == 'dip':
                args.decay_iter = None
                args.learning_rate = 1e-3
                args.epochs = 2000
                args.cam_weight = 5e0
        # gpu = args.gpu
        # cuda = True if gpu is not None else False
        # use_mult_gpu = isinstance(gpu, list)
        # if cuda:
        #     if use_mult_gpu:
        #         os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
        #     else:
        #         os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        # print(torch.cuda.device_count(), use_mult_gpu, cuda)

        main(image=args.image, network=args.network, layer=args.layer, 
                alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda, 
                tv_lambda=args.tv_lambda, epochs=args.epochs,
                learning_rate=args.learning_rate, momentum=args.momentum, 
                print_iter=args.print_iter, decay_iter=args.decay_iter,
                decay_factor=args.decay_factor, 
                device=args.device,method=args.method,cam_weight=args.cam_weight)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


