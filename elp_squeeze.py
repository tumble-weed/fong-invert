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
from total_variation import tv_norm,tv_norm_trunc
from elp import *
from elp import MaskGenerator
from torch import optim
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

def main(image, network='alexnet', size=227, 
         layer1='features.4', 
         layer0 = 'features.3',
         alpha=6, beta=2, 
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


    
        
    def setup_network(model,layer0,layer1,backward=False):
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

        activations0 = register_forward_hook(model,layer0)
        activations1 = register_forward_hook(model,layer1)

        return activations0,activations1
    
    activations0,activations1 = setup_network(model,layer0,layer1)    
    if False:
        ref_edge_mags,ref_edge_dirs = channelwise_edge_gradients(ref)
        ref_edge_alignment =  get_edge_alignment(ref_edge_mags)
        # import ipdb;ipdb.set_trace()
        for chan in range(3):
            utils.img_save2(ref_edge_mags[:,chan:chan+1].cpu().numpy(), f'ref_edge_mags_{chan}.png')
        ref_scores = model(ref)
        target_class = ref_scores.argmax(dim=-1)
    # target_class = 153

    # print('target class: ',target_class)
    # assert len(activations) == 1
    # ref_acts = activations[0].detach()
    ref_scores = model(ref)
    ref_act0= activations0[0].detach().clone()
    ref_act1= activations1[0].detach().clone()

    


    # extremal_perturbation(model,
    #                     input,
    #                     target,
    #                     areas=[0.1],
    #                     perturbation=BLUR_PERTURBATION,
    #                     max_iter=800,
    #                     num_levels=8,
    #                     step=7,
    #                     sigma=21,
    #                     jitter=True,
    #                     variant=PRESERVE_VARIANT,
    #                     print_iter=None,
    #                     debug=False,
    #                     reward_func=simple_reward,
    #                     resize=False,
    #                     resize_mode='bilinear',
    #                     smooth=0)
    
    areas=[0.1]
    perturbation=BLUR_PERTURBATION
    max_iter=800
    num_levels=8
    step=7
    sigma=21
    jitter=False
    variant=PRESERVE_VARIANT
    print_iter=25
    debug=False
    reward_func=simple_reward
    resize=False
    resize_mode='bilinear'
    smooth=0
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.01
    regul_weight = 300#300
    device = ref.device

    regul_weight_last = max(regul_weight / 2, 1)

    if True:
        print(
            f"extremal_perturbation:\n"
            # f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- image size: {list(ref.shape)}\n"
            f"- reward function: {reward_func.__name__}"
        )

    # Disable gradients for model parameters.
    # TODO(av): undo on leaving the function.
    for p in model.parameters():
        p.requires_grad_(False)

    # Get the perturbation operator.
    # The perturbation can be applied at any layer of the network (depth).
    print(colorful.red("is ref normalized"))
    perturbation = Perturbation(
        ref,
        num_levels=num_levels,
        type=perturbation
    ).to(device)

    perturbation_str = '\n  '.join(perturbation.__str__().split('\n'))
    if debug:
        print(f"- {perturbation_str}")

    # Prepare the mask generator.
    shape = perturbation.pyramid.shape[2:]
    mask_generator = MaskGenerator(shape, step, sigma).to(device)
    h, w = mask_generator.shape_in
    pmask = torch.ones(len(areas), 1, h, w).to(device)
    if debug:
        print(f"- mask resolution:\n  {pmask.shape}")

    # Prepare reference area vector.
    max_area = np.prod(mask_generator.shape_out)
    reference = torch.ones(len(areas), max_area).to(device)
    for i, a in enumerate(areas):
        reference[i, :int(max_area * (1 - a))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmask],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((len(areas), 2, 0))

    for t in range(max_iter):
        pmask.requires_grad_(True)

        # Generate the mask.
        mask_, mask = mask_generator.generate(pmask)

        # Apply the mask.
        if variant == DELETE_VARIANT:
            x = perturbation.apply(1 - mask_)
        elif variant == PRESERVE_VARIANT:
            x = perturbation.apply(mask_)
        elif variant == DUAL_VARIANT:
            x = torch.cat((
                perturbation.apply(mask_),
                perturbation.apply(1 - mask_),
            ), dim=0)
        else:
            assert False

        # Apply jitter to the masked data.
        if jitter and t % 2 == 0:
            x = torch.flip(x, dims=(3,))

        # Evaluate the model on the masked data.
        y = model(x)
        x_act0 = activations0[0]
        x_act1 = activations1[0]
        
        # Get reward.
        # reward = reward_func(y, target, variant=variant)
        
        # be as close to act1, while being as far away from act0
        loss0 = (x_act0 - ref_act0).norm(2)
        loss1 = (x_act1 - ref_act1).norm(2)
        feat_loss = 1*loss1 - 0*loss0 

        # Reshape reward and average over spatial dimensions.
        
        feat_loss = feat_loss.reshape(len(areas), -1).mean(dim=1)

        # Area regularization.
        mask_sorted = mask.reshape(len(areas), -1).sort(dim=1)[0]
        regul =  ((mask_sorted - reference)**2).mean(dim=1) * regul_weight
        loss = (0*feat_loss + regul).sum()

        # Gradient step.
        optimizer.zero_grad()
        (loss).backward()
        optimizer.step()
        trends['loss'].append(loss.item())
        trends['feat_loss'].append(feat_loss.item())        
        trends['loss0'].append(loss0.item())        
        trends['loss1'].append(loss1.item())        
        trends['regul'].append(regul.item())
        
        pmask.data = pmask.data.clamp(0, 1)

        # Record energy.
        hist = torch.cat(
            (hist,
             torch.cat((
                 feat_loss.detach().cpu().view(-1, 1, 1),
                 regul.detach().cpu().view(-1, 1, 1)
             ), dim=1)), dim=2)

        # Adjust the regulariser/area constraint weight.
        if False:
            regul_weight *= 1.0035

        # Diagnostics.
        debug_this_iter = debug and (t in (0, max_iter - 1)
                                     or regul_weight / regul_weight_last >= 2)

        if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="")
            for i, area in enumerate(areas):
                print(" [area:{:.2f} loss:{:.2f} reg:{:.2f}]".format(
                    area,
                    hist[i, 0, -1],
                    hist[i, 1, -1]), end="")
            print()
            utils.img_save2(x,(f'elp_x_{t}.png'))
            for lname in ['loss','loss0','loss1','regul','energy']:
                utils.save_plot2(trends[lname],lname,f'elp_{lname}.png')
        if debug_this_iter:
            '''
            regul_weight_last = regul_weight
            for i, a in enumerate(areas):
                plt.figure(i, figsize=(20, 6))
                plt.clf()
                ncols = 4 if variant == DUAL_VARIANT else 3
                plt.subplot(1, ncols, 1)
                plt.plot(hist[i, 0].numpy())
                plt.plot(hist[i, 1].numpy())
                plt.plot(hist[i].sum(dim=0).numpy())
                plt.legend(('energy', 'regul', 'both'))
                plt.title(f'target area:{a:.2f}')
                plt.subplot(1, ncols, 2)
                imsc(mask[i], lim=[0, 1])
                plt.title(
                    f"min:{mask[i].min().item():.2f}"
                    f" max:{mask[i].max().item():.2f}"
                    f" area:{mask[i].sum() / mask[i].numel():.2f}")
                plt.subplot(1, ncols, 3)
                imsc(x[i])
                if variant == DUAL_VARIANT:
                    plt.subplot(1, ncols, 4)
                    imsc(x[i + len(areas)])
                plt.pause(0.001)
            '''

    mask_ = mask_.detach()

    # Resize saliency map.
    if False:
        mask_ = resize_saliency(input,
                                mask_,
                                resize,
                                mode=resize_mode)

        # Smooth saliency map.
        if smooth > 0:
            mask_ = imsmooth(
                mask_,
                sigma=smooth * min(mask_.shape[2:]),
                padding_mode='constant'
            )

#==============================================================
#==============================================================        



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
            args.layer1 = 'classifier.6'
            args.layer0 = 'classifier.4'
            args.alpha = 6.
            args.beta = 2.
            # args.alpha_lambda = 1e-5
            args.alpha_lambda = 0e-1
            # args.tv_lambda = 1e-5
            # args.tv_lambda = 1e-1
            args.tv_lambda = 1e-1
            
            
            # args.learning_rate = 1e2
            args.momentum = 0.9
            args.print_iter = 25
            
            args.decay_factor = 1e-1
            args.device = 'cuda'
            args.method = 'fong'
            
            if args.method == 'fong':
                args.decay_iter = None
                args.learning_rate = 1e-1
                args.epochs = 1000
                # args.cam_weight = 5*1
                args.cam_weight = 4
                # args.cam_weight = 1
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

        main(image=args.image, network=args.network, 
             layer0=args.layer0, 
             layer1=args.layer1, 
                alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda, 
                tv_lambda=args.tv_lambda, epochs=args.epochs,
                learning_rate=args.learning_rate, momentum=args.momentum, 
                print_iter=args.print_iter, decay_iter=args.decay_iter,
                decay_factor=args.decay_factor, 
                device=args.device,method=args.method,cam_weight=args.cam_weight)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


