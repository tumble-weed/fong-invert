# CUBLAS_WORKSPACE_CONFIG=:4096:8 python invert.py
# import register_ipdb
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
from total_variation import tv_norm
import torchvision.transforms
import torch.distributions as dist
from prediction_ranking import get_prediction_ranks,get_classname

import dutils
import hooks
from torch_memory_snippet import *

from hooks import setup_network

#==================================================================
tensor_to_numpy = lambda x: x.detach().cpu().numpy()
global MAJOR_PREFIX, MINOR_PREFIX
MAJOR_PREFIX = os.path.basename(__file__).split('.')[0]
MINOR_PREFIX = ''
opts = utils.MyNamespace()

torch.manual_seed(0)
torch.use_deterministic_algorithms(True,warn_only=True)
np.random.seed(0)
random.seed(0)

print = utils.printl
# losses
def alpha_prior(x, alpha=2.):
    return torch.abs(x.view(-1)**alpha).sum()

#==================================================================
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
def get_save_dir():
    # import ipdb;ipdb.set_trace()
    return os.path.join(utils.DEBUG_DIR,MAJOR_PREFIX,MINOR_PREFIX)
def main(image, network='alexnet', size=227, layer='features.4', alpha=6, beta=2, 
        alpha_lambda=1e-5,  tv_lambda=1e-5, epochs=200, learning_rate=1e2, 
        momentum=0.9, decay_iter=100, decay_factor=1e-1, print_iter=25, 
        device='cpu',method=None,target_class=None):
    #SET OPTS
    utils.cipdb('DBG_CLASS')
    opts.runner = "self"


    opts.image = image;del image
    if isinstance(opts.image,str):
        opts.image = [opts.image]
    opts.epochs = epochs;del epochs
    opts.network = network;del network
    opts.purge = False;
    opts.learning_rate = learning_rate;del learning_rate
    opts.layer = layer;del layer
    opts.print_iter = print_iter;del print_iter
    opts.target_class = target_class    
    opts.sync = False
    
    
    opts.observer_model = None
    utils.SYNC = opts.sync
    global MAJOR_PREFIX,MINOR_PREFIX
    # utils.SAVE_DIR = os.path.join(utils.DEBUG_DIR,'_'.join([MAJOR_PREFIX,MINOR_PREFIX]))
    # utils.SAVE_DIR = os.path.join(utils.DEBUG_DIR,MAJOR_PREFIX,MINOR_PREFIX)
    # import ipdb;ipdb.set_trace()
    utils.set_save_dir(get_save_dir(),purge=opts.purge)

    # import ipdb; ipdb.set_trace()
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
    ref = []
    if isinstance(opts.image,list):
        for imagei in opts.image:
            refi = transform(Image.open(imagei)).unsqueeze(0)
            ref.append(refi)
        ref = torch.cat(ref,dim=0)
    else:
        ref = transform(Image.open(opts.image)).unsqueeze(0)
    print(ref.size())
    ref = ref.to(device)
    def get_model(network,device):
        model = models.__dict__[network](pretrained=True)
        model.eval()
        model.to(device)
        return model
    model = get_model(opts.network,device)
    if opts.observer_model is not None:
        model2 = get_model(opts.observer_model,device)        
    
    opts.LOSS_MODE = 'max_gradcam_weights'
    opts.JITTER = False

    opts.ZOOM_JITTER = False
    opts.conv_layer_ixs = None


    if True:
        activations_last_mpool = []
        # import ipdb;ipdb.set_trace()
        if opts.network == 'vgg16':
            layer_name = 'features.30'
        else:
            assert False
        register_forward_hook(model,get_pytorch_module(model, layer_name),activations_last_mpool)
        register_backward_hook(get_pytorch_module(model, layer_name))



    ref_scores = 0

    if True:
        
        ref.requires_grad_(True)
        ref_scores = model(ref)
        
        # opts.target_class = ref_scores.argmax(dim=-1)
        # opts.target_class = [667]
        # opts.target_class = [5000000]
        
        # assert False
        
        if len(opts.image) == 1:
            assert len(activations_last_mpool) == 1
        ref_acts = activations_last_mpool[0].detach()
        ref_scores[:,opts.target_class[0]].sum().backward()
        ref_cam_grads = get_pytorch_module(model, layer_name).grads.detach().clone()
        ref_cam_weights = ref_cam_grads.mean(dim=(-1,-2)).squeeze(0)
    # assert False

    if method == 'fong':
        prng = np.random.RandomState(1)
        if False:
            x_ = (torch.tensor( 1e-3 * 
                prng.randn(*ref.size())
                )).float().to(device).requires_grad_(True)
        else:
            n_unique_x = len(opts.target_class) * len(opts.image)
            x__ = (torch.tensor( 
                prng.uniform(size = (n_unique_x,)+ref.shape[1:])
                )).float().to(device).requires_grad_(True)
        optimizer = torch.optim.SGD([x__], lr=opts.learning_rate, momentum=momentum)
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



    for i in range(opts.epochs):

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
            x_ = x__

            x_ = (x_- 0.5)*2*3
        
        if opts.JITTER:
            opts.jitter = 30
            lim_0,lim_1 = opts.jitter,opts.jitter
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            x_ = torch.roll(x_, shifts=(off1, off2), dims=(2, 3))                        

        # x_pre_noise = x_.clone()
        # x_pre_noise.retain_grad()
        # x_for_forward = torch.tile(x_pre_noise, (opts.n_noise,1,1,1))
        x_for_forward = x_
        #===================================================================
        x_scores = model(x_for_forward)
        trends['x_scores'].append(tensor_to_numpy(x_scores))        

        if opts.LOSS_MODE == 'max':
            if (len(opts.target_class) == 1) and (len(opts.image) == 1):
                print(colorful.brown(f"class: {x_scores.mean(dim=0).argmax()} target class {opts.target_class}" ))
        x_probs = torch.softmax(x_scores, dim=-1)


        alpha_term = alpha_prior(x_, alpha=alpha)
        trends['alpha_term'].append(alpha_term.item())
        #---------------------------------------------------------------
        print(colorful.red("add tv norm with gradient clipping"))
        print(colorful.red("add gradient clipping to the network gradients"))
        tv_norm_x,tv_map = tv_norm(x__, beta=beta)
        tv_term = tv_norm_x
        #---------------------------------------------------------------        
        # assert False
        if opts.LOSS_MODE == 'max_gradcam_weights':
            assert ref_cam_weights.ndim == 1
            CAM_W_MODE = 'average'
            if False:
                loss_term = -1*(ref_cam_weights[None,:,None,None]*activations_last_mpool[0]).sum()
            elif CAM_W_MODE == 'average':
                act = activations_last_mpool[0]
                w = torch.zeros_like(act[:1])
                w[0,:,act.shape[2]//2,act.shape[3]//2] = ref_cam_weights
                loss_term = -1*(w*activations_last_mpool[0]).sum()
            elif CAM_W_MODE == 'locationwise':
                w_mask = (ref_acts * ref_cam_grads).sum(dim=1) > 0
                w = ref_cam_grads * w_mask
                loss_term = -1*(w*activations_last_mpool[0]).sum()

        #===============================================================
        activations_last_mpool_ = (activations_last_mpool[0])
        if opts.network == 'alexnet':
            if not activations_last_mpool_.min() >= 0 :
                import ipdb; ipdb.set_trace()
        #===============================================================
        if method == 'fong':
            loss_lambda = 1
            opts.tv_lambda = 1e-2
            if CAM_W_MODE == 'locationwise':
                opts.tv_lambda = 1e0
            elif CAM_W_MODE == 'average':
                opts.tv_lambda = 1e-1
            else:
                assert False
            print(colorful.orange(f"using loss_lambda = {loss_lambda}"))
            # import ipdb;ipdb.set_trace()
            tot_loss = alpha_lambda*alpha_term + opts.tv_lambda*tv_term + loss_lambda*loss_term
        else:
            tot_loss = loss_term
        # tot_loss = loss_term
        #===============================================================
        cat = dist.Categorical(logits=x_scores)
        entropy = cat.entropy()
        if i == 0:
            # x_scores0 = x_scores
            cat0 = cat
        #===============================================================
        activations_last_mpool_ = (activations_last_mpool[0])
        #===============================================================
        trends['entropy'].append(entropy.sum().item())  

        trends['tv_term'].append(np.log(tv_term.item()))
        trends['loss_term'].append(loss_term.item())
        
        optimizer.zero_grad()
        tot_loss.backward(retain_graph=True)

        if method == 'fong':
            if x__.grad.isnan().any() or x__.grad.isinf().any():
                import ipdb; ipdb.set_trace()    
        
            
        #-------------------------------------------------------------------------
        #-------------------------------------------------------------------------


        optimizer.step()

        prediction_ranks,classnames = get_prediction_ranks(x_scores[0,...])
        # if i == epochs -1:
        #     import ipdb; ipdb.set_trace()
        if (i) % opts.print_iter == 0:
            SAVE_DONE =  [False]
            def save():
                utils.cipdb('DBG_CLASS_SAVE')
                if len(opts.target_class) == 1:
                    print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f\tprob/max_prob: %f/%f' % (i+1,
                        alpha_term.item(), tv_term.data.item(),
                        loss_term.item(), tot_loss.item(),
                        x_probs[0,opts.target_class].item(),x_probs[0].max(dim=-1)[0].item()
                        ))
                else:
                    print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f\t' % (i+1,
                        alpha_term.item(), tv_term.data.item(),
                        loss_term.item(), tot_loss.item(),

                        
                        ))
                    
                try:
                    if (len(opts.target_class) == 1) and (len(opts.image) == 1):
                        utils.img_save2(x__,f'x_iter_{i}.png')
                        utils.img_save2(x__,f'x.png',syncable=True)
                    for ii,target_class in enumerate(opts.target_class):
                        utils.img_save2(x__[ii:ii+1],f'x_{target_class}_iter_{i}.png')
                        utils.img_save2(x__[ii:ii+1],f'x_{target_class}.png',syncable=True) 
                    for ii in range(ref.shape[0]):
                        # get imroot from opts.image[ii]
                        imroot = os.path.basename(opts.image[ii]).split('.')[0]
                        utils.img_save2(x__[ii:ii+1],f'x_{imroot}_iter_{i}.png')
                        utils.img_save2(x__,f'x_{imroot}.png',syncable=True)                        
                        
                except Exception as e:
                    print(e)
                    raise e
                    # import ipdb;ipdb.set_trace()
                # import ipdb;ipdb.set_trace()
                if (len(opts.target_class) == 1) and (len(opts.image) == 1):
                    
                    

                    utils.img_save2(tv_map/tv_map.max(),f'tv_map_{i}.png')

                
                for lname in ['tv_term','loss_term','entropy','x_entropy','grad_alignment',
                              'loss_term0','max_adaptive_weight','mean_adaptive_weight','alpha_term']:
                        utils.save_plot2(trends[lname],lname,f'{lname}.png')
                for lname in ['loss_term','kl_div_observer','spearman_corr']:
                    utils.save_plot2(trends[lname],lname,f'{lname}.png',syncable=True)                    
                import pickle
                with open(os.path.join(utils.SAVE_DIR,'trends.pkl'),'wb') as f:
                    pickle.dump(trends,f)
                    pickle.dump(tensor_to_numpy(x__),f)
                    pickle.dump(opts.as_dict(),f)
                if False:
                    for lname in ['avg_angle','avg_mag']:
                        for li in range(len(avg_angles)):
                            utils.save_plot2(trends[f'{lname}_layer_{li}'],f'{lname}_layer_{li}',f'{lname}_layer_{li}.png')
                SAVE_DONE =  [False]
            if os.environ.get('DBG_SAVE',False):  
                save()    
            else:
                utils.run_in_another_thread(save)
            
        if decay_iter is not None:
            if (i+1) % decay_iter == 0:
                decay_lr(optimizer, decay_factor)
        if True and method == 'fong':
            print(colorful.red("move the clamp to before save"))
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
    if False:
        act_diff = get_region_importances(x__,model)
    # import ipdb; ipdb.set_trace()
    #===============================================
    if SAVE_DONE[0]:
        del trends
    del model
    import gc;gc.collect()
    #===============================================

def main2():
    import argparse
    import sys
    import traceback

    if True:
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
            if False:
                args.image = 'grace_hopper.jpg'
            elif False:
            
                # garter snake
                args.dataset = 'imagenet'
                args.image = '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000006.JPEG'
                args.target_class = [57]
            elif True:            
                # mouse trap
                args.dataset = 'imagenet'
                args.image = '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000009.JPEG'
                args.target_class = [674]            
                # args.target_class = None


            # args.network = 'alexnet'
            # args.network = 'vgg19'
            args.network = 'vgg16'
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
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            args.method = 'fong'
            
            if args.method == 'fong':
                args.decay_iter = None
                args.learning_rate = 1e-1
                args.epochs = 2000
                
            elif args.method == 'dip':
                args.decay_iter = None
                args.learning_rate = 1e-3
                args.epochs = 2000

        # gpu = args.gpu
        # cuda = True if gpu is not None else False
        # use_mult_gpu = isinstance(gpu, list)
        # if cuda:
        #     if use_mult_gpu:
        #         os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
        #     else:
        #         os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        # print(torch.cuda.device_count(), use_mult_gpu, cuda)
        # import ipdb; ipdb.set_trace()

        main(image=args.image, network=args.network, layer=args.layer, 
                alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda, 
                tv_lambda=args.tv_lambda, epochs=args.epochs,
                learning_rate=args.learning_rate, momentum=args.momentum, 
                print_iter=args.print_iter, decay_iter=args.decay_iter,
                decay_factor=args.decay_factor, 
                device=args.device,method=args.method,
                target_class = args.target_class
                )



if __name__ == '__main__':
    main2()