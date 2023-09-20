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
import tqdm
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
import torch.distributions as dist
from prediction_ranking import get_prediction_ranks,get_classname
from gradient_alignment import grad_alignment
import dutils
import hooks
from torch_memory_snippet import *

from hooks import setup_network
from sparsity import setup_network_for_comprehensive_sparsity
# import torch.cuda.amp as amp
from apex import amp
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
def get_save_dir():
    return os.path.join(utils.DEBUG_DIR,MAJOR_PREFIX,MINOR_PREFIX)
def main(image, network='alexnet', size=227, layer='features.4', alpha=6, beta=2, 
        alpha_lambda=1e-5,  tv_lambda=1e-5, epochs=200, learning_rate=1e2, 
        momentum=0.9, decay_iter=100, decay_factor=1e-1, print_iter=25, 
        device='cpu',method=None,cam_weight=None,use_fp16=False):
    #SET OPTS
    if os.environ.get('DBG_FP16',False) == '1':
        use_fp16 = True
    utils.cipdb('DBG_CLASS')
    opts.runner = "self"
    opts.cam_loss = True
    opts.cam_weight = 0
    opts.image = image;del image
    if isinstance(opts.image,str):
        opts.image = [opts.image]
    opts.epochs = epochs;del epochs
    opts.network = network;del network
    opts.purge = False;
    opts.learning_rate = learning_rate;del learning_rate
    opts.layer = layer;del layer
    opts.print_iter = print_iter;del print_iter
    opts.n_noise = 1
    opts.sync = False
    opts.noise_jitter = False
    opts.min_noise_mag = None
    opts.observer_model = None
    opts.use_fp16 = use_fp16; del use_fp16
    data_type = torch.half if opts.use_fp16 else torch.float
    utils.SYNC = opts.sync
    global MAJOR_PREFIX,MINOR_PREFIX
    # utils.SAVE_DIR = os.path.join(utils.DEBUG_DIR,'_'.join([MAJOR_PREFIX,MINOR_PREFIX]))
    # utils.SAVE_DIR = os.path.join(utils.DEBUG_DIR,MAJOR_PREFIX,MINOR_PREFIX)
    # import ipdb;ipdb.set_trace()
    # utils.cipdb('DBG_MATCH')
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
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        return model
    model = get_model(opts.network,device)
    if opts.observer_model is not None:
        model2 = get_model(opts.observer_model,device)        
    
    opts.LOSS_MODE = 'max'
    opts.JITTER = False
    opts.noise_mag = 1
    opts.noise_mags = None
    opts.ZOOM_JITTER = False
    opts.conv_layer_ixs = None
    # if os.environ.get('DBG_LEAK') == '1':
    if False:
        pass
    else:
        gradcam_layer = 'features.11'
        activations_gradcam,grads_gradcam = [] , []
        setup_network(model,get_pytorch_module(model, gradcam_layer),activations_gradcam,grads=grads_gradcam,backward=True)
        if True:
            activations = []
            if opts.LOSS_MODE == 'match':
                setup_network(model,get_pytorch_module(model, opts.layer),activations)
            # setup for gradcam
            # default gradcam layer for alexnet
        

            # import ipdb; ipdb.set_trace()
            # hooks.assets['backward',gradcam_layer].append(grads_gradcam)
            # hooks.assets['forward',gradcam_layer].append(activations_gradcam)
            TRACK_LAST_MPOOL_SPARSITY = False
            GET_CAM = False
            if GET_CAM or TRACK_LAST_MPOOL_SPARSITY:
                activations_last_mpool = []
                register_forward_hook(model,get_pytorch_module(model, 'features.12'),activations_last_mpool)
            activations_sparsity,noisy_activations_sparsity,avg_angles,avg_mags = [],[],[],[]
            utils.cipdb('DBG_ABLATION')
            # import ipdb; ipdb.set_trace()
            setup_network_for_comprehensive_sparsity(model.features,
                                                    activations_sparsity,noisy_activations_sparsity,avg_angles,avg_mags,
                                                    layer_type=nn.Conv2d,
                                                    noise_mag=opts.noise_mag,
                                                    layer_ixs = opts.conv_layer_ixs,
                                                    noise_mags=opts.noise_mags,
                                                    angular_simplicity=False)
            print('see which layers are noisy')
            utils.cipdb('DBG_MATCH')

        pass

    utils.cipdb('DBG_LEAK')
    if os.environ.get('REF_EDGE_ALIGN',False):
        ref_edge_mags,ref_edge_dirs = channelwise_edge_gradients(ref)
        ref_edge_alignment =  get_edge_alignment(ref_edge_mags)
        # import ipdb;ipdb.set_trace()
        for chan in range(3):
            utils.img_save2(ref_edge_mags[:,chan:chan+1].cpu().numpy(), f'ref_edge_mags_{chan}.png')
    ref_scores = 0
    if opts.LOSS_MODE == 'match':
        original_noise_flag = model.features.noise_flag
        model.features.noise_flag = False
        ref_scores = model(ref)
        model.features.noise_flag = original_noise_flag
        utils.cipdb('DBG_LEAK')

        # opts.target_class = ref_scores.argmax(dim=-1)
        opts.target_class = [10001]
        # opts.target_class = [5000000]
        
        # assert False
        
        if len(opts.image) == 1:
            assert len(activations) == 1
        ref_acts = activations[0].detach()
        # import ipdb; ipdb.set_trace()
        # ref_acts = torch.tile(ref_acts, (opts.n_noise,) + tuple([1 for _ in ref_acts.shape[1:]]))
    else:
        opts.target_class = [200]
        print('target class: ',opts.target_class)
    if os.environ.get('DBG_LEAK',False):
        #===============================================
        del trends
        hooks.clear_hooks()
        import sparsity
        sparsity.clear_hooks()
        del model
        import gc;gc.collect()
        import ipdb; ipdb.set_trace()
        #===============================================    
        return
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
                ,dtype = data_type
                )).to(device).requires_grad_(True)
            # import ipdb;ipdb.set_trace()
        optimizer = torch.optim.SGD([x__], lr=opts.learning_rate, momentum=momentum)
        if opts.use_fp16:
            static_loss_scale = 256
            static_loss_scale = "dynamic"
            # _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)
            model, optimizer = amp.initialize(model, optimizer,
                                    #   opt_level=args.opt_level,
                                    #   keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                    #   loss_scale=args.loss_scale
                                      )

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

    opts.NOISE_ANNEAL = False
    utils.cipdb('DBG_LEAK')
    for i in tqdm.tqdm(range(opts.epochs)):
        print(colorful.green(get_save_dir()))
        utils.cipdb('DBG_CLASS')
#             acts = get_acts(model, x_)
        if opts.NOISE_ANNEAL:
            assert False,'not compatible with only layer 1 noise'
            print(colorful.red("annealing noise"))
            model.features.noise_mag = 1 * int(i < 500) + 0.5 * int(i >= 500)
            if i == 0:
                original_learning_rate = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = original_learning_rate * int(i < 500) + original_learning_rate*0.5 * int(i >= 500)
        del activations_sparsity[:]
        del noisy_activations_sparsity[:]        
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
            x_ = x__
            if False and 'inbuilt color jitter':
                color_jitter = transforms.ColorJitter(
                                                        # brightness=0.5*1,
                                                        # contrast=0.5*1,
                                                        # saturation=0.5*1,
                                                        hue=0.5
                                                        )            
                x_ = color_jitter(x_)
                x_ = x_.contiguous()
            x_ = (x_- 0.5)*2*3
        
        if opts.JITTER:
            opts.jitter = 30
            lim_0,lim_1 = opts.jitter,opts.jitter
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            x_ = torch.roll(x_, shifts=(off1, off2), dims=(2, 3))                        
        # model.features.noise_mag = 0
        
            # x_ = jitter_color(x_, 0.1)
        # import ipdb; ipdb.set_trace()                 
        x_pre_noise = x_.clone()
        x_pre_noise.retain_grad()
        x_for_forward = torch.tile(x_pre_noise, (opts.n_noise,1,1,1))
        # if True and 'custom color jitter':
        #     x_for_forward = jitter_color(x_for_forward, 0.05)

        opts.MAX_SCALE=1.2
        if opts.ZOOM_JITTER and 'zoom jitter':
            # zoom by random scale and random crop
            if i == 0:
                prng_zoom = np.random.RandomState(1)
            random_scale = prng_zoom.uniform(1, opts.MAX_SCALE)
            zoomed = torch.nn.functional.interpolate(x_for_forward, scale_factor=random_scale, mode='bilinear', align_corners=False)
            indicators = torch.ones_like(x_for_forward)[:1].requires_grad_(True)
            zoomed_indicators = torch.nn.functional.interpolate(indicators, scale_factor=random_scale, mode='bilinear', align_corners=False)
            #random crop zoomed to original size
            if  zoomed.shape[2] > ref.shape[2] or zoomed.shape[3] > ref.shape[3]:
                try:
                    top = prng_zoom.randint(0, zoomed.shape[2]-ref.shape[2],size=(zoomed.shape[0]))
                    left = prng_zoom.randint(0, zoomed.shape[3]-ref.shape[3],size=(zoomed.shape[0]))
                    x_for_forward = torch.stack([zoomed[i,:,left[i]:left[i]+ref.shape[2],top[i]:top[i]+ref.shape[3]] for i in range(zoomed.shape[0])])
                    indicator_crops = torch.stack([zoomed_indicators[0,:,left[i]:left[i]+ref.shape[2],top[i]:top[i]+ref.shape[3]] for i in range(zoomed.shape[0])])
                    indicator_crops.sum().backward()
                    indicator_weights = indicators.grad.detach().clone()
                    def _hook(grad):
                        reweight = indicator_weights / indicator_weights.max()
                        reweight[reweight == 0] = 1
                        return grad / reweight
                    x_pre_noise.register_hook(_hook)
                except ValueError:
                    import ipdb; ipdb.set_trace()
        if all([noisy_layer.noise_mag == 0 for noisy_layer in model.features.noisy_layers]):
            print(colorful.red("No Noise"))
        # utils.cipdb('DBG_NOISE_JITTER')
        if opts.noise_jitter and (opts.noise_mag != 0):
            # sample a float uniformly etween min_noise_mag and noise_mag
            noise_mag = np.random.uniform(opts.min_noise_mag, opts.noise_mag)
            for layer in model.features.noisy_layers:
                layer.noise_mag = noise_mag       
        if opts.observer_model is not None:
            from observer import validate_with_observer
            #=====================================================
            original_noise_mags = []
            for noisy_layer in model.features.noisy_layers:
                original_noise_mags.append(noisy_layer.noise_mag)
                noisy_layer.noise_mag = 0
            # import ipdb; ipdb.set_trace()
            x_for_kl_div = (x__- 0.5)*2*3
            kl_div_observer,spearman_corr = validate_with_observer(model,model2,x_for_kl_div)
            for orig_mag,noisy_layer in zip(original_noise_mags,model.features.noisy_layers):
                noisy_layer.noise_mag = orig_mag
                
            #=====================================================
            # import ipdb;ipdb.set_trace()
            trends['kl_div_observer'].append(kl_div_observer.mean().item())
            trends['spearman_corr'].append(spearman_corr)
        #===================================================================
        x_scores = model(x_for_forward)
        # print('see shape of x_scores')
        # utils.cipdb('DBG_MATCH')
        trends['x_scores'].append(tensor_to_numpy(x_scores))
        # utils.cipdb("DBG_OBSERVER")

        
        # print(x_probs0[:,opts.target_class],x_probs1[:,opts.target_class])
        try:
            for li,(avg_angle,avg_mag) in enumerate(zip(avg_angles,avg_mags)):
                trends[f'avg_angle_layer_{li}'].append(avg_angle[0].item())
                trends[f'avg_mag_layer_{li}'].append(avg_mag[0].item())        
        except IndexError:
            pass
        if opts.LOSS_MODE == 'max':
            if (len(opts.target_class) == 1) and (len(opts.image) == 1):
                print(colorful.brown(f"class: {x_scores.mean(dim=0).argmax()} target class {opts.target_class}" ))
        x_probs = torch.softmax(x_scores, dim=-1)
        if activations:
            acts = activations[0]
        gradcam_acts = activations_gradcam[0]
        alpha_term = alpha_prior(x_, alpha=alpha)
        #---------------------------------------------------------------
        print(colorful.red("add tv norm with gradient clipping"))
        print(colorful.red("add gradient clipping to the network gradients"))
        tv_norm_x,tv_map = tv_norm(x__, beta=beta)
        tv_norm_trunc_x,tv_trunc_map = tv_norm_trunc(x__, beta=beta, T = 0.15)
        if x__.shape[0] == 1:
            tv_norm_trunc2_x,tv_trunc2_map = tv_norm_trunc2(
            # x__.detach().clone().requires_grad_(True), 
            x__,
            beta=beta, tv_lambda=tv_lambda,ratio = 0.5)
        tv_term = tv_norm_x
        # tv_term = tv_norm_trunc_x
        # tv_term = tv_norm_trunc2_x
        #---------------------------------------------------------------
        if False:
            x_edge_mags,x_edge_dirs = channelwise_edge_gradients(x__)
            x_edge_alignment =  get_edge_alignment(x_edge_mags)        
            if (i+1)%print_iter == 0:
                print(x_edge_alignment)
        #---------------------------------------------------------------
        if os.environ.get('DBG_MATCH',False):
            import ipdb;ipdb.set_trace()
        if opts.LOSS_MODE == 'match':
            # loss_term = norm_loss(acts, ref_acts)
            # return torch.abs(x.view(-1)**alpha).sum()
            assert (opts.image.shape[1:] == (3,224,224))
            """
            if len(opts.image) > 1:
                import ipdb; ipdb.set_trace()
            """
            # if opts.runner == "self":
            if opts.get("adaptive_match",False):
                # loss_term = torch.div(((acts.mean(dim=0,keepdim=True) - ref_acts)**2).sum(), ((ref_acts)**2).sum())
                if i == 0:
                    starting_delta = (acts.mean(dim=0,keepdim=True) - ref_acts)**2
                    starting_delta = starting_delta.detach()
                    prev_relative_remaining = 1
                loss_numer = (acts.mean(dim=0,keepdim=True) - ref_acts)**2
                relative_remaining = (loss_numer/(1 + starting_delta)).detach()
                relative_remaining = relative_remaining * torch.ones_like(relative_remaining).sum()/relative_remaining.sum()
                prev_relative_remaining = (prev_relative_remaining*0.99 + relative_remaining*0.01)
                # if opts.get
                # prev_relative_remaining = torch.clip(prev_relative_remaining,0,10)
                loss_term = (loss_numer*prev_relative_remaining).sum() * (1/((ref_acts)**2).sum())
                loss_term0 = torch.div(((acts.mean(dim=0,keepdim=True) - ref_acts)**2).sum(), ((ref_acts)**2).sum())
                trends['loss_term0'].append(loss_term0.item())
                trends['max_adaptive_weight'].append(prev_relative_remaining.max().item())
                trends['mean_adaptive_weight'].append(prev_relative_remaining.mean().item())
            else:
                import ipdb; ipdb.set_trace()
                for ii in range(opts.image):
                    loss_term_ii = torch.div(((acts[range(ii,acts.shape[0],len(opts.image))].mean(dim=0,keepdim=True) - ref_acts[ii:ii+1])**2).sum(), ((ref_acts[ii:ii+1])**2).sum())
                
            
        elif opts.LOSS_MODE == 'max':
            assert len(opts.target_class) == x__.shape[0]
            target_classes = [c for k in range(opts.n_noise) for c in opts.target_class ]
            # torch.tile(torch.tensor([1,2]),4) = tensor([1, 2, 1, 2, 1, 2, 1, 2])
            loss_term = -x_scores[np.arange(len(target_classes)),target_classes].view(len(opts.target_class),opts.n_noise,-1).mean(dim=1).sum()
        #===============================================================
        if TRACK_LAST_MPOOL_SPARSITY:
            activations_last_mpool_ = (activations_last_mpool[0])
            if opts.network == 'alexnet':
                if not activations_last_mpool_.min() >= 0 :
                    import ipdb; ipdb.set_trace()
            last_mpool_sparsity = activations_last_mpool_.abs().sum()
        #===============================================================
        if False:
            sparsities = []
            for activations_i in activations_sparsity:
                sparsities.append(activations_i[0].abs().sum())
            # sparsity_lambdas = [0.0005 for _ in sparsities]
            sparsity_lambdas = [0 for _ in sparsities]
            if False:
                sparsity_lambdas[0] = 0.001
                # sparsity_lambdas[1] = 0.001
                # sparsity_lambdas[2] = 0.005
                # sparsity_lambdas[3] = 0.01
            elif False and i > 100:
                sparsity_lambdas[0] = 0.001
                # sparsity_lambdas[1] = 0.001
                # sparsity_lambdas[2] = 0.005
                # sparsity_lambdas[3] = 0.01
            elif True:
                # sparsity_lambdas[0] = 0.001 * float(i< 100)
                sparsity_lambdas[2] = 0.005 * float(i< 100)
            sparsity = sum([sparsity_lambdas[i]*sparsities[i] for i in range(len(sparsities))])
        else:
            sparsity = 0
            
        
        # import ipdb; ipdb.set_trace()

        score_sparsity = x_scores.abs().sum()
        if method == 'fong':
            # score_sparsity_lambda = 0.009
            score_sparsity_lambda = 0.001 * 0.1 * 0
            loss_lambda = 1
            # cam_weight = 5*0
            last_mpool_sparsity_lambda = 0*0.001
            opts.tv_lambda = 1e-1
            # if dutils.get('score_sparsity_ablation',False):
            #     score_sparsity_lambda = dutils.get('score_sparsity_lambda',None)
            # import ipdb;ipdb.set_trace()
            # score_sparsity_lambda = dutils.getif('score_sparsity_lambda',None,score_sparsity_lambda,'score_sparsity_ablation')
            
            # if opts.target_class == 153:
            #     loss_lambda = 1.5
            print(colorful.orange(f"using score_sparsity_lambda = {score_sparsity_lambda}"))
            print(colorful.orange(f"using loss_lambda = {loss_lambda}"))
            if False and 'last_mpool_sparsity':
                tot_loss = alpha_lambda*alpha_term + tv_lambda*tv_term + loss_lambda*loss_term + score_sparsity_lambda*score_sparsity + last_mpool_sparsity_lambda*last_mpool_sparsity
            else:
                tot_loss = alpha_lambda*alpha_term + opts.tv_lambda*tv_term + loss_lambda*loss_term + score_sparsity_lambda*score_sparsity + sparsity
        else:
            tot_loss = loss_term
        # tot_loss = loss_term
        #===============================================================
        if i > 0:
            cat_prev = cat
        cat = dist.Categorical(logits=x_scores)
        entropy = cat.entropy()
        if i == 0:
            # x_scores0 = x_scores
            cat0 = cat
        if i>0:
            kl_from_start = dist.kl_divergence(cat, cat0)
            kl_from_prev = dist.kl_divergence(cat, cat_prev) 
        #===============================================================
        if TRACK_LAST_MPOOL_SPARSITY:
            if i >0 :
                activations_last_mpool_prev_ =   activations_last_mpool_
            activations_last_mpool_ = (activations_last_mpool[0])
            if i >0 :
                change_in_mpool_from_prev = (activations_last_mpool_ - activations_last_mpool_prev_).norm(2)
                trends['change_in_mpool_from_prev'].append(change_in_mpool_from_prev.item())  

        #===============================================================
        trends['entropy'].append(entropy.sum().item())  
        if i >0:      
            trends['kl_from_start'].append(np.log(kl_from_start.sum().item()))
            trends['kl_from_prev'].append(kl_from_prev.sum().item())        
        trends['tv_term'].append(np.log(tv_term.item()))
        trends['loss_term'].append(loss_term.item())
        
        optimizer.zero_grad()
        # tot_loss.backward(retain_graph=True)
        if opts.use_fp16:
            # optimizer.backward(loss)
            with amp.scale_loss(tot_loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            tot_loss.backward(retain_graph=True)

        

        if len(x__) == 1:
            grad_alignment_ = grad_alignment(x_pre_noise.grad,name='loss_term')
            trends['grad_alignment'].append(grad_alignment_)
        if method == 'fong':
            if x__.grad.isnan().any() or x__.grad.isinf().any():
                import ipdb; ipdb.set_trace()    
        if 'gradcam':
            
            # weights = torch.mean(grads_gradcam[0], dim=(2, 3), keepdim=True)
            weights = torch.mean(gradcam_acts.grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * gradcam_acts, dim=1, keepdim=True)
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
        opts.simonyan_saliency = True
        if opts.simonyan_saliency:
            x_for_simonyan = x_for_forward.detach().clone().requires_grad_(True)
            x_scores_for_simonyan = model(x_for_simonyan)
            loss_simonyan = - x_scores_for_simonyan[:,opts.target_class].sum()
            loss_simonyan.backward(retain_graph=True)
            saliency = x_for_simonyan.grad.abs().sum(dim=1,keepdim=True).mean(dim=0,keepdim=True)
        else:
            saliency = torch.zeros_like(x__)
            
        #-------------------------------------------------------------------------
        if i>0 and 'fg_remove entropy' and False:
            from elp_masking import get_masked_input
            
            cam0_nrmz = cam0/(cam0.max() + (cam0.max() == 0).float())
            
            cam0_nrmz_up_float = torch.nn.functional.interpolate(cam0_nrmz, size=x_.size()[-2:], mode='bilinear', align_corners=False)
            cam0_nrmz_up = (cam0_nrmz_up_float>0.5).float()
            # import ipdb; ipdb.set_trace()
            bg,_ = get_masked_input(
                                        x_,
                                        # 1-torch.sigmoid( 100 * (cam_nrmz_up - 0.5)) ,
                                        1 - cam0_nrmz_up_float.detach()*cam0_nrmz_up,
                                        # perturbation=BLUR_PERTURBATION,
                                        num_levels=8,
                                        # variant=PRESERVE_VARIANT,
                                        smooth=0)
            bg = bg* cam0_nrmz_up +  (bg * (1 - cam0_nrmz_up)).detach()
            bg_scores = model(bg)
            bg_probs = torch.softmax(bg_scores,dim=-1)
            bg_entropy = -torch.sum(bg_probs * torch.log(bg_probs), dim=-1)  # Compute the entropy of the batch
            x_entropy = -torch.sum(x_probs * torch.log(x_probs), dim=-1)  # Compute the entropy of the batch
            if False:
                print(colorful.purple(f"bg class {bg_scores.argmax(dim=-1).item()},{bg_probs.max().item()}"))
            '''
            entropy_fg_removed = 
            
            logits = torch.randn(10, 5)  # Example batch of logits with shape (batch_size, num_classes)
            probs = F.softmax(logits, dim=-1)  # Compute the softmax probabilities
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # Compute the entropy of the batch
            '''
            trends['bg_entropy'].append(bg_entropy.sum().item())
            trends['x_entropy'].append(x_entropy.sum().item())
        #-------------------------------------------------------------------------
        if 'cam loss' and False:
            if True:
                cam_norm = cam.sum()
                cam0_norm = cam0.sum()
                cam0_sparsity = (cam0>0).float().sum()
                # import ipdb; ipdb.set_trace()
                cam_loss = opts.cam_weight*cam_norm + 0*10*tv_cam
                if i > 0:
                    cam_loss = opts.cam_weight*cam_norm + 10*tv_cam + 0*bg_entropy.sum()
            else:
                print(colorful.red("using squared sum for cam loss"))
                cam_loss = cam_weight*(cam**2).sum()
            # import ipdb; ipdb.set_trace()
            if opts.cam_loss:
                # import ipdb; ipdb.set_trace()
                cam_loss.backward()
        
            trends['cam0_norm'].append(cam0_norm.item())
            trends['cam0_sparsity'].append(cam0_sparsity.item())
            trends['cam_norm'].append(cam_norm.item())
        trends['cam_tv'].append(tv_cam.item())

            
        optimizer.step()

        prediction_ranks,classnames = get_prediction_ranks(x_scores[0,...])
        # if i == epochs -1:
        #     import ipdb; ipdb.set_trace()
        if (i) % opts.print_iter == 0:
            SAVE_DONE =  [False]
            def save():
                utils.cipdb('DBG_CLASS_SAVE')
                if len(opts.target_class) == 1:
                    print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f\tCAM: %f\tprob/max_prob: %f/%f' % (i+1,
                        alpha_term.item(), tv_term.data.item(),
                        loss_term.item(), tot_loss.item(),
                        cam.sum().item(),
                        x_probs[0,opts.target_class].item(),x_probs[0].max(dim=-1)[0].item()
                        ))
                else:
                    print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f\tCAM: %f\t' % (i+1,
                        alpha_term.item(), tv_term.data.item(),
                        loss_term.item(), tot_loss.item(),
                        cam.sum().item(),
                        
                        ))
                    
                try:
                    if (len(opts.target_class) == 1) and (len(opts.image) == 1):
                        utils.img_save2(x__,f'x_iter_{i}.png')
                        utils.img_save2(x__,f'x.png',syncable=True)
                    for ii,target_class in enumerate(opts.target_class):
                        utils.img_save2(x__[ii:ii+1],f'x_{target_class}_iter_{i}.png')
                        utils.img_save2(x__[ii:ii+1],f'x_{target_class}.png',syncable=True) 
                    import torchvision.utils as vutils
                    grid= vutils.make_grid(x__,nrows = int(x__.shape[0]**0.5),normalize=False,scale_each=False)
                    
                    utils.img_save2(grid,f'x_iter_{i}.png')
                    utils.img_save2(grid,f'x.png',syncable=True)                 
                    # import ipdb;ipdb.set_trace()    
                    """
                    for ii in range(ref.shape[0]):
                        # get imroot from opts.image[ii]
                        imroot = os.path.basename(opts.image[ii]).split('.')[0]
                        utils.img_save2(x__[ii:ii+1],f'x_{imroot}_iter_{i}.png')
                        utils.img_save2(x__,f'x_{imroot}.png',syncable=True)                        
                    """

                except Exception as e:
                    import ipdb;ipdb.set_trace()
                # import ipdb;ipdb.set_trace()
                if (len(opts.target_class) == 1) and (len(opts.image) == 1):
                    if i > 0:
                        utils.img_save2(bg,f'bg_{i}.png')
                    utils.img_save2(saliency,f'saliency_{i}.png')
                    utils.img_save2(saliency,f'saliency.png')
                    
                    cam_up = torch.nn.functional.interpolate(cam, size=ref.size()[-2:], mode='bilinear', align_corners=False)
                    cam_up = cam_up/( cam_up.max() + (cam_up.max() == 0).float())
                    # cam_up = (cam_up>0).float()
                    utils.img_save2(cam_up,f'gradcam_{i}.png')
                    utils.img_save2(cam_up,f'gradcam.png')
                    
                    cam0_up = torch.nn.functional.interpolate(cam0, size=ref.size()[-2:], mode='bilinear', align_corners=False)
                    cam0_up = cam0_up/( cam0_up.max() + (cam0_up.max() == 0).float())
                    # cam_up = (cam_up>0).float()
                    utils.img_save2(cam0_up,f'gradcam0_{i}.png')

                    utils.img_save2(tv_map/tv_map.max(),f'tv_map_{i}.png')
                    utils.img_save2(tv_trunc_map/tv_trunc_map.max(),f'tv_trunc_map_{i}.png')
                
                for lname in ['cam0_norm','cam0_sparsity','cam_norm','cam_tv','tv_term','loss_term','entropy','kl_from_start','kl_from_prev','change_in_mpool_from_prev','bg_entropy','x_entropy','grad_alignment',
                              'loss_term0','max_adaptive_weight','mean_adaptive_weight']:
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
    hooks.clear_hooks()
    import sparsity
    sparsity.clear_hooks()
    del model
    import gc;gc.collect()
    #===============================================
    # for k in list(sparsity.hook_dict.keys()):
    #     del sparsity.hook_dict[k]
    #     del sparsity.assets[k]
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
            args.image = 'grace_hopper.jpg'
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
                # if dutils.get('score_sparsity_ablation',False):
                #     args.epochs = dutils.get('epochs',args.epochs)
                # args.epochs = dutils.getif('epochs',None,args.epochs,'score_sparsity_ablation')
                # import ipdb;ipdb.set_trace()
                
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
        # import ipdb; ipdb.set_trace()
        utils.cipdb('DBG_LEAK1')
        main(image=args.image, network=args.network, layer=args.layer, 
                alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda, 
                tv_lambda=args.tv_lambda, epochs=args.epochs,
                learning_rate=args.learning_rate, momentum=args.momentum, 
                print_iter=args.print_iter, decay_iter=args.decay_iter,
                decay_factor=args.decay_factor, 
                device=args.device,method=args.method,cam_weight=args.cam_weight)



if __name__ == '__main__':
    main2()