# CUBLAS_WORKSPACE_CONFIG=:4096:8 python invert.py
import register_ipdb
import torch
import torch.nn as nn
import colorful
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.models as models
import torchvision.utils as vutils
import numpy as np

import os
import utils
import matplotlib.pyplot as plt

from PIL import Image
#==================================================================
# from dip.common_utils import get_noise#,get_params
# from dip.common_utils import tv_loss
# from dip.models import skip
from collections import defaultdict
#==================================================================
from hooks import get_pytorch_module,register_backward_hook,register_forward_hook
from diversity import get_region_importances
import random
from elp_masking import get_masked_input
import torch
from total_variation import tv_norm,tv_norm_trunc
# from saliency_utils import GaussianSmoothing,SoftMaxFilter2d,multi_thresh_sigmoid
import saliency_utils
from utils import MyNamespace
import fong_utils
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



def get_save_dir():
    return os.path.join(utils.DEBUG_DIR,MAJOR_PREFIX,MINOR_PREFIX)
def main(ref, model, network=None,size=227, layer='features.4', alpha=6, beta=2, 
        alpha_lambda=1e-5,  tv_lambda=1e-5, epochs=200, learning_rate=1e2, 
        momentum=0.9, decay_iter=100, decay_factor=1e-1, print_iter=25, 
        device='cpu',method=None,target_class=None,dataset=None,
        mode = 'saliency',feat_layer=None,detransform=None,n_areas = 40,
        ref2=None):
    # SET OPTS
    # import ipdb;ipdb.set_trace()
    # opts.image = image;del image
    opts.epochs = epochs;del epochs
    opts.network = network;del network
    opts.feat_layer = feat_layer;del feat_layer
    opts.purge = False;
    opts.print_iter = print_iter;del print_iter
    opts.sync = False
    opts.mode = mode;del mode
    opts.n_areas = n_areas;del n_areas
    # opts.n_areas = 40
    opts.target_class = target_class; del target_class
    opts.loss_mode = 'l2_max_score'
    # opts.loss_mode = 'neg_score'
    # opts.loss_mode = 'switched_neg_score'
    opts.dataset = dataset;del dataset
    opts.sharpness = 20
    opts.area_l = 2
    opts.UTILIZE_T_GRAD = True

    utils.SYNC = opts.sync
    global MAJOR_PREFIX,MINOR_PREFIX
    utils.set_save_dir(get_save_dir(),purge=opts.purge)    
    utils.SYNC_DIR = utils.SAVE_DIR
    trends = defaultdict(list)

    # ref,bbox_info,target_ids,classnames,detransform = fong_utils.get_image(opts.image,opts.dataset)
    # if opts.target_class is None:
    #     opts.target_class = target_ids[0]    
    ref = ref.to(device)
    ref_scores = model(ref)
    if opts.mode in ['self-saliency','cosaliency']:
        ref_feats = opts.feat_layer.feats.detach().clone()
        if opts.mode == 'cosaliency':
            ref2 = ref2.to(device)
            ref_scores2 = model(ref2)
            ref_feats2 = opts.feat_layer.feats.detach().clone()
    # target_class = ref_scores.argmax(dim=-1)
    
    print('target class: ',opts.target_class)
    
    #==========================================================
    prng = np.random.RandomState(2)
    # if False:
    #     x_ = (torch.tensor( 1e-3 * 
    #         prng.randn(*ref.size())
    #         )).float().to(device).requires_grad_(True)
    # else:
    #     x__ = (torch.tensor( 
    #         prng.uniform(size = ref.size())
    #         )).float().to(device).requires_grad_(True)
    window_size = 41
    def get_mask_logit(window_size=41):
        initial_mask_logit_mag = 1
        mask_logit_full_shape = (1,1) + (ref.shape[-2]+ 2*(window_size//2),
                                ref.shape[-1]+ 2*(window_size//2))

        prng = np.random.RandomState(initial_mask_logit_mag)
        mask_logit_full_ = initial_mask_logit_mag*prng.normal(size=(mask_logit_full_shape))
        mask_logit_full = torch.tensor(mask_logit_full_).float().to(device).requires_grad_(True)
        return mask_logit_full
    mask_logit_full = get_mask_logit(window_size=window_size)
    if opts.mode == 'cosaliency':
        mask_logit_full2 = get_mask_logit(window_size=window_size)
    if opts.mode == "class-transition":
        x__ = (torch.tensor( 
            prng.uniform(size = ref.size())
            )).float().to(device).requires_grad_(True)
        optimizer = torch.optim.SGD([mask_logit_full,x__], lr=learning_rate, momentum=momentum)
    elif opts.mode in ["self-saliency","saliency"]:
        optimizer = torch.optim.SGD([mask_logit_full], lr=learning_rate, momentum=momentum)
    elif opts.mode in ["cosaliency"]:
        optimizer = torch.optim.SGD([mask_logit_full,mask_logit_full2], lr=learning_rate, momentum=momentum)
    #==========================================================
    from utils import TrackChange
    change_in_mask_logit_full = TrackChange(mask_logit_full,name='mask_logit_full')

    for i in range(opts.epochs):
        if 'saliency':
            def get_max_of_smooth_mask(mask_logit_full,window_size):            
                smoothen = saliency_utils.GaussianSmoothing(1, (window_size,window_size),     
                (window_size - 1)/4, 
                dim=2,device=device)
                max_filter = saliency_utils.SoftMaxFilter2d((window_size,window_size),1,(1,1),
                                            padder =  'reflection',
                    # sharpness=1e-1
                    # sharpness=1e1
                    sharpness=1e0
                    ).to(device)
                smoothed_mask = smoothen(mask_logit_full)
                if True:
                    max_of_smooth_mask = max_filter( smoothed_mask)
                elif False:
                    print(colorful.yellow_on_red('skipping MAX after smooth'))
                    max_of_smooth_mask = ( smoothed_mask)
                return max_of_smooth_mask
            max_of_smooth_mask = get_max_of_smooth_mask(mask_logit_full,window_size)
            max_of_smooth_mask_rep = torch.cat([max_of_smooth_mask.clone() for _ in range(opts.n_areas)],dim=0)
            #===========================================================================
            if opts.mode == 'cosaliency':
                max_of_smooth_mask2 = get_max_of_smooth_mask(mask_logit_full2,window_size)
                max_of_smooth_mask2_rep = torch.cat([max_of_smooth_mask2.clone() for _ in range(opts.n_areas)],dim=0)
            #===========================================================================
            if opts.mode == "class-transition":
                x_ = (x__- 0.5)*2*3                

            # import areas_jax as areas
            # import areas_scipy as areas
            import areas_torch as areas
            required_areas=torch.linspace(0,1*np.prod(max_of_smooth_mask.shape[-2:]),opts.n_areas).to(device)
            if 'prev_thresholds' not in locals():
                prev_thresholds = None
            thresholds = areas.get_thresholds(max_of_smooth_mask,
                            sharpness = opts.sharpness,
                           required_areas=required_areas,l=opts.area_l,mode='cumsum',t_prev=prev_thresholds)
            prev_thresholds = thresholds.detach().clone()
            

            multi_mask = saliency_utils.multi_thresh_sigmoid(max_of_smooth_mask_rep,thresholds,opts.sharpness,trends)
            if opts.mode == 'cosaliency':
                thresholds2 = areas.get_thresholds(max_of_smooth_mask2,
                                sharpness = opts.sharpness,
                            required_areas=required_areas,l=opts.area_l,mode='cumsum')
                multi_mask2 = saliency_utils.multi_thresh_sigmoid(max_of_smooth_mask2_rep,thresholds2,opts.sharpness,trends)
            
            if opts.area_l == 1:
                obs_areas =multi_mask.sum(dim=(1,2,3)) 
            elif opts.area_l == 2:
                obs_areas = (multi_mask**2).sum(dim=(1,2,3)) 
            if ((obs_areas[1:] - obs_areas[:-1]) < 0).any():
                import pdb;pdb.set_trace()            
            trends['observed_areas'].append(tensor_to_numpy(obs_areas))
            trends['area_error'].append(tensor_to_numpy(obs_areas - required_areas).sum())
            # masked = ref * multi_mask
            # from elp_masking import get_masked_input,DELETE_VARIANT,PRESERVE_VARIANT,laplacian_pyramid_blending
            import elp_masking
            if opts.mode in ['saliency','self-saliency','cosaliency']:
                masking_variant = elp_masking.PRESERVE_VARIANT
                # masking_variant = elp_masking.DELETE_VARIANT if opts.mode == 'cosaliency' else elp_masking.PRESERVE_VARIANT
                masked,_ = elp_masking.get_masked_input(
                            ref,
                            multi_mask,
                            # perturbation=BLUR_PERTURBATION,
                            num_levels=8,

                            variant=masking_variant,
                            # variant=elp_masking.DELETE_VARIANT,
                            smooth=0)            
                if opts.mode == 'cosaliency':
                    masked2,_ = elp_masking.get_masked_input(
                                ref2,
                                multi_mask2,
                                # perturbation=BLUR_PERTURBATION,
                                num_levels=8,

                                variant=elp_masking.PRESERVE_VARIANT,
                                # variant=elp_masking.DELETE_VARIANT,
                                smooth=0)                                
            elif opts.mode == 'class-transition':
                masked = elp_masking.laplacian_pyramid_blending(x_, ref, multi_mask, num_levels=6)

            # show_scores = masked_forward(model,ref,mask_multi_ste,masking_options=OPTIONS
        if i == 0:
            change_in_multi_mask = TrackChange(multi_mask,name='change_in_multi_mask')
            change_in_masked = TrackChange(masked,name='change_in_masked')            
        # import ipdb; ipdb.set_trace()                 
        # with torch.inference_mode():
        
        masked_scores = model(masked)

        masked_probs = torch.softmax(masked_scores, dim=-1)
        #---------------------------------------------------------------
        if os.environ.get('DBG_TRANSITION',False):
            opts.mode = "class-transition"
        if opts.mode == 'saliency':
            # from losses import get_loss,get_score_smoothness
            import losses
            loss_term_per_thresh,loss_term,tot_loss = losses.get_loss(opts.loss_mode,masked_scores,opts.target_class)
            score_smoothness = losses.get_score_smoothness(masked_scores,opts.target_class)
            
        elif opts.mode == "class-transition":
            pass
        elif opts.mode == 'self-saliency':
            # import ipdb;ipdb.set_trace()
            pred_feats = opts.feat_layer.feats
            loss_term_per_thresh = ((ref_feats - pred_feats)**2)
            loss_term_per_thresh = loss_term_per_thresh.view(loss_term_per_thresh.shape[0],-1).sum(dim=-1)
            tot_loss =loss_term = loss_term_per_thresh.sum()
            score_smoothness = torch.zeros_like(tot_loss)
        elif opts.mode == 'cosaliency':
            pred_feats = opts.feat_layer.feats
            masked_scores2 = model(masked2)
            pred_feats2 = opts.feat_layer.feats
            def get_entropy(scores):
                p = torch.softmax(scores,dim=1)
                logp = torch.log_softmax(scores,dim=1)
                entropy = -torch.sum(p*logp,dim=1)
                return entropy
            loss_term_per_thresh = 1*(( (pred_feats ) - (pred_feats2) )**2).sum(dim=-1) 
            
            # loss_term_per_thresh = ( (pred_feats -ref_feats) - (pred_feats2-ref_feats2) )**2
            # loss_term_per_thresh = ( (pred_feats - pred_feats[:1].detach()) - (pred_feats2 - pred_feats2[:1].detach()) )**2
            # loss_term_per_thresh = -1000*( pred_feats/pred_feats.norm(dim=1,keepdim=True) * pred_feats2/pred_feats2.norm(dim=1,keepdim=True) ).sum(dim=-1)
            assert loss_term_per_thresh.ndim == 1
            loss_term_per_thresh = loss_term_per_thresh.view(loss_term_per_thresh.shape[0],-1).sum(dim=-1) + 10*(get_entropy(masked_scores) + get_entropy(masked_scores2)).sum()
            tot_loss = loss_term = loss_term_per_thresh.sum()
            score_smoothness = torch.zeros_like(tot_loss)            

        trends['loss_term'].append(loss_term.item())        
        trends['loss_term_per_thresh'].append(tensor_to_numpy(loss_term_per_thresh))        
        trends['score_smoothness'].append(score_smoothness.item())
        optimizer.zero_grad()
        #==============================================================
        if False:
            def uniform_grad(g):
                print('uniform_grad')
                print('uniform_grad HACK')
                gnorm = g.flatten(start_dim=1).norm(dim=1)
                avg_gnorm = gnorm.mean()
                g = avg_gnorm*g/(gnorm[:,None,None,None] + 1e-6)
                # multiplier = loss_term_per_thresh
                multiplier = (masked_scores[:,opts.target_class] - masked_scores.amax(dim=-1)).abs().detach()
                g = g * multiplier[:,None,None,None]
                return g
            #==============================================================
            multi_mask.register_hook(uniform_grad)        
        def utilize_t_grad(g):
            print('utilize_t_grad')
            # denom = 
            # numer = (multi_mask**2 * (1 - multi_mask))
            # common_term = sharpness * (numer/denom).sum(dim=(1,2,3),keepdim=True)
            # gnew = g - common_term * numer
            eps = 1e-4
            sigma = multi_mask + eps
            
            denom = (sigma**2 * (1 - sigma)).sum(dim=(1,2,3),keepdim=True)
            numer = g.sum(dim=(1,2,3),keepdim=True)
            multiplier = (sigma**2) * (1 -sigma)
            gnew = g -  multiplier * ((numer + eps)/(denom + eps)) 
            if gnew.isnan().any() or gnew.isinf().any():
                import ipdb; ipdb.set_trace()
            return gnew
        if opts.UTILIZE_T_GRAD:
            max_of_smooth_mask_rep.register_hook(utilize_t_grad)  
        def gradient_power_hook(g):
            losses.get_gradient_power(g,loss_term_per_thresh,trends)
            # lambda g,losses=loss_term_per_thresh,trends=trends:
            pass
        masked.register_hook(gradient_power_hook)
        tot_loss.backward()
        # losses.get_gradient_power(masked.grad,loss_term_per_thresh,trends)
        # import ipdb; ipdb.set_trace()
        if method == 'fong':
            if mask_logit_full.grad.isnan().any() or mask_logit_full.grad.isinf().any():
                import ipdb; ipdb.set_trace()    

        optimizer.step()

        change_in_mask_logit_full.update(mask_logit_full)
        change_in_multi_mask.update(multi_mask)
        change_in_masked.update(masked)
        trends['change_in_mask_logit_full'] = change_in_mask_logit_full.change
        # import ipdb; ipdb.set_trace()
        peak_of_smooth_mask = np.unravel_index((tensor_to_numpy(max_of_smooth_mask  [0,0].argmax())),max_of_smooth_mask[0,0].shape)
        trends['peak_of_smooth_mask'].append(peak_of_smooth_mask)
        if opts.mode == 'cosaliency':
            peak_of_smooth_mask2 = np.unravel_index((tensor_to_numpy(max_of_smooth_mask2  [0,0].argmax())),max_of_smooth_mask2[0,0].shape)
            trends['peak_of_smooth_mask2'].append(peak_of_smooth_mask2)

        if opts.print_iter is not None and (i) % opts.print_iter == 0:
            def save():
                print(f'Epoch {i} \tLoss:{loss_term.item()}')
                try:
                    #==========================================================
                    if True:
                    # if (i)// opts.print_iter <=1:
                        # utils.img_save2(ref,f'ref.png',syncable=True)    

                        def save_ref(ref,peak_of_smooth_mask,imroot='ref'):
                            ref_ = detransform(ref[0].detach().clone())
                            plt.figure()
                            plt.imshow(ref_)
                            plt.scatter(peak_of_smooth_mask[1],peak_of_smooth_mask[0])
                            plt.draw()
                            # plt.savefig(os.path.join(utils.SAVE_DIR,f'max_of_smooth_mask_{i}.png'))
                            plt.savefig(os.path.join(utils.SAVE_DIR,f'{imroot}.png'))
                            plt.close()
                        save_ref(ref,peak_of_smooth_mask,imroot='ref')
                        if opts.mode == 'cosaliency':
                            save_ref(ref2,peak_of_smooth_mask2,imroot='ref2')

                    utils.img_save2(mask_logit_full,f'mask_logit_full_{i}.png',syncable=True)
                    utils.img_save2(mask_logit_full,f'mask_logit_full.png',syncable=True)
                    # utils.img_save2(max_of_smooth_mask,f'max_of_smooth_mask_{i}.png',syncable=True)
                    # utils.img_save2(max_of_smooth_mask,f'max_of_smooth_mask.png',syncable=True)
                    #==========================================================
                    masked_grid = vutils.make_grid(masked)
                    utils.img_save2(masked_grid,f'grid_{i}.png',syncable=True)                    
                    utils.img_save2(masked_grid,f'grid.png',syncable=True)                    
                    #==========================================================
                    utils.save_plot2(tensor_to_numpy(loss_term_per_thresh),'loss_term_per_thresh',f'loss_term_per_thresh_{i}.png')
                    utils.save_plot2(tensor_to_numpy(loss_term_per_thresh),'loss_term_per_thresh',f'loss_term_per_thresh.png')
                    #==========================================================
                    mask_grid = vutils.make_grid(multi_mask)
                    utils.img_save2(mask_grid,f'masks_{i}.png',syncable=True)                    
                    utils.img_save2(mask_grid,f'masks.png',syncable=True)                    

                    #==========================================================
                    # plot the maximum point of max_of_smooth_mask
                    # import ipdb; ipdb.set_trace()
                    def save_masks(peak_of_smooth_mask,
                    max_of_smooth_mask,
                    imroot='max_of_smooth_mask'):
                        plt.figure()
                        plt.imshow(max_of_smooth_mask[0,0].data.detach().cpu().numpy())
                        plt.scatter(peak_of_smooth_mask[1],peak_of_smooth_mask[0])
                        plt.draw()
                        plt.savefig(os.path.join(utils.SAVE_DIR,f'{imroot}_{i}.png'))
                        plt.savefig(os.path.join(utils.SAVE_DIR,f'{imroot}.png'))
                        plt.close()
                        # import ipdb;ipdb.set_trace()
                    save_masks(peak_of_smooth_mask,
                    max_of_smooth_mask,
                    imroot='max_of_smooth_mask')
                    if opts.mode == 'cosaliency':
                        save_masks(peak_of_smooth_mask2,
                    max_of_smooth_mask2,
                    imroot='max_of_smooth_mask2')
                    #==========================================================
                    if False and (i > 499):
                        import ipdb;ipdb.set_trace()
                        # detect large change in loss_term

                except Exception as e:
                    import ipdb;ipdb.set_trace()
                    pass
                # import ipdb;ipdb.set_trace()
                              
                for lname in ['loss_term','area_error','score_smoothness','change_in_mask_logit_full','grad_power']:
                    utils.save_plot2(trends[lname],lname,f'{lname}.png')
                for lname in ['grad_mag_per_thresh']:
                    utils.save_plot2(trends[lname][-1],lname,f'{lname}_{i}.png')
                    utils.save_plot2(trends[lname][-1],lname,f'{lname}.png')
            # import ipdb; ipdb.set_trace()
            if os.environ.get('DBG_SAVE',False):  
                save()    
            else:
                utils.run_in_another_thread(save)
        #===============================================
        # print(masked_scores[:,opts.target_class])
        # import ipdb;ipdb.set_trace()
        # continue        
        #===============================================
        #==============================
        change = np.abs(np.diff(trends['loss_term'][-10:]))
        mean_change = np.mean(change)
        if i > 0:
            if  change[-1] > 10*mean_change:
                import ipdb;ipdb.set_trace()
        #==============================
        if decay_iter is not None:
            if (i+1) % decay_iter == 0:
                fong_utils.decay_lr(optimizer, decay_factor)
        if True and method == 'fong':
            # x__.data.copy_(torch.clamp(x__, 0, 1))
            pass
        
    # f, ax = plt.subplots(1,2)
    # ax[0].imshow(detransform(img_[0]))
    # ax[1].imshow(detransform(x_[0].data.cpu()))
    # for a in ax:
    #     a.set_xticks([])
    #     a.set_yticks([])
    # plt.show()


# def main2():
#     import argparse
#     import sys
#     import traceback


#     import sys;
#     if False:
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--image', type=str,
#                 default='/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000006.JPEG')
#         parser.add_argument('--network', type=str, default='alexnet')
#         parser.add_argument('--size', type=int, default=227)
#         parser.add_argument('--layer', type=str, default='features.4')
#         parser.add_argument('--alpha', type=float, default=6.)
#         parser.add_argument('--beta', type=float, default=2.)
#         parser.add_argument('--alpha_lambda', type=float, default=1e-5)
#         parser.add_argument('--tv_lambda', type=float, default=1e-5)
#         parser.add_argument('--epochs', type=int, default=200)
#         parser.add_argument('--learning_rate', type=int, default=1e2)
#         parser.add_argument('--momentum', type=float, default=0.9)
#         parser.add_argument('--print_iter', type=int, default=25)
#         parser.add_argument('--decay_iter', type=int, default=100)
#         parser.add_argument('--decay_factor', type=float, default=1e-1)
#         parser.add_argument('--gpu', type=int, nargs='*', default=None)
#         parser.add_argument('--method', type=str,  default='fong')
#         parser.add_argument('--target_class', type=int,  default=None)
#         args = parser.parse_args()
#     else:
#         args = argparse.Namespace()
#         if False:
#             args.dataset = 'imagenet'
#             from find_imagenet_images import find_images_for_class_id
#             some_images = find_images_for_class_id(153)
#             args.image = some_images[0]
#         else:
#             # args.image = 'grace_hopper.jpg'
#             # """
#             # garter snake
#             args.dataset = 'imagenet'
#             args.image = '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000006.JPEG'
#             args.target_class = 57
#             # """
#             """
#             # mouse trap
#             args.dataset = 'imagenet'
#             args.image = '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000009.JPEG'
#             # args.target_class = 674            
#             args.target_class = None
#             """

        
#         args.network = 'vgg16'
#         args.size = 227
#         # args.layer = 'features.4'
#         args.layer = 'classifier.6'
#         args.alpha = 6.
#         args.beta = 2.
#         # args.alpha_lambda = 1e-5
#         args.alpha_lambda = 0e-1
#         # args.tv_lambda = 1e-5
#         # args.tv_lambda = 1e-1
#         args.tv_lambda = 1e-1
        
        
#         # args.learning_rate = 1e2
#         args.momentum = 0.9
#         args.print_iter = 25
        
#         args.decay_factor = 1e-1
#         args.device = 'cuda'
#         args.method = 'fong'
        
#         if args.method == 'fong':
#             args.decay_iter = None
#             args.learning_rate = 1e-4
#             # args.learning_rate = 1e-5 # (1e-5 for no utilize_) l2_max_score
#             # args.learning_rate = 1e-2
#             # args.learning_rate = 1e-1
#             args.epochs = 8000

#     # gpu = args.gpu
#     # cuda = True if gpu is not None else False
#     # use_mult_gpu = isinstance(gpu, list)
#     # if cuda:
#     #     if use_mult_gpu:
#     #         os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
#     #     else:
#     #         os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
#     # print(torch.cuda.device_count(), use_mult_gpu, cuda)
#     # import ipdb;ipdb.set_trace()
#     main(image=args.image, network=args.network, layer=args.layer, 
#             alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda, 
#             tv_lambda=args.tv_lambda, epochs=args.epochs,
#             learning_rate=args.learning_rate, momentum=args.momentum, 
#             print_iter=args.print_iter, decay_iter=args.decay_iter,
#             decay_factor=args.decay_factor, 
#             device=args.device,method=args.method,
#             target_class= args.target_class)

# if __name__ == '__main__':
#     main2()
