import numpy as np
from inversion.common_utils import get_noise#,get_params
from inversion.common_utils import tv_loss
from inversion.models import skip
from torch import nn
import torch
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
from matplotlib import pyplot as plt
############################################################
imsize = 224
# Something divisible by a power of two
imsize_net = 256
# VGG and Alexnet need input to be correctly normalized

'''
outs = [] 
def get_class_ix(name):
    if name is None:
        return None
    class_ix = None
    for k,v in corresp.items():
        if name in v:
            class_ix = int(k)
            break
    return class_ix
if layer_to_use == 'fc8':
    # Choose class
    # name = 'black swan'
    # name = 'golden retriever'
    # name = 'cheeseburger'
    name = 'capuchin'
    # name = 'guenon'
    class_ix = get_class_ix(name)

    name_neg = 'capuchin'
    name_neg = None
    neg_class_ix = get_class_ix(name_neg)
else:
    class_ix = 2 # Choose here

def closure():
    
    global i, net_input
    
    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std()/50
    
    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)[:, :, :imsize, :imsize]
    
#     out = out* (1-mask)
   
        
    scores = cnn(vgg_preprocess_caffe(out))
    score_loss = - scores[:,class_ix].sum()
    if neg_class_ix is not None:
        score_loss += scores[:,neg_class_ix].sum()
    total_loss = 5 * score_loss 
    
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(vgg_preprocess_caffe(out), beta=2)
        
    use_mask = False
    if use_mask:
        total_loss += nn.functional.mse_loss(out * mask, mask * 0, size_average=False) * 1e1
    
    total_loss.backward()

    print ('Iteration %05d    Loss %.3f' % (i, total_loss.item()), '\r', end='')
    if PLOT and  i % 100==0:
        out_np = np.clip(tensor_to_numpy(out), 0, 1)
        plot_image_grid([out_np], 3, 3, interpolation='lanczos');
        
        outs.append(out_np)
    i += 1
    
    return total_loss
'''
############################################################

def optimize(num_iter,cnn,
            ref_feats,cnn_feats,tail=None):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
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
    outs = []
    #------------------------------------------------
    if 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            # closure()
            #######################################################            
            if param_noise:
                for n in [x for x in net.parameters() if len(x.size()) == 4]:
                    n = n + n.detach().clone().normal_() * n.std()/50
            net_input = net_input_saved
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)

            out = net(net_input)[:, :, :imsize, :imsize]
            cnn_feats[0] = None
            out1 = (out - 0.5)*2
            _ = cnn(out1)
            # if j == num_iter - 1:
            #     import ipdb;ipdb.set_trace()
            cnn_featsj = cnn_feats[0]
            if tail is not None:
                later_featsj= tail(cnn_featsj)
            total_loss = 5 * torch.nn.functional.mse_loss(cnn_featsj,ref_feats)
    
            if tv_weight > 0:
                total_loss += tv_weight * tv_loss(out1, beta=2)
            
            total_loss.backward()

            print ('Iteration %05d    Loss %.3f' % (j, total_loss.item()), '\r', end='')
            if PLOT and  j % 100==0:
                out_np = tensor_to_numpy(out)
                out_np_clipped =  np.clip(out_np, 0, 1)
                # out_np_clipped = np.clip(tensor_to_numpy(out), 0, 1)
                #------------------------------------------
                plt.figure()
                plt.imshow(np.transpose(out_np_clipped,(0,2,3,1))[0])
                plt.show()
                #------------------------------------------
                outs.append(out_np)
            optimizer.step()
        # import pdb;pdb.set_trace()
        return out_np

