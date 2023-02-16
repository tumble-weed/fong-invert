import numpy as np
import torch
import skimage.io
import os
mu = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]
DEBUG_DIR = 'debugging'
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
tensor_to_numpy = lambda t:t.detach().cpu().numpy()

def img_save(img, savename):
    '''
    An adaptive image saving method, that works for numpy arrays, torch tensors. Also can accomodate multiple shapes of the input as well as ranges of values    
    '''    
    if isinstance(img,torch.Tensor):
        img = tensor_to_numpy(img)
    
    # shape:
    print('img has shape: ',img.shape)
    if img.ndim == 4:
        print('got 4d input, assuming first channel is batch, saving the fist image')
        img = img[0]
    if img.ndim == 3:
        if img.shape[0] == 1:
            print('got input with 1 channel, assuming grayscale')
            img = img[0]
        elif img.shape[0] == 3:
            print('got input with 3 channels')
            img = np.transpose(img,(1,2,0))
    if img.min() >= 0 and img.max <= 1:
        print('got img with values in [0,1] range')
    else:
        print('TODO: figure out what to do with min < 0 and max> 1')
    skimage.io.imsave(savename,img)
    
def img_save2(img,basename):
    img_save(img,os.path.join(DEBUG_DIR,basename))