import torch
import numpy as np
import colorful
import utils
import skimage.io
from PIL import Image
import torchvision
import pickle
#==========================================================
vgg_mean = (0.485, 0.456, 0.406)
vgg_std = (0.229, 0.224, 0.225)
def get_vgg_transform(size=224):
    vgg_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size),
            torchvision.transforms.Normalize(mean=vgg_mean,std=vgg_std),
            ]
        )
    return vgg_transform
def get_image_tensor(impath,size=(224,),dataset=None):
    im_ = skimage.io.imread(impath)
    if im_.ndim == 2:
        im_ = np.concatenate([im_[...,None],im_[...,None],im_[...,None]],axis=-1)
    im_pil = Image.fromarray(im_)
    if dataset == 'imagenet':
        # from cnn import get_vgg_transform
        vgg_transform = get_vgg_transform(size)
        ref = vgg_transform(im_pil).unsqueeze(0)
        return ref
    elif dataset in ['pascal','voc']:
        # vgg_mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229, 0.224, 0.225)
        bgr_mean = [103.939, 116.779, 123.68]
        mean = [m / 255. for m in reversed(bgr_mean)]
        std = [1 / 255.] * 3
        
        vgg_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size),
                torchvision.transforms.Normalize(mean=mean,std=std),
                ]
            )
        ref = vgg_transform(im_pil).unsqueeze(0)
        # import ipdb;ipdb.set_trace()
        return ref
    else:
        assert False
#==========================================================
def create_jigsaw(images,patch_size,n_trials):
    device = images.device
    #====================================================
    # print(colorful.red("hardcoding in create_jigsaw"))
    # images = torch.zeros(1,3,224,224,device =device)
    # patch_size = 19
    #====================================================
    # assert images.shape[-1] == images.shape[-1]
    n_ticks_on_y = images.shape[-2]//patch_size
    n_ticks_on_x = images.shape[-1]//patch_size
    # import ipdb; ipdb.set_trace()
    src_y = np.random.randint(patch_size//2,images.shape[-2]-(patch_size//2),n_ticks_on_x*n_ticks_on_y*n_trials)
    src_x = np.random.randint(patch_size//2,images.shape[-1]-(patch_size//2),n_ticks_on_x*n_ticks_on_y*n_trials)
    # import ipdb; ipdb.set_trace()
    tgt_y = np.arange(patch_size//2,images.shape[-2]-(patch_size//2)+1,patch_size)#[None,:]
    tgt_x = np.arange(patch_size//2,images.shape[-1]-(patch_size//2)+1,patch_size)#[None,:]
    #tgt_y = np.tile(tgt_y,[n_trials,1])
    tgt_y,tgt_x = np.meshgrid(tgt_y,tgt_x,indexing='ij')
    tgt_y,tgt_x = tgt_y.flatten(),tgt_x.flatten()
    tgt_y,tgt_x = tgt_y[None,],tgt_x[None,]
    tgt_y,tgt_x = np.tile(tgt_y,[n_trials,1]),np.tile(tgt_x,[n_trials,1])
    tgt_y,tgt_x = tgt_y.flatten(),tgt_x.flatten()
    patch_y,patch_x = np.meshgrid(np.arange(patch_size),np.arange(patch_size))
    patch_y,patch_x = patch_y.flatten(),patch_x.flatten()
    assert tgt_y.shape == src_y.shape
    assert tgt_x.shape == src_x.shape
    # import ipdb; ipdb.set_trace()
    # src_bticks = np.arange(images.shape[0])[:,None]
    src_patches = images[0,:,src_y[:,None] - (patch_size//2) + patch_y[None,:],src_x[:,None]- (patch_size//2) + patch_x[None,:]]
    jigsaws = torch.zeros(images.shape[0]*n_trials,images.shape[1],n_ticks_on_y*patch_size,n_ticks_on_x*patch_size,device=device)
    bticks = np.tile(np.arange(images.shape[0]*n_trials)[:,None],(1,n_ticks_on_x*n_ticks_on_y)).flatten()
    bticks = np.tile(bticks[:,None],(1,patch_y.shape[0]))
    tgt_patches = jigsaws[bticks,:,tgt_y[:,None] - (patch_size//2) + patch_y[None,:],tgt_x[:,None]- (patch_size//2) + patch_x[None,:]]
    print(tgt_patches.shape)
    src_patches = src_patches.permute(1,2,0)
    jigsaws[bticks,:,tgt_y[:,None] - (patch_size//2) + patch_y[None,:],tgt_x[:,None]- (patch_size//2) + patch_x[None,:]] = src_patches
    jigsaws = torch.nn.functional.interpolate(jigsaws,images.shape[-2:])
    utils.img_save2(jigsaws,f'jigsaw.png')
    # import ipdb; ipdb.set_trace()
    return jigsaws    
import torch
from combine_patches.aggregation import combine_patches
from jitter import jitter_patches,extract_patches
def create_jittered(ref,patch_size = (7,7),jitter_radius = 5,stride = 1):
    assert ref.shape[1] in [1,3]
    ref = ref.permute(0,2,3,1).contiguous()
    img_shape = ref.shape[1:3]
    patches = extract_patches(ref, patch_size, stride)
    
    I = torch.arange(np.prod([ref.shape[1] - 2*(patch_size[0]//2),ref.shape[2] - 2*(patch_size[1]//2)]),device=device).long()[:,None]
    D = torch.zeros(I.shape[0])
    aggregation_results =  combine_patches(patches, patch_size, stride, 
                                    ref.shape[1:3]+(3,),as_np=False,patch_aggregation='distance-weighted',
                                    distances=D,I=I)        
    ref_combined = aggregation_results['combined'].unsqueeze(0)

    assert ref_combined.shape == ref.shape
    assert torch.allclose(ref_combined.float(),ref.float())
    print('STAGE:ref_combined')
    jitteredI = jitter_patches(I.squeeze(1),img_shape,patch_size,radius = jitter_radius)
    jitteredI = jitteredI.unsqueeze(1)
    jittered_patches = patches[jitteredI.T][0]
    # import ipdb; ipdb.set_trace()
    aggregation_results_jittered =  combine_patches(jittered_patches, patch_size, stride, 
                                    ref.shape[1:3]+(3,),as_np=False,patch_aggregation='distance-weighted',
                                    distances=D,I=jitteredI)        
    jittered = aggregation_results_jittered['combined'].unsqueeze(0)
    if jitter_radius == 0:
        assert torch.allclose(jittered,ref)
    jittered = jittered.permute(0,3,1,2).contiguous()
    return jittered
def find_factors(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

import bisect

def closest_elements(n1, n2):
    n2 = sorted(n2)
    result = []
    for n in n1:
        index = bisect.bisect_left(n2, n)
        if index == 0:
            result.append(n2[0])
        elif index == len(n2):
            result.append(n2[-1])
        else:
            if n - n2[index - 1] < n2[index] - n:
                result.append(n2[index - 1])
            else:
                result.append(n2[index])
    return result
    
def jigsaw_test(images,target_class,model,min_size,max_size,n_scales,n_trials):
    device = images.device
    assert max_size <= images.shape[-2]/2, 'jigsaws can only be created for patch size <= half image width'
    factors = find_factors(images.shape[-2])
    if False:
        sizes0 = np.linspace(min_size,max_size,n_scales)
        sizes = closest_elements(sizes0, factors)
    else:
        sizes = np.linspace(min_size,max_size,n_scales).astype(np.uint32)
        sizes = np.concatenate([sizes,[images.shape[-2]-1]],axis=0)
    # import ipdb; ipdb.set_trace()
    scores= []
    for s in sizes:
        # 10 trials * 5 images = 50 jigsaws
        # jigsaws = create_jigsaw(images,s,n_trials)
        jigsaws = create_jittered(images,patch_size = (7,7),jitter_radius = 1,stride = 1)
        
        # import ipdb; ipdb.set_trace()
        scores_s = model(jigsaws)
        scores.append(scores_s)
        utils.img_save2(jigsaws,f'jigsaw_{s}.png')
        # import ipdb; ipdb.set_trace()
    scores = torch.stack(scores,dim=0)
    probs = torch.softmax(scores,dim=-1)
    #======================================================
    # calculate probabilities and scores
    scores_target = scores[:,:,target_class]
    probs_target = probs[:,:,target_class]        

    print(sizes)
    print(probs_target.mean(dim=1))
    import ipdb; ipdb.set_trace()
    pass

if __name__ == '__main__':
    import torchvision
    device = 'cuda'
    model = torchvision.models.alexnet(pretrained=True)
    model.eval()
    model.to(device)
    # ref = torch.ones(1,3,224,224,device=device)
    
    min_size = 16
    max_size = 112
    n_scales = 10
    n_trials = 10
    target_class = 153
    from find_imagenet_images import find_images_for_class_id
    impaths = find_images_for_class_id(target_class)
    ref = get_image_tensor(impaths[0],size=224,dataset='imagenet').to(device)

    jigsaw_test(ref,target_class,model,min_size,max_size,n_scales,n_trials)
    if True:
        with open(f'/root/evaluate-saliency-4/fong-invert/DeepInversion/generations/rn50_inversion/best_images/output_00030_gpu_0.pkl','rb') as f:
            generated_np = pickle.load(f)
        generated = torch.tensor(generated_np,device=device)
        max_blur = 10
        target_class = 153
        n_blur=11
        max_ix = None
        max_score =  -100
        for ii in range(0,generated.shape[0]):
            scores = model(generated[ii:ii+1])
            if scores[0,target_class] > max_score:
                max_score = scores[0,target_class]
                max_ix = ii
        jigsaw_test(generated[max_ix:max_ix+1],target_class,model,min_size,max_size,n_scales,n_trials)
        