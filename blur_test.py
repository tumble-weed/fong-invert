import torch
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import skimage.io
from elp_masking import get_masked_input,DELETE_VARIANT,PRESERVE_VARIANT
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
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
class GaussianBlur(nn.Module):
    def __init__(self, sigma):
        super(GaussianBlur, self).__init__()
        ksize = int(2 * 4 * sigma + 1) if int(2 * 4 * sigma + 1) % 2 != 0 else int(2 * 4 * sigma + 2)
        self.kernel = nn.Parameter(self.create_gaussian_kernel(ksize, sigma))

    def forward(self, x):
        original_size = x.shape
        # import ipdb; ipdb.set_trace()
        device=x.device
        x = F.pad(x, [self.kernel.size(2) // 2, self.kernel.size(2) // 2,
                      self.kernel.size(2) // 2, self.kernel.size(2) // 2], mode='reflect')
        # import ipdb; ipdb.set_trace()
        chans = x.shape[1]
        x_out = torch.zeros(original_size,device=device)
        # pad_size = self.kernel.shape[-2] // 2
        # padded = torch.nn.functional.pad(x, [pad_size]*4, mode='reflect')
        for ci in range(chans):
            x_out[:,ci:ci+1] = F.conv2d(x[:,ci:ci+1], self.kernel,padding=0)
        return x_out

    def create_gaussian_kernel(self, ksize, sigma):
        kernel = torch.zeros(1, 1, ksize, ksize)
        mean = ksize // 2
        for i in range(ksize):
            for j in range(ksize):
                
                kernel[0, 0, i, j] = torch.exp(-((i - mean) ** 2 + (j - mean) ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

def get_edge_map(tensor):
    # Compute the Sobel operator kernels to detect edges
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=tensor.device)
    sobel_y = sobel_x.t()

    # Compute the horizontal and vertical gradient components for each channel
    grad_x = F.conv2d(tensor.flatten(start_dim=0,end_dim=1).unsqueeze(1), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(tensor.flatten(start_dim=0,end_dim=1).unsqueeze(1), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    # import ipdb; ipdb.set_trace()
    grad_x = grad_x.view(tensor.shape[0],-1,*tensor.shape[2:])
    grad_y = grad_y.view(tensor.shape[0],-1,*tensor.shape[2:])
    # Compute the magnitude and direction of the gradient for each channel
    magnitudes = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    directions = torch.atan2(grad_y, grad_x)

    return magnitudes, directions

def remove_edges(images,keep_larger_than=0.5,order='lerf'):
    device = images.device
    edge_mag,edge_theta = get_edge_map(images)
    edge_mag = edge_mag.sum(dim=1,keepdim=True)
    blur = GaussianBlur(torch.tensor(5.,device=device))
    blur.to(device)
    blurred_edge_mag = blur(edge_mag)
    max_edge_mag = blurred_edge_mag.flatten(start_dim=1,end_dim=-1).max(dim=-1)[0]
    # keep_larger_than = 0.3
    if False:
        keep_region = blurred_edge_mag/max_edge_mag > (keep_larger_than)
    else:
        from areas import get_thresholds
        from saliency_utils import multi_thresh_sigmoid
        sharpness = 20
        required_areas = torch.tensor([keep_larger_than*np.prod(blurred_edge_mag.shape[-2:])],device=device)
        area_l = 1
        thresholds = get_thresholds((blurred_edge_mag/max_edge_mag) ,sharpness = sharpness,
                        required_areas=required_areas,l=area_l)
        keep_region = multi_thresh_sigmoid((blurred_edge_mag/max_edge_mag),thresholds,sharpness,None)
    if order == 'morf':
        keep_region = 1 - keep_region
        # obs_areas =multi_mask.sum(dim=(1,2,3)) 
    edge_removed,_ = get_masked_input(
                    images,
                    keep_region,
                    # perturbation=BLUR_PERTURBATION,
                    num_levels=8,
                    variant=PRESERVE_VARIANT,
                    # variant=DELETE_VARIANT,
                    smooth=0)     
    # utils.img_save2(edge_removed,f'edge_removed.png')
    # import ipdb; ipdb.set_trace()
    return edge_removed
def blur_test(images,model,max_blur,target_class,n_blur=10,mode='remove-edges'):
    order = 'morf'
    device = images.device
    if mode=='remove-edges':
        # blur_sigmas = torch.linspace(1e-2,1e0,n_blur,device=device)
        blur_sigmas = torch.linspace(0.1,1 if order == 'lerf' else 3,n_blur,device=device)
        blur_sigmas = torch.pow(10,(-blur_sigmas.flip(0)))
    else:
        blur_sigmas = torch.linspace(0.1,max_blur,n_blur,device=device)
    scores = []
    original_scores = model(images)
    original_probs = F.softmax(original_scores,dim=1)
    for si,s in enumerate(blur_sigmas):
        """
        # Load the input image and convert it to a PyTorch tensor
        input_image = Image.open("path/to/input/image.jpg").convert("RGB")
        input_tensor = transforms.ToTensor()(input_image)

        # Initialize the GaussianBlur module with the specified sigma value
        sigma = 2
        gaussian_blur = GaussianBlur(sigma)

        # Apply the Gaussian blur to the input tensor
        output_tensor = gaussian_blur(input_tensor.unsqueeze(0)).squeeze(0)

        # Convert the output tensor back to a PIL Image and save it
        output_image = transforms.ToPILImage()(output_tensor)
        output_image.save("path/to/output/image.jpg")
        """
        if True:
            # assert False,'s has to be a percentage'
            blurred_s = remove_edges(images,keep_larger_than=s,order=order)
        else:
            blur = GaussianBlur(s)
            blur.to(device)
            blurred_s = blur(images)
            if s < 1:
                blurred_s = (images)
        scores_s = model(blurred_s)
        scores.append(scores_s)
        utils.img_save2(blurred_s,f'blurred_{s:2.2f}.png')
    
    scores = torch.stack(scores,dim=0)
    probs = torch.softmax(scores,dim=-1)
    #======================================================
    # calculate probabilities and scores
    kl_from_original = F.kl_div(original_probs.log(),probs.view(-1,probs.shape[-1]),log_target=True,reduction='none').sum(dim=-1) - F.kl_div(original_probs.log(),original_probs,log_target=True,reduction='none').sum(dim=-1)
    kl_from_original = kl_from_original.view(probs.shape[0],probs.shape[1])
    scores_target = scores[:,:,target_class]
    probs_target = probs[:,:,target_class]
    print(probs_target)
    # import ipdb; ipdb.set_trace()
    #======================================================
    # utils.img_save2(x__,f'x_{i}.png')
    utils.save_plot2(tensor_to_numpy(probs_target),'blur_test_probs',f'blur_test_probs.png',x=tensor_to_numpy(blur_sigmas))
    utils.save_plot2(tensor_to_numpy(scores_target),'blur_test_scores',f'blur_test_scores.png',x=tensor_to_numpy(blur_sigmas))
    utils.save_plot2(tensor_to_numpy(kl_from_original),'kl_from_original',f'kl_from_original.png',x=tensor_to_numpy(blur_sigmas))
    
    # import ipdb; ipdb.set_trace()
    if False:
        print(colorful.red("make a plot with the trajectories of all images (probs and scores)"))
        print(colorful.red("make a plot with the average trajectory (probs and scores)"))
        import ipdb; ipdb.set_trace()
        plot_multiple
        #======================================================
import pickle
def test_deepinversion(model,device='cuda'):

    # di_fname = f'/root/evaluate-saliency-4/fong-invert/DeepInversion/generations/rn50_inversion/best_images/output_00030_gpu_0.pkl' 
    di_fname = f'/root/evaluate-saliency-4/fong-invert/DeepInversion/generations/rn50_inversion/best_images/output_00062_gpu_0.pkl' 
    with open(di_fname,'rb') as f:
        generated_np = pickle.load(f)
    generated = torch.tensor(generated_np,device=device)
    max_blur = 4
    target_class = 153
    n_blur=41
    test_mode = 'remove-edges'
    # test_mode = 'blur'
    max_ix = None
    max_score =  -100
    for ii in range(0,generated.shape[0]):
        scores = model(generated[ii:ii+1])
        if scores[0,target_class] > max_score:
            max_score = scores[0,target_class]
            max_ix = ii

    from find_imagenet_images import find_images_for_class_id
    impaths = find_images_for_class_id(target_class)
    ref = get_image_tensor(impaths[0],size=224,dataset='imagenet').to(device)

    blur_test(ref,model,max_blur,target_class,n_blur=n_blur,mode=test_mode)    
    # blur_test(generated[max_ix:max_ix+1],model,max_blur,target_class,n_blur=n_blur)
    # blur_test(generated[1:1+1],model,max_blur,target_class,n_blur=n_blur)
    
    import ipdb; ipdb.set_trace()
    
if __name__ == '__main__':
    import torchvision
    device = 'cuda'
    model = torchvision.models.alexnet(pretrained=True)
    model.eval()
    model.to(device)
    if False:

        ref = torch.ones(1,3,224,224,device=device)
        max_blur = 10
        target_class = 10
        n_blur = 11
        blur_test(ref,model,max_blur,target_class,n_blur=n_blur)
    test_deepinversion(model,device = device)