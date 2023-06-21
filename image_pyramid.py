import torch
import torch.nn.functional as F
import torch
import kornia.filters as filters

def gaussian_pyramid(image, num_levels, downsampling_factor=2,sizes=None, antialias=True):
    gaussian_pyramid = []
    current = image
    # gaussian_pyramid.append(image)
    if sizes is not None:
        num_levels = len(sizes)
    for i in range(num_levels):
        gaussian_pyramid.append(current)
        # Downsample the current image by a factor of downsampling_factor
        if sizes is None:
            # import ipdb; ipdb.set_trace()
            new_size = (int(current.shape[2]/downsampling_factor), int(current.shape[3]/downsampling_factor))
            downsampled = F.interpolate(current, size=new_size,    
                                    mode='bilinear', 
                                    # mode='bicubic', 
                                    # align_corners=True, 
                                    recompute_scale_factor=False, antialias=antialias)
        else:
            downsampled = F.interpolate(current, 
                                        size=sizes[i],    
                                    mode='bilinear', 
                                    # mode='bicubic', 
                                    # align_corners=True, 
                                    recompute_scale_factor=False, antialias=antialias)
        # Upsample the downsampled image to the original size
        # Set the current image to the downsampled image for the next iteration
        current = downsampled
    # Add the lowest resolution image (the coarsest approximation) to the pyramid
    gaussian_pyramid.append(current)
    return gaussian_pyramid


def laplacian_pyramid(image, num_levels, downsampling_factor=2, antialias=True):
    pyramid = []
    gaussian_pyramid = []
    current = image
    # gaussian_pyramid.append(image)
    for i in range(num_levels):
        # Downsample the current image by a factor of downsampling_factor
        downsampled = F.interpolate(current, scale_factor=1/downsampling_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True, antialias=antialias)
        # Upsample the downsampled image to the original size
        import ipdb; ipdb.set_trace()
        upsampled = F.interpolate(downsampled, size=current.shape[2:], mode='bilinear', align_corners=False)
        # import ipdb; ipdb.set_trace()
        """
        if current.min() < 0:
            if not (upsampled.min() < 0):
                import ipdb; ipdb.set_trace()
        """
        # gaussian_pyramid.append(downsampled)
        # Calculate the Laplacian image by subtracting the upsampled image from the current image
        laplacian = current - upsampled
        # Add the Laplacian image to the pyramid
        pyramid.append(laplacian)
        gaussian_pyramid.append(current)
        # Set the current image to the downsampled image for the next iteration
        current = downsampled
    # Add the lowest resolution image (the coarsest approximation) to the pyramid
    pyramid.append(current)
    gaussian_pyramid.append(current)
    return pyramid,gaussian_pyramid

#====================================================
def custom_downsampling(image, scale_factor):
    # define custom downsampling function
    return filters.gaussian_blur2d(image, (3, 3),(1,1))[:, :, ::scale_factor, ::scale_factor]

def create_laplacian_pyramid(image, levels=4, scale_factor=2):
    pyramid = []
    current_level = image
    for i in range(levels):
        downsampled = custom_downsampling(current_level, scale_factor)
        upsampled = filters.interpolate(downsampled, scale_factor=scale_factor, mode='bilinear')
        pyramid.append(current_level - upsampled)
        current_level = downsampled
    pyramid.append(current_level)
    return pyramid

def create_gaussian_pyramid(image, levels=4, scale_factor=2):
    pyramid = []
    current_level = image
    for i in range(levels):
        downsampled = custom_downsampling(current_level, scale_factor)
        pyramid.append(downsampled)
        current_level = downsampled
    pyramid.append(current_level)
    return pyramid


if False:
    from image_pyramid import laplacian_pyramid
    # laplacian_pyramid(image, num_levels, downsampling_factor=2, anti_aliasing=True)
    x_grad_pyramid = laplacian_pyramid(x_pre_noise_.grad, 7, downsampling_factor=2, antialias=True)
    x_grad_pyramid_i = x_grad_pyramid[2]
    x_grad_pyramid_i = (x_grad_pyramid_i - x_grad_pyramid_i.min())/(x_grad_pyramid_i.max() - x_grad_pyramid_i.min())
    utils.img_save2(x_grad_pyramid_i,f'x_grad_pyramid_{i}.png')
    