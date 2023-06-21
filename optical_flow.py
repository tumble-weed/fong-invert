optical-flow

import torch
import torch.nn.functional as F

def compute_matching_cost(im1, im2, max_displacement):
    """
    Compute matching cost for optical flow using L2 distance similarity metric.
    Args:
        im1 (torch.Tensor): grayscale image tensor of shape (batch_size, 1, height, width)
        im2 (torch.Tensor): grayscale image tensor of shape (batch_size, 1, height, width)
        max_displacement (int): maximum pixel displacement to search for matching pixels in im2
    Returns:
        matching_cost (torch.Tensor): tensor of shape (batch_size, (2*max_displacement+1)**2, height, width) containing matching cost for each pixel
    """
    batch_size, _, height, width = im1.size()
    padding = max_displacement
    padded_im1 = F.pad(im1, [padding]*4, mode='replicate')
    padded_im2 = F.pad(im2, [padding]*4, mode='replicate')
    matching_cost = torch.zeros(batch_size, (2*max_displacement+1)**2, height, width).to(im1.device)
    for i in range(-max_displacement, max_displacement+1):
        for j in range(-max_displacement, max_displacement+1):
            shifted_im2 = padded_im2[:, :, padding+i:padding+i+height, padding+j:padding+j+width]
            diff = im1 - shifted_im2
            l2 = torch.sum(diff ** 2, dim=1, keepdim=True)
            matching_cost[:, (i+max_displacement)*(2*max_displacement+1) + j+max_displacement, :, :] = l2
    return matching_cost
