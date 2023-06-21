from torchvision.transforms import Resize as tv_resize
import torch
from patch_swd import PatchSWDLoss
def get_output_shape(initial_image_shape, size, aspect_ratio):
    """Get the size of the output pyramid level"""
    h, w = initial_image_shape
    h, w = int(size * aspect_ratio[0]), int((w * size / h) * aspect_ratio[1])
    return h, w

def _match_patch_distributions(synthesized_images, reference_images,mask, criteria, num_steps, lr, pbar):
    """
    Minimizes criteria(synthesized_images, reference_images) for num_steps SGD steps by differentiating self.synthesized_images
    :param reference_images: tensor of shape (b, C, H1, W1)
    :param synthesized_images: tensor of shape (b, C, H2, W2)
    :param debug_dir:
    """
    synthesized_images.requires_grad_(True)
    optim = torch.optim.Adam([synthesized_images], lr=lr)
    losses = []
    for i in range(num_steps):
        # Optimize image
        optim.zero_grad()
        loss = criteria(synthesized_images, reference_images,mask)
        loss.backward()
        optim.step()

        # Update staus
        losses.append(loss.item())
        pbar.step()
        pbar.print()

    return torch.clip(synthesized_images.detach(), -1, 1), losses

def apply_swd(masked,mask,
            pyramid_scales=(32, 64, 128, 227),
            # pyramid_scales=(227,),
            patch_size = 7,
            stride = 1,
            num_proj = 64,
            aspect_ratio = (1,1),
            ):
    if len(pyramid_scales) == 1:
        import colorful
        print(colorful.red("using single scale"))
    original_image_shape = masked.shape[-2:]
    all_losses = []
    criteria = PatchSWDLoss(patch_size=patch_size, stride=stride, num_proj=num_proj)
    for scale in pyramid_scales:
        lvl_masked = tv_resize(scale, antialias=True)(masked)
        lvl_mask = tv_resize(scale, antialias=True)(mask)
        lvl_output_shape = get_output_shape(original_image_shape, scale, aspect_ratio)
        # masked = tv_resize(lvl_output_shape, antialias=True)(synthesized_images)

        # synthesized_images, losses = _match_patch_distributions(lvl_masked, 
        #                                                         lvl_masked, 
        #                                                         lvl_mask,
        #                                                         criteria, 
        #                                                         num_steps, lr)
        loss = criteria(lvl_masked, lvl_masked,lvl_mask)
        all_losses += [loss]
    return sum(all_losses)

