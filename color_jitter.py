import torch
def jitter_color(tensor, factor):
    """
    Jitters the color of a tensor by a factor between 0 and 1.
    """
    device = tensor.device
    # Split tensor into separate color channels
    red_channel, green_channel, blue_channel = tensor.split(1, dim=1)
    
    # Compute jitter values for each channel
    max_val = 3
    min_val = -3
    red_jitter = (torch.rand(1,device=device) - 0.5) * 2*max_val * factor
    green_jitter = (torch.rand(1,device=device) - 0.5) * 2 *max_val* factor
    blue_jitter = (torch.rand(1,device=device) - 0.5) * 2*max_val * factor
    
    # Apply jitter to each channel
    new_red_channel = red_channel + red_jitter
    new_green_channel = green_channel + green_jitter
    new_blue_channel = blue_channel + blue_jitter
    
    # Clamp values to range [0, 1]
    red_channel = torch.clamp(new_red_channel, min_val, max_val)
    green_channel = torch.clamp(new_green_channel, min_val, max_val)
    blue_channel = torch.clamp(new_blue_channel, min_val, max_val)
    
    red_channel = (red_channel - new_red_channel).detach() + new_red_channel
    green_channel = (green_channel - new_green_channel).detach() + new_green_channel
    blue_channel = (blue_channel - new_blue_channel).detach() + new_blue_channel
    # Merge channels back into tensor
    jittered_tensor = torch.cat([red_channel, green_channel, blue_channel], dim=1)
    # import ipdb; ipdb.set_trace()
    return jittered_tensor