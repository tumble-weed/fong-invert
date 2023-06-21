import torch
import torch.nn.functional as F
global rand

class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256):
        super(PatchSWDLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj

    def forward(self, x, y,mask):
        b, c, h, w = x.shape

        global rand
        # Sample random normalized projections
        rand_ = torch.randn(self.num_proj, c*self.patch_size**2).to(x.device) # (slice_size**2*ch)
        if False:
            if 'rand' not in globals():
                rand = rand_
            else:
                import colorful
                print(colorful.red("using hardcoded rand"))
        else:
            rand = rand_
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        rand = rand.reshape(self.num_proj, c, self.patch_size, self.patch_size)

        box = torch.ones(1, 1*self.patch_size**2).to(x.device) # (slice_size**2*ch)
        box = box / torch.sum(box, dim=1, keepdim=True)  # noramlize to unit directions
        box = box.reshape(1, 1, self.patch_size, self.patch_size)

        small_mask = F.conv2d(mask, box)# .reshape(self.num_proj, -1)
        # Project patches
        # padded_x= F.pad(x, (self.patch_size//2, self.patch_size//2, self.patch_size//2, self.patch_size//2), mode='reflect')
        padded_x = x
        projx = F.conv2d(padded_x, rand).transpose(1,0)# .reshape(self.num_proj, -1)
        projx = torch.stack([projx[i:i+1][small_mask>0.] for i in range(self.num_proj)],dim=0)
        #projx.shape = (num_proj, elements)
        
        # padded_y = F.pad(y, (self.patch_size//2, self.patch_size//2, self.patch_size//2, self.patch_size//2), mode='reflect')
        padded_y = y
        projy = F.conv2d(padded_y, rand).transpose(1,0)#.reshape(self.num_proj, -1)
        # projy = projy[mask<0.5]
        projy = torch.stack([projy[i:i+1][small_mask==0] for i in range(self.num_proj)],dim=0)
        # projx.shape = (num_proj, b*h*w/stride**2) = (64,1*224*224/1**2) = (64,50176)
        # Duplicate patches if number does not equal
        # projx, projy = duplicate_to_match_lengths(projx, projy)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)
        projx, projy = projx[:,:min(projx.shape[-1],projy.shape[-1])], projy[:,:min(projx.shape[-1],projy.shape[-1])]
        loss = torch.abs(projx - projy).mean()

        return loss


def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates randomly selected entries from the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[1] == arr2.shape[1]:
        return arr1, arr2
    elif arr1.shape[1] < arr2.shape[1]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2