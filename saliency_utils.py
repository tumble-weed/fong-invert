import torch
import math
import torch.nn.functional as F
from torch import nn
import numpy as np
# from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
PADDERS = {
    'constant_minus_inf':lambda padding:torch.nn.ConstantPad2d(padding,-np.inf),
    'zero':torch.nn.ZeroPad2d,
    'reflection':torch.nn.ReflectionPad2d,
    }
import numbers
import torch

def log_softmax(x, dim=-1):
    return x - x.logsumexp(dim=dim, keepdim=True)

def softmax(x, dim=-1):
    return torch.exp(log_softmax(x, dim=dim))

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma,pad:str=None,normalize=True, unit_max = False,dim=2,order=1,device='cpu'):
        super(GaussianSmoothing, self).__init__()

        padder = lambda t:t
        if pad is not None:
            padder = PADDERS[pad]((kernel_size[1]//2,kernel_size[1]//2,
                    kernel_size[0]//2,kernel_size[0]//2))        
        self.padder = padder
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        assert not all([normalize, unit_max])
        
        kernel = torch.pow(kernel,order)
        if normalize:
            # Make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
        if unit_max:
            kernel = kernel/kernel.max()
        
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        kernel = kernel.to(device)
        # import pdb;pdb.set_trace()
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(self.padder(input), weight=self.weight, groups=self.groups)
    
    
#=============================================================
# SoftMax filter
#=============================================================
class SoftMaxFilter2d(torch.nn.Module):
    def __init__(self,kernel_size,in_channels,stride,padder:str='zero',sharpness=0.1,device='cpu',clean=True,normalize=False,detach=False):
        super(SoftMaxFilter2d,self).__init__()
        #...............................................
        # len(kernel_size)
        if not hasattr(kernel_size,'__len__'):
            print('here 0')
            kernel_size = (kernel_size,kernel_size)
        if len(kernel_size) == 1:
            print('here')
            kernel_size = (kernel_size[0],kernel_size[0])
        assert len(kernel_size) == 2
        #...............................................
        if not hasattr(stride,'len'):
            stride = (stride,stride)
        if len(stride) == 1:
            stride = (stride[0],stride[0])
        assert len(stride) == 2
        #...............................................
        self.kernel_size = kernel_size
        self.stride = stride
        print(kernel_size)
        self.padder = PADDERS[padder]((kernel_size[1]//2,kernel_size[1]//2,kernel_size[0]//2,kernel_size[0]//2))
        self.in_channels = in_channels
        #...............................................
        self.k = torch.ones((1,self.in_channels) + self.kernel_size).float()    
        self.sharpness = sharpness
        self.clean = clean
        self.normalize = normalize
        self.detach = False
    def forward(self,x):
        padded = self.padder(x)

        # windows = padded.unfold(2,self.kernel_size[0],1).unfold(3,self.kernel_size[0],1)
        # windows = windows.reshape(*windows.shape[:4],-1)
        windows = torch.nn.functional.unfold(padded,self.kernel_size,1)
        factor = 1
        if self.normalize:
            factor = windows.std(dim=-1,keepdim=True)
        
        
        softmax_of_windows = torch.nn.functional.softmax(self.sharpness * (windows/factor),dim=1)
        if self.detach:
            softmax_of_windows = softmax_of_windows.detach()
        softmax_of_windows  =  softmax_of_windows * windows
        softmax_of_windows = softmax_of_windows.sum(dim=1,keepdim=True)
        softmax_of_windows = softmax_of_windows.reshape(x.shape)
        # assert  softmax_of_windows.shape == x.shape
        if self.clean:
            torch.cuda.empty_cache()
        assert not softmax_of_windows.isnan().any()
        return softmax_of_windows

def multi_thresh_sigmoid(m,t,sharpness,trends):
    # sharpness = 20;print(f'setting sharpness to {sharpness}')
    assert len(t.shape) == 1
    arg = (m - t[:,None,None,None]) * sharpness
    assert arg.shape[0] == t.shape[0]
    assert arg.shape[1:] == m.shape[1:]
    if sharpness < 100:
        s = torch.sigmoid(arg)
    else:
        s0 = (m >= t[:,None,None,None]).float()
        s = (s0 - m).detach() + m
    factor = (s * (1-s))
    # from utils import track_per_threshold
    # track_per_threshold(trends,factor.shape[0],'sigmoid_damping_factor',factor.sum(dim=(1,2,3)))
    return  s              