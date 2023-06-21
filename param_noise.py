import torch
from torch import nn
import copy
class NoisyModel(nn.Module):
    def __init__(self, model,layer_type, noise_mag=0.1):
        super().__init__()
        self.model = model
        self.noise_mag = noise_mag
        self.noise_flag = True
        self.layer_type = layer_type
        
    def forward(self,x):
        noisy_model = copy.deepcopy(self.model)
        for mod in noisy_model.modules():
            if isinstance(mod,nn.ReLU):
                mod.inplace = False
            if isinstance(mod,self.layer_type):
                # mod.register_forward_hook(self.add_noise)
                if self.noise_flag:
                    # take std within a weight across spatial locations
                    std = mod.weight.std(dim=(-1,-2),keepdim=True)
                    # take std at a location across weights
                    # std = mod.weight.std(dim=0,keepdim=True)
                    # import ipdb; ipdb.set_trace()
                    mod.weight.data.copy_(mod.weight + self.noise_mag*std*torch.randn_like(mod.weight))
        # import ipdb; ipdb.set_trace()
        return noisy_model(x)
    pass

def test():
    from torchvision import models
    model = models.resnet18(pretrained=True)
    noisy_model = NoisyModel(model,nn.Conv2d)
    ref = torch.ones(1,1,224,224)
    noisy_model(ref)