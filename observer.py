import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from scipy.stats import spearmanr
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
def validate_with_observer(model,observer,x_pre_noise):
# if True:
    # for layer in model.features.noisy_layers:
    #     layer.noise_mag
    x_pre_noise = x_pre_noise.detach().clone()
    with torch.inference_mode():
        orignal_noise_flag = model.features.noise_flag
        model.features.noise_flag = False
        x_scores0 = model(x_pre_noise)
        model.features.noise_flag = orignal_noise_flag
        x_scores1 = observer(x_pre_noise)
    if False:
        x_probs0 = F.softmax(x_scores0,dim=1)
        x_probs1 = F.softmax(x_scores1,dim=1)
        kl_div_observer = F.kl_div(x_probs0.log(),x_probs1,log_target=True)
    else:
        # import ipdb; ipdb.set_trace()
        # kl_from_start = dist.kl_divergence(cat, cat0)
        # kl_from_prev = dist.kl_divergence(cat, cat_prev)
        dist0 = dist.Categorical(logits=x_scores0)
        dist1 = dist.Categorical(logits=x_scores1)
        kl_div_observer = dist.kl_divergence(dist0, dist1).squeeze()
    # calculate spearman rank correlation between x_scores0[0] and x_scores1[0]

    # Assume x_scores0 and x_scores1 are numpy arrays with the same shape
    assert x_scores0.ndim == 2
    assert x_scores1.ndim == 2
    rho, pval = spearmanr(tensor_to_numpy(x_scores0[0]), tensor_to_numpy(x_scores1[0]))
    # print("Spearman's rank correlation coefficient: {:.3f}".format(rho))    
    return kl_div_observer,rho