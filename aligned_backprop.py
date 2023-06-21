import torch
import utils
def register_keep_according_to_alignment(tensor,other_tensor,other_grad,mode='misaligned',FACTOR=1):
    def closure(g):

        # if (g- (other_grad)).abs().max() > 1e-5:
        #     import ipdb; ipdb.set_trace()

        
        if mode == 'misaligned':
            assert False, 'selected is wrong'

            selected = (1 - torch.sign(other_grad) * torch.sign(other_grad))/2. * (g) 
            assert g.norm() >= selected.norm()
            return FACTOR * g.norm() * selected/(selected.norm() + (selected.norm() == 0).float())
        elif mode == 'aligned':

            selected = (torch.sign(other_grad) * torch.sign(g)) * (g) 
            assert g.norm() >= selected.norm()
            # import ipdb; ipdb.set_trace()
            return FACTOR * g.norm() * selected/(selected.norm() + (selected.norm() == 0).float())     
    tensor.register_hook(closure)
    other_tensor.register_hook(lambda g:torch.zeros_like(g))
    # utils.cipdb('DBG_ALIGN')
