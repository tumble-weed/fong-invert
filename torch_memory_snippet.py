# prints currently alive Tensors and Variables
import torch
import gc
def print_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                # print(gc.get_referrers(obj))
        except:
            pass