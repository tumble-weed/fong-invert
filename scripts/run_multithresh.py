# EPOCHS=50 CUDA_VISIBLE_DEVICES=0 python -m ipdb -c c  scripts/run_multi_class.py
#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
from scripts.run_utils import set_opts
import dutils
import multithresh_saliency
import importlib
import utils
import itertools
import argparse
MAJOR_PREFIX1 = os.path.basename(__file__).split('.')[0]
SKIP=False
if os.environ.get('SKIP',False) == '1':
    SKIP= True
print = utils.printl
# import multiprocessing as mp
# mp.set_start_method('spawn')
# from collections import OrderedDict
# PARAMETERS = OrderedDict(dict(
common_setting = {
    # 'ZOOM_JITTER':False,
    'network':'vgg19',
    'epochs':2000,
    # 'LOSS_MODE':'match',
    # "cam_loss":False,
    "simonyan_saliency":False,
    'purge':False,
    'print_iter':500,
    'sync':True,
    # 'observer_model':"resnet50"
    'observer_model':None
    }

setting = {
    "standard":{
        
    }
    # 'noisy':{
    #     # 'noise_mag':1,
    #     'JITTER':False,
    #     'jitter':0,
    #     'tv_lambda':1e-1,
    #     'MAX_SCALE':1.2,
    #     # 'epochs':2000,
    #     # 'network':'vgg19',
    # },
    # 'noiseless':{   
    #     'noise_mag':0,
    #     'JITTER':True,
    #     'jitter':30,
    #     'tv_lambda':1e-1,
    #     'MAX_SCALE':1.2,
    # }
    }
# classes = [
#     1, 25, 63, 92, 151,283, 
#     933, 946, 980,    94, 107, 985,  154, 207, 250, 270, 
#     277,  292, 294, 309,311,325, 
#     340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
#     967, 574, 487]

NETWORKS = [
        'alexnet',
            'vgg19']


# ))
# order = list(PARAMETERS.keys())
# attribute_positions = {k:i for i,k in enumerate(order)}
batch_size = 1
#===========================================================
if os.environ.get('DBG_FAST',False):
    pass
if os.environ.get('EPOCHS',False):

    pass
#=============================================================
class Runner():
    def __init__(self):
        self.gen = None
        pass
    def run_full(self,args):
        MAJOR_PREFIX1_ = MAJOR_PREFIX1
        if args.experiment == 'best_alexnet':
            pass
        if self.gen is None:
            self.gen = itertools.product(setting.items())
        GROUP_PURGED = False
        for combo in self.gen:
            (sname,sval) = combo
            #-------------------------------------------------------
            # if i > 0:
            #     set_opts(multithresh_saliency,{'purge':False})
            set_opts(multithresh_saliency,common_setting)                
            set_opts(multithresh_saliency,sval)        
            # multithresh_saliency.opts.set_and_lock('ZOOM_JITTER',ZOOM_JITTER)            
            #-------------------------------------------------------
            print(combo)
            multithresh_saliency.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1_,
                                        f'{multithresh_saliency.opts.network}_{sname}_{"_".join([str(c) for c in labels_i])}')
            if args.dry_run:
                # import ipdb;ipdb.set_trace()
                # utils.set_save_dir(multithresh_saliency.get_save_dir(),purge=False)    
                print(multithresh_saliency.get_save_dir())
                continue
            #-------------------------------------------------------
            if common_setting.get('purge',False) and not GROUP_PURGED:
                group_folder = os.path.dirname(multithresh_saliency.get_save_dir())
                os.system(f'rm -rf {group_folder}')
                os.makedirs(group_folder)
                utils.sync_to_gdrive(group_folder)
                # import ipdb; ipdb.set_trace()
                GROUP_PURGED = True         
            utils.SYNC_DIR = os.path.join(utils.DEBUG_DIR,multithresh_saliency.MAJOR_PREFIX,MAJOR_PREFIX1_)
            #-------------------------------------------------------
            # if SKIP:
            #     if os.path.isdir(utils.SAVE_DIR):
            #         last_saved_iter = (multithresh_saliency.opts.epochs//multithresh_saliency.opts.print_iter) * multithresh_saliency.opts.print_iter -1
            #         if os.path.exists(os.path.join(utils.SAVE_DIR,f'x_{last_saved_iter}.png')):
            #             print(f'SKIPPING {utils.SAVE_DIR}')
            #             continue   
            # utils.SYNC_DIR = os.path.join(utils.DEBUG_DIR,multithresh_saliency.MAJOR_PREFIX,MAJOR_PREFIX1_)
            utils.cipdb('DBG_BEST_ALEXNET')
            utils.cipdb('DBG_COMPARE_ZOOM_ALEXNET')
            utils.cipdb('DBG_NOISE_ABLATION_ALEXNET')
            import ipdb; ipdb.set_trace()
            multithresh_saliency.main2()
    

def main():            
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',type=str,default=None)
    parser.add_argument('--dry_run',action="store_true",default=False)
    args = parser.parse_args()
    runner = Runner()
    runner.run_full(args)
if __name__ == '__main__':
    main()
