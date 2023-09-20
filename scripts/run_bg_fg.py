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
import invert_noisy_fg_bg
import importlib
import utils
import itertools
import argparse
# global MAJOR_PREFIX1
MAJOR_PREFIX1 = os.path.basename(__file__).split('.')[0]
print = utils.printl
# import multiprocessing as mp
# mp.set_start_method('spawn')
common_setting = {
    # 'ZOOM_JITTER':False,
    # 'network':'vgg19',
    'epochs':2000,
    # 'LOSS_MODE':'match',
    "cam_loss":False,
    "simonyan_saliency":False,
    'purge':True,
    'print_iter':500,
    'sync':True,
    'observer_model':"resnet50"
    }
setting = {
    'noiseless':{   
        'noise_mag':0,
        'JITTER':True,
        'jitter':30,
        'tv_lambda':1e-1,
        'MAX_SCALE':1.2,
    }
}

conv_layers = {
    'vgg19':[
            1,
               3,5,7,10,12,14,17,19,21,24,26,28
               ],
    'alexnet':[
        11,9,7,4,1
               ]    
}

classes = [
    1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 
    277, 283, 292, 294, 309,311,325, 
    340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
    967, 574, 487]
NOISE_MAGS = [1]
NETWORKS = [
            'alexnet',
                # 'vgg19'
                ]
ZOOM_JITTERS = [False]
# BG_WEIGHTS = [1,10,100]
# FG_WEIGHTS = [1,0.1,0.01]
BG_WEIGHTS = [1,]
FG_WEIGHTS = [0.1]
USE_MASK = [False]
ALIGN = [False]
batch_size = 1
#=============================================================
if os.environ.get('DBG_FAST',False):
    common_setting['epochs'] = 50
    classes = classes[:2]
    os.environ['DBG_SAVE'] = '1'
    common_setting['print_iter'] = 24
if os.environ.get('DBG_FEW_CLASSES',False):
    classes = classes[:2]
if os.environ.get('EPOCHS',False):
    common_setting['epochs'] = int(os.environ['EPOCHS'])
if os.environ.get('DBG_MASK',False):
    common_setting['epochs'] = 500
    classes = classes[:2]
    os.environ['DBG_SAVE'] = '1'
    common_setting['print_iter'] = 24
    BG_WEIGHTS = [1,10]
    FG_WEIGHTS = [1,10]
    USE_MASK = [False]
#=============================================================
class Runner():
    def __init__(self):
        self.gen = None
        pass
    def run(self,args):
        MAJOR_PREFIX1_ = MAJOR_PREFIX1
        if args.experiment == 'use_mask':
            USE_MASK[:] = [True]
            MAJOR_PREFIX1_ += '_mask'                        
        elif args.experiment == 'align':
            USE_MASK[:] = [False]
            ALIGN[:] = [True]
            MAJOR_PREFIX1_ += '_align'                        
            del classes[3:]
            # common_setting['epochs'] = 1
            # common_setting['sync'] = False
            BG_WEIGHTS = [1]
            FG_WEIGHTS = [1]            
        # global MAJOR_PREFIX1
        if self.gen is None:
            self.gen = itertools.product(NETWORKS,ZOOM_JITTERS,setting.items(),BG_WEIGHTS,FG_WEIGHTS,USE_MASK,ALIGN)
        # utils.SYNC = True
        # import ipdb; ipdb.set_trace()
        GROUP_PURGED = False
        for combo in self.gen:
            for i in range((len(classes) + batch_size -1 )//batch_size):   
                network,ZOOM_JITTER,(sname,sval),bg_weight,fg_weight,use_mask,align = combo     
                labels_i = classes[i*batch_size:(i+1)*batch_size]
                
                #-------------------------------------------------------
                # if i > 0:
                #     set_opts(invert_noisy_fg_bg,{'purge':False})                
                set_opts(invert_noisy_fg_bg,common_setting)
                invert_noisy_fg_bg.opts.set_and_lock('ZOOM_JITTER',ZOOM_JITTER)
                invert_noisy_fg_bg.opts.set_and_lock('network',network)
                utils.cipdb('DBG_CLASS1')                
                invert_noisy_fg_bg.opts.set_and_lock('target_class',labels_i)
                set_opts(invert_noisy_fg_bg,{'bg_weight':bg_weight})        
                set_opts(invert_noisy_fg_bg,{'fg_weight':fg_weight})        
                set_opts(invert_noisy_fg_bg,{'conv_layer_ixs':[min(conv_layers[network])]})        
                set_opts(invert_noisy_fg_bg,{'use_mask':use_mask})
                set_opts(invert_noisy_fg_bg,sval)
                # utils.cipdb('DBG_ALIGN')
                set_opts(invert_noisy_fg_bg,{'aligned_backprop':align})
                
                #-------------------------------------------------------
                print(combo)
                invert_noisy_fg_bg.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1_,
                                            f'{invert_noisy_fg_bg.opts.network}_{sname}_zoom_{invert_noisy_fg_bg.opts.ZOOM_JITTER}_{"_".join([str(c) for c in labels_i])}_bg_{invert_noisy_fg_bg.opts.bg_weight}_fg_{invert_noisy_fg_bg.opts.fg_weight}')

                if args.dry_run:
                    # import ipdb;ipdb.set_trace()
                    # utils.set_save_dir(invert_noisy.get_save_dir(),purge=False)    
                    print(invert_noisy_fg_bg.get_save_dir())
                    continue
                #-------------------------------------------------------
                if common_setting.get('purge',False) and not GROUP_PURGED:
                    group_folder = os.path.dirname(invert_noisy_fg_bg.get_save_dir())
                    os.system(f'rm -rf {group_folder}')
                    os.makedirs(group_folder)
                    utils.sync_to_gdrive(group_folder)
                    # import ipdb; ipdb.set_trace()
                    GROUP_PURGED = True         
                utils.SYNC_DIR = os.path.join(utils.DEBUG_DIR,invert_noisy_fg_bg.MAJOR_PREFIX,MAJOR_PREFIX1_)

                invert_noisy_fg_bg.main2()
def main():            
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',type=str,default=None)
    parser.add_argument('--dry_run',action="store_true",default=False)
    args = parser.parse_args()
    runner = Runner()
    runner.run(args)
if __name__ == '__main__':
    main()
