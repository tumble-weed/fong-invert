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
import invert_noisy
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
    # 'network':'vgg19',
    'epochs':2005,
    # 'LOSS_MODE':'match',
    # "cam_loss":False,
    "simonyan_saliency":False,
    'purge':False,
    'print_iter':500,
    'sync':True,
    'observer_model':"resnet50"
    }

setting = {
    'noisy':{
        # 'noise_mag':1,
        'JITTER':False,
        'jitter':0,
        'tv_lambda':1e-1,
        'MAX_SCALE':1.2,
        # 'epochs':2000,
        # 'network':'vgg19',
    },
    'noiseless':{   
        'noise_mag':0,
        'JITTER':True,
        'jitter':30,
        'tv_lambda':1e-1,
        'MAX_SCALE':1.2,
    }
    }
classes = [
    1, 25, 63, 92, 151,283, 
    933, 946, 980,    94, 107, 985,  154, 207, 250, 270, 
    277,  292, 294, 309,311,325, 
    340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
    967, 574, 487]
other_classes = list(range(1000))
other_classes = [c for c in other_classes if c not in classes]
classes = classes + other_classes[:0]
# import ipdb;ipdb.set_trace()
NOISE_MAGS = [1]
NETWORKS = [
        'alexnet',
            'vgg19']
ZOOM_JITTERS = [True,False]
CAM_WEIGHTS = [0]
CAM_LOSS=[False]
# ))
# order = list(PARAMETERS.keys())
# attribute_positions = {k:i for i,k in enumerate(order)}
batch_size = len(classes)
#===========================================================
if os.environ.get('DBG_FAST',False):
    common_setting['epochs'] = 500
    classes = classes[:2]
    os.environ['DBG_SAVE'] = '1'
    common_setting['print_iter'] = 100
if os.environ.get('DBG_FEW_CLASSES',False):
    classes = classes[:2]    
if os.environ.get('DBG_NOISE_JITTER',False):
    common_setting['epochs'] = 500
    classes = classes[:2]
    os.environ['DBG_SAVE'] = '1'
    common_setting['print_iter'] = 100    
    setting["noisy"]["noise_jitter"] = True
    setting["noisy"]["min_noise_mag"] = 0.8
    NETWORKS = ["alexnet"]
    MAJOR_PREFIX1 = f"{MAJOR_PREFIX1}_noise_jitter"
if os.environ.get('DBG_OBSERVER',False):
    common_setting['epochs'] = 500
    classes = classes[:2]
    os.environ['DBG_SAVE'] = '1'
    common_setting['print_iter'] = 100    
    observer_model = "resnet50"
    setting["noisy"]["observer_model"]  = observer_model
    NETWORKS = ["alexnet"]
    MAJOR_PREFIX1 = f"{MAJOR_PREFIX1}_observer_{observer_model}"    
if os.environ.get('EPOCHS',False):
    common_setting['epochs']  = int(os.environ.get('EPOCHS'))
#=============================================================
class Runner():
    def __init__(self):
        self.gen = None
        pass
    def run_full(self,args):
        MAJOR_PREFIX1_ = MAJOR_PREFIX1
        if args.experiment == 'best_alexnet':
            ZOOM_JITTERS[:] = [True]
            NETWORKS[:] = ["alexnet"]
            setting['noiseless']['MAX_SCALE'] = 1.
            MAJOR_PREFIX1_ += "_best_alexnet"
            pass
        # if args.experiment == 'best_alexnet_redo_noiseless':
        #     ZOOM_JITTERS[:] = [True]
        #     NETWORKS[:] = ["alexnet"]
        #     setting['noiseless']['MAX_SCALE'] = 1.
        #     MAJOR_PREFIX1_ += "_best_alexnet"
        #     del setting['noisy']
        #     classes[:] = classes[:6]
        #     pass        
        if args.experiment == 'best_alexnet_cam':
            ZOOM_JITTERS[:] = [True]
            NETWORKS[:] = ["alexnet"]
            CAM_WEIGHTS[:] = [5]
            CAM_LOSS[:] = [True]
            MAJOR_PREFIX1_ += "_best_alexnet_cam"
            pass        
        elif args.experiment == 'best_vgg':
            ZOOM_JITTERS[:] = [True]
            NETWORKS[:] = ["vgg19"]
            setting['noiseless']['MAX_SCALE'] = 1.
            MAJOR_PREFIX1_ += "_best_vgg"
            pass        
        if args.experiment == 'best_vgg_redo_noiseless':
            ZOOM_JITTERS[:] = [True]
            NETWORKS[:] = ["vgg19"]
            setting['noiseless']['MAX_SCALE'] = 1.
            MAJOR_PREFIX1_ += "_best_vgg"
            del setting['noisy']
            classes[:] = classes[:6]
            pass                
        elif args.experiment == 'compare_zoom_alexnet':
            ZOOM_JITTERS[:] = [False,True]
            NETWORKS[:] = ["alexnet"]
            classes[:] = classes[:10]
            MAJOR_PREFIX1_ += "compare_zoom_alexnet"
            
            del setting['noiseless']

        elif args.experiment == 'noise_ablation_alexnet':
            ZOOM_JITTERS[:] = [True]
            NETWORKS[:] = ["alexnet"]        
            # classes[:] = classes[:10]    
            NOISE_MAGS[:] = [0,0.2,0.5,0.8,1]
            MAJOR_PREFIX1_ += "noise_ablation_alexnet"
            del setting['noiseless']
        elif args.experiment == 'dummy':
            MAJOR_PREFIX1_ += "_dummy"
            NOISE_MAGS[:] = [0.5]

        if self.gen is None:
            self.gen = itertools.product(NETWORKS,ZOOM_JITTERS,setting.items(),NOISE_MAGS,CAM_WEIGHTS,CAM_LOSS)
        GROUP_PURGED = False
        # import ipdb;ipdb.set_trace()
        for combo in self.gen:
            for i in range((len(classes) + batch_size -1 )//batch_size):        
                network,ZOOM_JITTER,(sname,sval),noise_mag,cam_weight,cam_loss = combo
                """
                def keep_printing_combo(combo=combo):
                    import time
                    while True:
                        print(colorful.yellow_on_blue(combo))
                        time.sleep(5)
                utils.run_in_another_thread(keep_printing_combo)
                """
                # continue
                if args.networks and (network not in args.networks):
                    continue
                if args.noise_settings and (sname not in args.noise_settings):
                    continue
                # if args.zooms and (ZOOM_JITTER not in args.networks):
                #     continue
                
                labels_i = classes[i*batch_size:(i+1)*batch_size]

                #-------------------------------------------------------
                # if i > 0:
                #     set_opts(invert_noisy,{'purge':False})
                set_opts(invert_noisy,common_setting)                
                invert_noisy.opts.set_and_lock('ZOOM_JITTER',ZOOM_JITTER)
                invert_noisy.opts.set_and_lock('network',network)
                invert_noisy.opts.set_and_lock('noise_mag',noise_mag)
                invert_noisy.opts.set_and_lock('target_class',labels_i)
                invert_noisy.opts.set_and_lock('cam_weight',cam_weight)
                invert_noisy.opts.set_and_lock('cam_loss',cam_loss)
                set_opts(invert_noisy,sval)        
                #-------------------------------------------------------
                print(combo)
                invert_noisy.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1_,
                                            f'{invert_noisy.opts.network}_{sname}_zoom_{invert_noisy.opts.ZOOM_JITTER}_noise{invert_noisy.opts.noise_mag}_cam{invert_noisy.opts.cam_weight}')
                """
                invert_noisy.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1_,
                                            f'{invert_noisy.opts.network}_{sname}_zoom_{invert_noisy.opts.ZOOM_JITTER}_{"_".join([str(c) for c in labels_i])}_noise{invert_noisy.opts.noise_mag}_cam{invert_noisy.opts.cam_weight}')
                """
                if args.dry_run:
                    # import ipdb;ipdb.set_trace()
                    # utils.set_save_dir(invert_noisy.get_save_dir(),purge=False)    
                    print(invert_noisy.get_save_dir())
                    continue
                #-------------------------------------------------------
                if common_setting.get('purge',False) and not GROUP_PURGED:
                    group_folder = os.path.dirname(invert_noisy.get_save_dir())
                    os.system(f'rm -rf {group_folder}')
                    os.makedirs(group_folder)
                    utils.sync_to_gdrive(group_folder)
                    # import ipdb; ipdb.set_trace()
                    GROUP_PURGED = True         
                utils.SYNC_DIR = os.path.join(utils.DEBUG_DIR,invert_noisy.MAJOR_PREFIX,MAJOR_PREFIX1_)
                #-------------------------------------------------------
                # if SKIP:
                #     if os.path.isdir(utils.SAVE_DIR):
                #         last_saved_iter = (invert_noisy.opts.epochs//invert_noisy.opts.print_iter) * invert_noisy.opts.print_iter -1
                #         if os.path.exists(os.path.join(utils.SAVE_DIR,f'x_{last_saved_iter}.png')):
                #             print(f'SKIPPING {utils.SAVE_DIR}')
                #             continue   
                # utils.SYNC_DIR = os.path.join(utils.DEBUG_DIR,invert_noisy.MAJOR_PREFIX,MAJOR_PREFIX1_)
                utils.cipdb('DBG_BEST_ALEXNET')
                utils.cipdb('DBG_COMPARE_ZOOM_ALEXNET')
                utils.cipdb('DBG_NOISE_ABLATION_ALEXNET')
                invert_noisy.main2()
    

def main():            
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',type=str,default=None)
    parser.add_argument('--dry_run',action="store_true",default=False)
    parser.add_argument('--networks',nargs="*",type=str,default=[])
    parser.add_argument('--noise_settings',nargs="*",type=str,default=[])
    # parser.add_argument('--zooms',nargs="*",type=str,default=[])

    args = parser.parse_args()
    runner = Runner()
    runner.run_full(args)
if __name__ == '__main__':
    main()
