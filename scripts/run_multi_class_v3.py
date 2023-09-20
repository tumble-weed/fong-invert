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
MAJOR_PREFIX1 = os.path.basename(__file__).split('.')[0]
SKIP=False
if os.environ.get('SKIP',False) == '1':
    SKIP= True
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
    }
setting = {
    'noisy':{
        'noise_mag':1,
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
    1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 
    277, 283, 292, 294, 309,311,325, 
    340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
    967, 574, 487]
if os.environ.get('DBG_FAST',False):
    common_setting['epochs'] = 500
    classes = classes[:2]
    os.environ['DBG_SAVE'] = '1'
    common_setting['print_iter'] = 100
if os.environ.get('EPOCHS',False):
    common_setting['epochs']  = int(os.environ.get('EPOCHS'))
batch_size = 1
NETWORKS = [
        'alexnet',
            'vgg19'
            ]
utils.SYNC_DIR = os.path.join(utils.DEBUG_DIR,invert_noisy.MAJOR_PREFIX,MAJOR_PREFIX1)
for network in NETWORKS:
    for ZOOM_JITTER in [True,False]:
        for sname,sval in setting.items():
            
            # invert_noisy.MINOR_PREFIX = f'{common_setting["network"]}_{sname}_zoom_{ZOOM_JITTER}'
            # invert_noisy.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1,
            #                                 f'{network}_{sname}_zoom_{ZOOM_JITTER}')
            # for k,v in sval.items():
            #     invert_noisy.opts.set_and_lock(k,v)
            # for k,v in common_setting.items():
            #     invert_noisy.opts.set_and_lock(k,v)        
            set_opts(invert_noisy,common_setting)
            set_opts(invert_noisy,sval)        
            invert_noisy.opts.set_and_lock('ZOOM_JITTER',ZOOM_JITTER)
            invert_noisy.opts.set_and_lock('network',network)
            for i in range((len(classes) + batch_size -1 )//batch_size):        
                if i > 0:
                    set_opts(invert_noisy,{'purge':False})
                utils.cipdb('DBG_CLASS1')
                '''
                if sname == 'noisy':
                    labels_i = classes[classes.index(340):][i*batch_size:(i+1)*batch_size]
                    pass
                else:
                    labels_i = classes[i*batch_size:(i+1)*batch_size]
                '''
                labels_i = classes[i*batch_size:(i+1)*batch_size]
                invert_noisy.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1,
                                            f'{network}_{sname}_zoom_{ZOOM_JITTER}_{"_".join([str(c) for c in labels_i])}')
                # for k,v in sval.items():
                #     invert_noisy.opts.set_and_lock(k,v)
                invert_noisy.opts.set_and_lock('target_class',labels_i)
                # invert_noisy.opts.set_and_lock('layer','features.11')
                # utils.oipdb('DBG_CLASS3')
                if SKIP:
                    if os.path.isdir(utils.SAVE_DIR):
                        last_saved_iter = (invert_noisy.opts.epochs//invert_noisy.opts.print_iter) * invert_noisy.opts.print_iter -1
                        if os.path.exists(os.path.join(utils.SAVE_DIR,f'x_{last_saved_iter}.png')):
                            print(f'SKIPPING {utils.SAVE_DIR}')
                            continue
                invert_noisy.main2()
