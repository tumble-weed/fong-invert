"""
SKIP=1 EPOCHS=200 python -m ipdb -c c scripts/run_noisy_layer_ablation.py
"""
#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import dutils
import invert_noisy
# import programmatic_pdb
import utils
from scripts.run_utils import set_opts
MAJOR_PREFIX1 = os.path.basename(__file__).split('.')[0]
SKIP = False
print = utils.printl
common_setting = {'ZOOM_JITTER':False,'network':'alexnet','epochs':2000,'LOSS_MODE':'max',
                  'purge':True}
setting = {
    'noisy':{
        'noise_mag':1,
        'JITTER':False,
        'jitter':0,
        'tv_lambda':1e-1,
        'MAX_SCALE':1.2,
        # 'network':'vgg19',
        "learning_rate":1e-1,
        'purge':True,
    },
    # 'noiseless':{   
    #     'noise_mag':0,
    #     'JITTER':True,
    #     'jitter':5,
    #     'tv_lambda':1e-3,
    # }
}
layers = {
    'alexnet':[f'features.{i}' for i in range(13)] + [f'classifier.{i}' for i in range(7)]
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
# classes = [1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 277, 283, 292, 294, 309,
#                             311,
#                             325, 340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
#                             967, 574, 487]
classes = [153]
"""
from programmatic_pdb import ProgrammaticIpdb
pipdb = ProgrammaticIpdb()
# p = ProgrammaticIpdb(ipdb.__main__.def_colors)

pipdb.onecmd('b invert_noisy.py:227') # Works before set_running_trace
pipdb.set_running_trace()
"""
# pipdb.onecmd('b foo')   

if os.environ.get('EPOCHS',False):
    setting['noisy']['epochs'] = int(os.environ['EPOCHS'])
if os.environ.get('SKIP',False):
    SKIP = True

NOISE_MAGS = [0.2,0.5,0.8,1]
items = zip(conv_layers[common_setting['network']],NOISE_MAGS)
import inspect
import itertools
class Runner():
    def __init__(self):
        self.gen = None
        pass
    def noise_mag_and_layer(self,invert_noisy):
        parent = self.get_parent()
        if self.gen is None:
            self.gen = itertools.product(conv_layers[common_setting['network']],NOISE_MAGS)
        # import ipdb; ipdb.set_trace()
        for conv_lix,noise_mag in self.gen:
            invert_noisy.opts.set_and_lock('layer',f"features.{conv_lix}")
            invert_noisy.opts.set_and_lock('noise_mag',noise_mag)
            invert_noisy.opts.set_and_lock('conv_layer_ixs',[conv_lix])
            invert_noisy.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1,
                                            f'{common_setting["network"]}_ablation_noise_layer_{conv_lix}_label_{parent["label"]}_noise_{noise_mag}')
            if SKIP:
                if os.path.isdir(os.path.join(utils.DEBUG_DIR,invert_noisy.MAJOR_PREFIX,invert_noisy.MINOR_PREFIX)):
                    print('Skipping',invert_noisy.MINOR_PREFIX)
                    continue
            invert_noisy.main2()
    def get_parent(self):
        return inspect.currentframe().f_back.f_back.f_locals

    def layer(self,invert_noisy):
        parent = self.get_parent()
        if self.gen is None:
            self.gen = conv_layers[common_setting['network']]
        for conv_lix in next(self.gen):
            invert_noisy.opts.set_and_lock('layer',f"features.{conv_lix}")
            invert_noisy.opts.set_and_lock('conv_layer_ixs',[conv_lix])
            invert_noisy.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1,
                                            f'{common_setting["network"]}_ablation_noise_layer_{conv_lix}_label_{parent["label"]}')
            if SKIP:
                if os.path.isdir(os.path.join(utils.DEBUG_DIR,invert_noisy.MAJOR_PREFIX,invert_noisy.MINOR_PREFIX)):
                    print('Skipping',invert_noisy.MINOR_PREFIX)
                    continue
            invert_noisy.main2()

runner = Runner()
# runner.noise_mag_and_layer(invert_noisy)
# for conv_lix in conv_layers[common_setting['network']]:
for sname,sval in setting.items():
    for label in classes:
        # for k,v in sval.items():
        #     invert_noisy.opts.set_and_lock(k,v)
        # for k,v in common_setting.items():
        #     invert_noisy.opts.set_and_lock(k,v)            
        set_opts(invert_noisy,common_setting)
        set_opts(invert_noisy,sval)
        
        invert_noisy.opts.set_and_lock('target_class',[label])
        # invert_noisy.opts.set_and_lock('layer','features.11')
        # invert_noisy.opts.set_and_lock('conv_layer_ixs',[conv_lix])
        """
        invert_noisy.opts.set_and_lock('layer',f"features.{conv_lix}")
        # import ipdb; ipdb.set_trace()
        invert_noisy.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1,
                                                    f'{common_setting["network"]}_ablation_noise_layer_{conv_lix}_label_{label}')
        """
        runner.noise_mag_and_layer(invert_noisy)