#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import utils
import dutils
# import multi_image_invert_noisy
# import invert_noisy
invert_noisy = utils.import_and_reload('invert_noisy')

from scripts.run_utils import set_opts
MAJOR_PREFIX1 = os.path.basename(__file__).split('.')[0]
print = utils.printl
common_setting = {'ZOOM_JITTER':False,'network':'alexnet','epochs':2000,'LOSS_MODE':'match',
                  'purge':True}
setting = {
    # 'noisy':{
    #     'noise_mag':1,
    #     'JITTER':False,
    #     'jitter':0,
    #     'tv_lambda':0e-1,
    #     'MAX_SCALE':1.2,
    #     'learning_rate':1e0,
    # },
    'noiseless':{   
        'noise_mag':0,
        'JITTER':True,
        'jitter':11,
        # 'tv_lambda':1e-3,
        # 'learning_rate':1e0,
    }
}
TV_LAMBDA_FACTORS = [-2,-1]
# LEARNING_RATES = [0,-1]
layers = {
    'alexnet':[f'features.{i}' for i in range(13)] + [f'classifier.{i}' for i in range(7)]
}
conv_layers = {
    'vgg19':[
            1,
               3,5,7,10,12,14,17,19,21,24,26,28
               ],    
    'alexnet':[1,4,7,9,11]
}
# classes = [1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 277, 283, 292, 294, 309,
                            # 311,
                            # 325, 340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
                            # 967, 574, 487]
if os.environ.get('DBG_LAYER',False):
    layers[common_setting['network']]  = [int(os.environ.get('DBG_LAYER'))]
if os.environ.get('EPOCHS',False):
    common_setting['epochs']  = int(os.environ.get('EPOCHS'))
common_setting['image'] = ['grace_hopper.jpg']
# for ZOOM_JITTER in [True,False]:
batch_size = 3
for tv_lambda_factor in TV_LAMBDA_FACTORS:
    for sname,sval in setting.items():
        for layer in layers[common_setting['network']]:
            for adaptive_match in [True,False]:
                #=================================================    
                # for k,v in sval.items():
                #     multi_image_invert_noisy.opts.set_and_lock(k,v)
                # for k,v in common_setting.items():
                #     multi_image_invert_noisy.opts.set_and_lock(k,v)
                set_opts(invert_noisy,common_setting)
                set_opts(invert_noisy,sval)        
                import copy
                conv_layers_i = copy.deepcopy(conv_layers[common_setting['network']])
                conv_layers_i = [layer_ii for layer_ii in conv_layers_i if layer_ii < layer]
                    
                invert_noisy.opts.set_and_lock('conv_layer_ixs',conv_layers_i)
                invert_noisy.opts.set_and_lock('layer',f"features.{layer}")
                invert_noisy.opts.set_and_lock('adaptive_match',adaptive_match)
                invert_noisy.opts.set_and_lock('tv_lambda',10**tv_lambda_factor)
                
                # invert_noisy.MINOR_PREFIX = f'{sname}_{layer}_{common_setting["network"]}'
                invert_noisy.MINOR_PREFIX = os.path.join(MAJOR_PREFIX1,
                                                        f'{common_setting["network"]}_{sname}_{layer}_match_{adaptive_match}_tv_{str(10**tv_lambda_factor)[:4]}')
                for i in range((len(common_setting['image']) + batch_size -1 )//batch_size):
                    images_i = common_setting['image'][i*batch_size:(i+1)*batch_size]
                    print(layer)
                    invert_noisy.opts.set_and_lock('image',images_i)
                    # continue
                    # if sname == 'noiseless':
                    #     continue
                    # multi_image_invert_noisy.opts.set_and_lock('layer','features.11')
                    invert_noisy.main2()        