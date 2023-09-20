#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import dutils
import invert_understand_noise
setting = {
    'noisy':{
        'noise_mag':1,
        'JITTER':False,
        'tv_lambda':1e-3,
    },
    'noiseless':{   
        'noise_mag':0,
        'JITTER':True,
        'jitter':5,
        'tv_lambda':1e-3,
    }
}
for sname,sval in setting.items():
    #=================================================    
    invert_understand_noise.opts.set_and_lock('noise_mag',sval['noise_mag'])
    invert_understand_noise.opts.set_and_lock('SIMPLE_REF',True)
    invert_understand_noise.opts.set_and_lock('tv_lambda',sval['tv_lambda'])
    invert_understand_noise.opts.set_and_lock('JITTER',sval['JITTER'])
    invert_understand_noise.opts.set_and_lock('jitter',sval['jitter'])
    invert_understand_noise.opts.set_and_lock('layer','features.11')
    invert_understand_noise.MINOR_PREFIX = f'square_noise_mag_{sval["noise_mag"]}'
    invert_understand_noise.main2()
    
    
# dutils.score_sparsity_ablation = True
# dutils.epochs = 100
# score_sparsity_lambdas = [0.01,0.005,0.007]
# score_sparsity_lambdas = [0.008,0.009]
# for score_sparsity_lambda in score_sparsity_lambdas:
#     dutils.score_sparsity_lambda = score_sparsity_lambda
#     dutils.save_prefix = f'score_sparsity_{score_sparsity_lambda}'
#     invert.main2()

