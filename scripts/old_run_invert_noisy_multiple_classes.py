#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import dutils
import invert_noisy
import importlib
import utils
print = utils.printl
# import multiprocessing as mp
# mp.set_start_method('spawn')
setting = {
    'noisy':{
        'noise_mag':1,
        'JITTER':False,
        'jitter':0,
        'tv_lambda':1e-3,
        'MAX_SCALE':1.2,
        'epochs':2000,
        'network':'vgg19',
    },
    # 'noiseless':{   
    #     'noise_mag':0,
    #     'JITTER':True,
    #     'jitter':5,
    #     'tv_lambda':1e-3,
    # }
}
if os.environ.get('EPOCHS',False):
    setting['noisy']['epochs'] = int(os.environ['EPOCHS'])
classes = [
    # 1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 
    # 277, 283, 292, 294, 309,311,325, 
    340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
    967, 574, 487]
for ZOOM_JITTER in [True,False]:
    for sname,sval in setting.items():
        # if sname == 'noiseless':
        #     continue
        
        for label in classes:
            if label in [1,933] and ZOOM_JITTER:
                continue
            #=================================================    
            # importlib.reload(invert_noisy)
            # invert_noisy.opts.set_and_lock('noise_mag',sval['noise_mag'])
            # invert_noisy.opts.set_and_lock('JITTER',sval['JITTER'])
            # invert_noisy.opts.set_and_lock('jitter',sval['jitter'])
            # invert_noisy.opts.set_and_lock('MAX_SCALE',sval['MAX_SCALE'])
            # invert_noisy.opts.set_and_lock('tv_lambda',sval['tv_lambda'])            
            for k,v in sval.items():
                invert_noisy.opts.set_and_lock(k,v)
            
            invert_noisy.opts.set_and_lock('target_class',label)
            # invert_noisy.opts.set_and_lock('layer','features.11')
            invert_noisy.opts.set_and_lock('ZOOM_JITTER',ZOOM_JITTER)
            invert_noisy.MINOR_PREFIX = f'label_{label}_zoom_{ZOOM_JITTER}'
            invert_noisy.main2()

            # q = mp.Queue()
            # p = mp.Process(target=invert_noisy.main2, args=(q,))
            # p.start()
            # print(q.get())
            # p.join()            
    
    
        
        
    # dutils.score_sparsity_ablation = True
    # dutils.epochs = 100
    # score_sparsity_lambdas = [0.01,0.005,0.007]
    # score_sparsity_lambdas = [0.008,0.009]
    # for score_sparsity_lambda in score_sparsity_lambdas:
    #     dutils.score_sparsity_lambda = score_sparsity_lambda
    #     dutils.save_prefix = f'score_sparsity_{score_sparsity_lambda}'
    #     invert.main2()

