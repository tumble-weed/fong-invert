#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import dutils
import invert_angular_simplicity
for ANGULAR_SIMPLICITY_MODE0 in ['increase','decrease']:
    #=================================================
    if ANGULAR_SIMPLICITY_MODE0 == 'none':
        invert_angular_simplicity.opts.unlock('angular_simplicity_lambda')
        invert_angular_simplicity.opts.angular_simplicity_lambda = 0
        invert_angular_simplicity.opts.lock('angular_simplicity_lambda')
        ANGULAR_SIMPLICITY_MODE = 'increase'
    else:
        ANGULAR_SIMPLICITY_MODE = ANGULAR_SIMPLICITY_MODE0
    invert_angular_simplicity.opts.unlock('ANGULAR_SIMPLICITY_MODE')
    invert_angular_simplicity.opts.ANGULAR_SIMPLICITY_MODE = ANGULAR_SIMPLICITY_MODE
    invert_angular_simplicity.opts.lock('ANGULAR_SIMPLICITY_MODE')
    #=================================================
    invert_angular_simplicity.opts.unlock('angular_lambdas')
    invert_angular_simplicity.opts.angular_lambdas = [1000 for _ in range(5)]
    invert_angular_simplicity.opts.lock('angular_lambdas')
    
    #=================================================
    invert_angular_simplicity.opts.unlock('tv_lambda')
    invert_angular_simplicity.opts.epochs = 0
    invert_angular_simplicity.opts.lock('tv_lambda')
    #=================================================
    invert_angular_simplicity.opts.unlock('loss_lambda')
    invert_angular_simplicity.opts.epochs = 0
    invert_angular_simplicity.opts.lock('loss_lambda')
    #=================================================
    invert_angular_simplicity.opts.unlock('epochs')
    invert_angular_simplicity.opts.epochs = 1000
    invert_angular_simplicity.opts.lock('epochs')
    #=================================================    
    invert_angular_simplicity.opts.unlock('target_class')
    invert_angular_simplicity.opts.epochs = 667
    invert_angular_simplicity.opts.lock('target_class')
    #=================================================
    invert_angular_simplicity.MINOR_PREFIX = f'only_angular_{ANGULAR_SIMPLICITY_MODE0}'
    invert_angular_simplicity.main2()
    
    
# dutils.score_sparsity_ablation = True
# dutils.epochs = 100
# score_sparsity_lambdas = [0.01,0.005,0.007]
# score_sparsity_lambdas = [0.008,0.009]
# for score_sparsity_lambda in score_sparsity_lambdas:
#     dutils.score_sparsity_lambda = score_sparsity_lambda
#     dutils.save_prefix = f'score_sparsity_{score_sparsity_lambda}'
#     invert.main2()

