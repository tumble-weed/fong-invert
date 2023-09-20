#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import dutils
import invert_angular_simplicity
for target_class in [100,667,371,400]:
    #=================================================
    invert_angular_simplicity.opts.unlock('target_class')
    invert_angular_simplicity.opts.target_class = target_class
    invert_angular_simplicity.opts.lock('target_class')
    #=================================================
    invert_angular_simplicity.opts.unlock('epochs')
    invert_angular_simplicity.opts.epochs = 1000
    invert_angular_simplicity.opts.lock('epochs')
    #=================================================
    
    invert_angular_simplicity.MINOR_PREFIX = f'target_class_{target_class}'
    invert_angular_simplicity.main2()
    
    
# dutils.score_sparsity_ablation = True
# dutils.epochs = 100
# score_sparsity_lambdas = [0.01,0.005,0.007]
# score_sparsity_lambdas = [0.008,0.009]
# for score_sparsity_lambda in score_sparsity_lambdas:
#     dutils.score_sparsity_lambda = score_sparsity_lambda
#     dutils.save_prefix = f'score_sparsity_{score_sparsity_lambda}'
#     invert.main2()

