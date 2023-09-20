#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import dutils
import invert_x_diversity
for JS in [True,False]:
    for diversity_lambda in [10,30,100]:
        #=================================================    
        invert_x_diversity.opts.unlock('diversity_lambda')
        invert_x_diversity.opts.diversity_lambda = diversity_lambda
        invert_x_diversity.opts.lock('diversity_lambda')
        #=================================================    
        invert_x_diversity.opts.unlock('JS')
        invert_x_diversity.opts.JS = JS
        invert_x_diversity.opts.lock('JS')
        #=================================================
        invert_x_diversity.MINOR_PREFIX = f'diversity_lambda_{diversity_lambda}{"_JS" if JS else ""}'
        invert_x_diversity.main2()
    
    
# dutils.score_sparsity_ablation = True
# dutils.epochs = 100
# score_sparsity_lambdas = [0.01,0.005,0.007]
# score_sparsity_lambdas = [0.008,0.009]
# for score_sparsity_lambda in score_sparsity_lambdas:
#     dutils.score_sparsity_lambda = score_sparsity_lambda
#     dutils.save_prefix = f'score_sparsity_{score_sparsity_lambda}'
#     invert.main2()

