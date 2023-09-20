#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import utils
import invert_x_diversity
dirs = []
for JS in [True,False]:
    # assert False,'find good lambda from run_x_diversity'
    for diversity_lambda in [10,30]:
        for temp in [1,3,10,100]:
            if JS:
                if diversity_lambda > 10:
                    continue
            #================================================= 
            invert_x_diversity.opts.set_and_lock('diversity_lambda',diversity_lambda)   
            # invert_x_diversity.opts.unlock('diversity_lambda')
            # invert_x_diversity.opts.diversity_lambda = diversity_lambda
            # invert_x_diversity.opts.lock('diversity_lambda')
            #=================================================    
            invert_x_diversity.opts.set_and_lock('T',temp)
            # invert_x_diversity.opts.unlock('T')
            # invert_x_diversity.opts.T = temp
            # invert_x_diversity.opts.lock('T')
            #=================================================    
            invert_x_diversity.opts.set_and_lock('JS',JS)
            # invert_x_diversity.opts.unlock('JS')
            # invert_x_diversity.opts.JS = JS
            # invert_x_diversity.opts.lock('JS')
            #=================================================
            invert_x_diversity.MINOR_PREFIX = f'diversity_lambda_{diversity_lambda}{"JS" if JS else ""}_T_{temp}'
            invert_x_diversity.main2()
            dirs.append(utils.SAVE_DIR)
print(dirs)
# dutils.score_sparsity_ablation = True
# dutils.epochs = 100
# score_sparsity_lambdas = [0.01,0.005,0.007]
# score_sparsity_lambdas = [0.008,0.009]
# for score_sparsity_lambda in score_sparsity_lambdas:
#     dutils.score_sparsity_lambda = score_sparsity_lambda
#     dutils.save_prefix = f'score_sparsity_{score_sparsity_lambda}'
#     invert.main2()

