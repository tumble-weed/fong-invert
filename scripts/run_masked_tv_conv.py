#============================================
import os
mydir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(mydir,'..'))
import sys
sys.path.append(os.path.join(mydir,'..'))
#============================================
import utils
import invert_masked
dirs = []
# for layer in [0,1,2,3,4]:
for layer in [3,4]:
    #================================================= 
    tv_conv_factors = [0 for _ in range(5)]
    tv_conv_factors[layer] = 1e1
    invert_masked.opts.set_and_lock('tv_conv_factors',tv_conv_factors)   
    #=================================================
    invert_masked.opts.set_and_lock('tv_lambda',0)   
    invert_masked.opts.set_and_lock('loss_lambda',0)   
    #=================================================
    invert_masked.MINOR_PREFIX = f'tv_layer_{layer}'
    invert_masked.main2()
    dirs.append(utils.SAVE_DIR)
print(dirs)
