# for k,v in sval.items():
#     invert_noisy.opts.set_and_lock(k,v)
# for k,v in common_setting.items():
#     invert_noisy.opts.set_and_lock(k,v)            
                
def set_opts(mod,opts_dict):
    for k,v in opts_dict.items():
        mod.opts.set_and_lock(k,v)    