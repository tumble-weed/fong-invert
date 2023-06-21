import os
import glob
import pickle
import numpy as np
import torch
def load_pickle(pklname):
    try:
        with open(pklname,'rb') as f:
            trends = pickle.load(f)
            x__ = pickle.load(f)
            opts_dict = pickle.load(f)
        return trends, x__, opts_dict
    except FileNotFoundError as e:
        if os.environ.get('IGNORE_NO_PKL',False):
            pass
        else:
            raise e
        
    
def analyze(glob1):
    # subfolders = list(glob.glob(os.path.join(parent_dir,'*_noise1/')))
    subfolders = list(glob.glob(glob1))
    all_kl_div_observer = []
    all_bbox_area = []
    all_sam_area = []
    all_scoremap_area = []
    all_n_bboxes =  []
    print(len(subfolders))
    for f in subfolders:
        # print(f)
        pklname = os.path.join(f,'trends.pkl')
        try:
            trends, x__, opts_dict = load_pickle(pklname)
        except TypeError as e:
            if os.environ.get('IGNORE_NO_PKL',False):
                continue
            else: 
                raise e
        assert not (np.isnan(trends['kl_div_observer']).any())
        if len(np.array(trends['boxes_at_thresholds'])) > 1:
            import ipdb; ipdb.set_trace()
        if len(trends['bbox_areas'])== 0:
            print(np.array(trends['bbox_areas']).shape)
            # import ipdb; ipdb.set_trace()
        #=================================================
        sam_masks = trends['sam']
        # import ipdb; ipdb.set_trace()
        if False:
            sam_masks = sam_masks.max(dim=0,keepdim=True)[0]
            sam_area = sam_masks.sum()
        elif False:
            import kornia.contrib
            cc = kornia.contrib.connected_components(sam_masks,num_iterations=100)
        else:
            # import ipdb; ipdb.set_trace()
            sam_area = sam_masks.sum(dim=(1,2,3)).max()

        
        #=================================================
        all_n_bboxes.append(len(trends['dino']['boxes']))
        all_kl_div_observer.append(trends['kl_div_observer'][-1])        
        all_bbox_area.append(trends['bbox_areas'])
        all_scoremap_area.append((trends['scoremap']>0.5).astype(np.float32).sum())
        all_sam_area.append(sam_area.item())
        
    #===========================================================
    avg_kl_div_observer = np.mean(np.array(all_kl_div_observer))
    # import ipdb; ipdb.set_trace()
    avg_bbox_area = np.mean(np.array(all_bbox_area))
    avg_scoremap_area = np.mean(np.array(all_scoremap_area))
    avg_n_bboxes = np.mean(all_n_bboxes)
    avg_sam_area = np.mean(all_sam_area)
    #===========================================================
    print("-"*50)
    print(glob1)
    print('avg_kl_div_observer',avg_kl_div_observer)
    print('avg_bbox_area',avg_bbox_area)
    print('avg_scoremap_area',avg_scoremap_area)
    print('avg n_bboxes',avg_n_bboxes)
    print('avg sam area',avg_sam_area)
    print("-"*50)
# parent_dir = "/root/evaluate-saliency-4/fong-invert/debugging/run_multi_class_best_alexnet/"
parent_dir = "/root/evaluate-saliency-4/fong-invert/debugging/invert_noisy/"
#====================================================================+
for modelname in [
                    # 'alexnet',
                  'vgg'
                  ]:
    for noise_type in ['noisy','noiseless']:
        pattern = os.path.join(parent_dir,f'run_multi_class_best_{modelname}/*{noise_type}*/')
        analyze(pattern)
#====================================================================+
# glob1 = os.path.join(parent_dir,"*_noise0/")
# print(glob1)

if False:
    parent_dir = "/root/evaluate-saliency-4/fong-invert/debugging/invert_noisy/run_multi_class_best_alexnet/"
    glob1 = os.path.join(parent_dir,"*_noise1/")
    # print(glob1)
    analyze(glob1)
