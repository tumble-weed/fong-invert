
if False:
    """
    NOISE-JITTER
    """
    CUDA_VISIBLE_DEVICES=0 DBG_NOISE_JITTER=1 python -m ipdb -c c scripts/run_multi_class.py

if False:
    """
    Observer
    """
    CUDA_VISIBLE_DEVICES=0 DBG_OBSERVER=1 python -m ipdb -c c scripts/run_multi_class.py

if True:
    #===================================================================================
    CUDA_VISIBLE_DEVICES=0  python -m ipdb -c c scripts/run_multi_class.py --experiment best_alexnet --dry_run
    
    CUDA_VISIBLE_DEVICES=1  python -m ipdb -c c scripts/run_multi_class.py --experiment compare_zoom_alexnet
    
    CUDA_VISIBLE_DEVICES=1  python -m ipdb -c c scripts/run_multi_class.py --experiment noise_ablation_alexnet
    #===================================================================================
    
if True and 'multithresh':
    CUDA_VISIBLE_DEVICES=0  python -m ipdb -c c multithresh_saliency.py
    
if True:
    python -m ipdb analyze_results.py --parent_dir

if "BG-FG-MASK" and True:
    CUDA_VISIBLE_DEVICES=1  python -m ipdb -c c scripts/run_bg_fg.py --experiment use_mask
if "BG-FG" and True:
    # DBG_ALIGN=1 CUDA_VISIBLE_DEVICES=1  python -m ipdb -c c scripts/run_bg_fg.py --experiment align
    
    CUDA_VISIBLE_DEVICES=0  python -m ipdb -c c scripts/run_bg_fg.py --experiment align
if "GPNN" and True:
    """
    what is the difference between my_gpnn and my_gpnn_fast, which one did i use? i should check in gpnn_gradcam_*
    
    what is the difference between gpnn_gradcam_multi and ?
    PATHS:
        /root/evaluate-saliency-4/GPNN/gpnn_gradcam_multi.py
        /root/evaluate-saliency-4/GPNN/benchmark/pascal_run_competing_saliency_librecam.py
    """ 
    PYTHONPATH=/root/evaluate-saliency-4/GPNN:$PYTHONPATH python -m GPNN_model.my_gpnn 
if "DEEPINVERSION-imagenet" and True:
    python imagenet_inversion.py --bs=84 --do_flip --exp_name="rn50_inversion" --r_feature=0.01 --arch_name="resnet50" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25

if "DEEPINVERSION-NERF" and True:
    """
    This snippet will generate 84 images by inverting resnet50 model from torchvision package.
    """
    

    
    """
    This snippet will generate 1 nerf model
    """    
    if False:
        CUDA_VISIBLE_DEVICES=1 USE_NERF=1 python -m ipdb -c c imagenet_inversion.py --bs=21 --do_flip --exp_name="rn50_inversion" --r_feature=0.01 --arch_name="resnet50" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25
    
    USE_NERF=1 python -m ipdb -c c imagenet_inversion.py --bs=21 --do_flip --exp_name="rn50_inversion_nerf" --r_feature=0.01 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25
    pass

if "DEEPINVERSION-CUBS" and True:
    CUDA_VISIBLE_DEVICES=1 python -m ipdb -c c cubs_inversion.py --bs=7 --do_flip --exp_name="cubs_inversion" --r_feature=0.01 --arch_name="cubs" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25
    
if "DEEPINVERSION-PR" and True:
    USE_PR=1 CUDA_VISIBLE_DEVICES=1 python -m ipdb -c c imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion" --r_feature=0.01 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25
if "best-alexnet" and True:
    DBG_FEW_CLASSES=1  python -m ipdb -c c scripts/run_multi_class.py --experiment best_alexnet
    python -m ipdb -c c scripts/run_multi_class.py --experiment best_alexnet
    python -m ipdb -c c scripts/run_multi_class.py --experiment best_vgg