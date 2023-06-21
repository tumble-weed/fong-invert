cd /root/evaluate-saliency-4/fong-invert/Grounded-Segment-Anything
# PYTHON="python -m ipdb -c c"
PYTHON="python"
# export DBG_RUN_1=1
# export DBG_DONT_SAVE=1
$PYTHON grounded_sam_demo.py --noise_mode noisy --model vgg
$PYTHON grounded_sam_demo.py --noise_mode noiseless --model vgg
$PYTHON grounded_sam_demo.py --noise_mode noisy --model alexnet
$PYTHON grounded_sam_demo.py --noise_mode noiseless --model alexnet
cd -


# VIZ=1 DBG_DONT_SAVE=1 python -m ipdb -c c grounded_sam_demo.py --noise_mode noisy --model alexnet