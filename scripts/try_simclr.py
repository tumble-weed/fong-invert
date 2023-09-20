import sys
sys.path.append('simclr-pytorch')
export IMAGENET_PATH=.../raw-data
python train.py --problem eval --eval_only true --iters 1 --arch linear --data imagenet \
--ckpt pretrained_models/resnet50_imagenet_bs2k_epochs600_linear.pth.tar \
--encoder_ckpt pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar
