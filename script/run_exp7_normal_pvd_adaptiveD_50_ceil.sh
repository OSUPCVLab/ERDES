#!/usr/bin/env bash
SESSION="Normal vs PVD Adaptive Pool Topk 50% Ceil Experiment"
echo "$SESSION"

# Activate the correct conda environment
# Make sure `conda` is available in your shell environment
# source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to your system
conda activate erdes

# All models - topk 50
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd_adaptive_topk/unet3d
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd_adaptive_topk/resnet3d
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd_adaptive_topk/unetplusplus
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd_topk_50_ceil/vnet
# python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd_topk_50_ceil/senet
# python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd_topk_50_ceil/swinunetr
# python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd_topk_50_ceil/unetr
