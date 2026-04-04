#!/usr/bin/env bash
SESSION="Normal vs PVD Adaptive Pool Topk 30% Ceil Experiment"
echo "$SESSION"

# Activate the correct conda environment
# Make sure `conda` is available in your shell environment
# source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to your system
conda activate erdes

# All models - topk 30
# python3 erdes/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd_topk_30_ceil/unet3d
# python3 erdes/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd_topk_30_ceil/resnet3d
# python3 erdes/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd_topk_30_ceil/unetplusplus
# python3 erdes/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd_topk_30_ceil/vnet
# python3 erdes/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd_topk_30_ceil/senet
# python3 erdes/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd_topk_30_ceil/swinunetr
# python3 erdes/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd_topk_30_ceil/unetr
