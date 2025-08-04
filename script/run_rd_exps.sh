#!/usr/bin/env bash
SESSION="Retinal Status Experiments"
echo "$SESSION"

# Activate the correct conda environment
# Make sure `conda` is available in your shell environment
# source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to your system
conda activate erdes

# Uncomment the model you wish to run
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=non_rd_vs_rd/resnet3d 
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=non_rd_vs_rd/swinunetr
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=non_rd_vs_rd/unet3d
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=non_rd_vs_rd/unetplusplus
# python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=non_rd_vs_rd/unetr
# python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=non_rd_vs_rd/vit
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=non_rd_vs_rd/vnet
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=non_rd_vs_rd/senet

