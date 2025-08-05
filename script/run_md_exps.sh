#!/usr/bin/env bash
SESSION="Macular Status Experiment"
echo "$SESSION"

# Activate the correct conda environment
# Make sure `conda` is available in your shell environment
# source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to your system
conda activate erdes

# Uncomment the model you wish to run
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/resnet3d
# python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=macula_detached_vs_intact/swinunetr
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/unet3d
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/unetplusplus
# python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=macula_detached_vs_intact/unetr
# python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=macula_detached_vs_intact/vit
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/vnet
# python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/senet
