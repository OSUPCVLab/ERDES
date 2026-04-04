#!/usr/bin/env bash
SESSION="Normal vs PVD Experiment"
echo "$SESSION"

# Activate the correct conda environment
# Make sure `conda` is available in your shell environment
# source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to your system
conda activate erdes

# Uncomment the model you wish to run
# python3 erdes/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd/resnet3d
# python3 erdes/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd/swinunetr
# python3 erdes/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd/unetplusplus
# python3 erdes/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd/unetr
# python3 erdes/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_pvd/vit
# python3 erdes/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd/vnet
# python3 erdes/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_pvd/senet

