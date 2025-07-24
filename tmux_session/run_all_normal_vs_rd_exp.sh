#!/usr/bin/env bash
SESSION="normal_vs_rd_experiment"

tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "conda activate erdes" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/resnet3d |& tee tmux_outputs/normal_vs_rd/2/resnet3d_training.log" C-m
#tmux send-keys -t $SESSION "python3 src/train.py trainer=gpu experiment=normal_vs_rd/swinunetr |& tee tmux_outputs/normal_vs_rd/2/swinunetr_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/unet3d |& tee tmux_outputs/normal_vs_rd/2/unet3d_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/unetplusplus |& tee tmux_outputs/normal_vs_rd/2/unetplusplus_training.log" C-m
#tmux send-keys -t $SESSION "python3 src/train.py trainer=gpu experiment=normal_vs_rd/unetr |& tee tmux_outputs/normal_vs_rd/unetr_training.log" C-m
#tmux send-keys -t $SESSION "python3 src/train.py trainer=gpu experiment=normal_vs_rd/vit |& tee tmux_outputs/normal_vs_rd/vit_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/vnet |& tee tmux_outputs/normal_vs_rd/2/vnet_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/efficientnet |& tee tmux_outputs/normal_vs_rd/2/efficientnet_b0_training.log" C-m
tmux attach -t $SESSION
