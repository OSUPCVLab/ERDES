#!/usr/bin/env bash
SESSION="normal_vs_rd_experiment"

tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "conda activate erdes" C-m
tmux send-keys -t $SESSION "cd tmux_session && mkdir -p tmux_outputs/normal_vs_rd && cd .." C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/resnet3d |& tee tmux_session/tmux_outputs/normal_vs_rd/resnet3d_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/swinunetr |& tee tmux_session/tmux_outputs/normal_vs_rd/swinunetr_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/unet3d |& tee tmux_session/tmux_outputs/normal_vs_rd/unet3d_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/unetplusplus |& tee tmux_session/tmux_outputs/normal_vs_rd/unetplusplus_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_rd/unetr |& tee tmux_session/tmux_outputs/normal_vs_rd/unetr_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=normal_vs_rd/vit |& tee tmux_session/tmux_outputs/normal_vs_rd/vit_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/vnet |& tee tmux_session/tmux_outputs/normal_vs_rd/vnet_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=normal_vs_rd/senet |& tee tmux_session/tmux_outputs/normal_vs_rd/senet_training.log" C-m
tmux attach -t $SESSION
