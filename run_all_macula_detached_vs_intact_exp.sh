#!/usr/bin/env bash
SESSION="macula_experiment"

tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "conda activate erdes" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/efficientnet |& tee tmux_outputs/macula_detached_vs_intact/2/efficientnet_b0_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/resnet3d |& tee tmux_outputs/macula_detached_vs_intact/2/resnet3d_training.log" C-m
#tmux send-keys -t $SESSION "python3 src/train.py trainer=gpu experiment=macula_detached_vs_intact/swinunetr |& tee tmux_outputs/macula_detached_vs_intact/2/swinunetr_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/unet3d |& tee tmux_outputs/macula_detached_vs_intact/2/unet3d_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/unetplusplus |& tee tmux_outputs/macula_detached_vs_intact/2/unetplusplus_training.log" C-m
#tmux send-keys -t $SESSION "python3 src/train.py trainer=gpu experiment=macula_detached_vs_intact/unetr |& tee tmux_outputs/macula_detached_vs_intact/2/unetr_training.log" C-m
#tmux send-keys -t $SESSION "python3 src/train.py trainer=gpu experiment=macula_detached_vs_intact/vit |& tee tmux_outputs/macula_detached_vs_intact/2/vit_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/vnet |& tee tmux_outputs/macula_detached_vs_intact/2/vnet_training.log" C-m
tmux attach -t $SESSION
