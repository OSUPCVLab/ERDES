#!/usr/bin/env bash
SESSION="macula_experiment"

tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "conda activate erdes" C-m
tmux send-keys -t $SESSION "cd tmux_session && mkdir -p tmux_outputs/macula_detached_vs_intact && cd .." C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/resnet3d |& tee tmux_session/tmux_outputs/macula_detached_vs_intact/resnet3d_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/swinunetr |& tee tmux_session/tmux_outputs/macula_detached_vs_intact/swinunetr_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/unet3d |& tee tmux_session/tmux_outputs/macula_detached_vs_intact/unet3d_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/unetplusplus |& tee tmux_session/tmux_outputs/macula_detached_vs_intact/unetplusplus_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=macula_detached_vs_intact/unetr |& tee tmux_session/tmux_outputs/macula_detached_vs_intact/unetr_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.strategy=ddp_find_unused_parameters_true trainer.devices=3 experiment=macula_detached_vs_intact/vit |& tee tmux_session/tmux_outputs/macula_detached_vs_intact/vit_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/vnet |& tee tmux_session/tmux_outputs/macula_detached_vs_intact/vnet_training.log" C-m
tmux send-keys -t $SESSION "python3 src/train.py trainer=ddp trainer.devices=3 experiment=macula_detached_vs_intact/senet |& tee tmux_session/tmux_outputs/macula_detached_vs_intact/senet_training.log" C-m
tmux attach -t $SESSION
