#!/bin/bash
args="-update_freq 1
        -mb_dim 320
        -refresh 5e2
        -num_steps 1e5
        -use_egreedy
        -target_refresh 1e3
        -learn_start 1
        -clip_delta
        -gamma .9
        -environment combolock
        -q_learning_rate 1e-3
        -num_frames 1
        "
th train.lua $args
