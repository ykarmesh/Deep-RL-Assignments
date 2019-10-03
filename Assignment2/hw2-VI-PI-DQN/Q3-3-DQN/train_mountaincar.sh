#!/bin/bash

python3 DQN_Implementation.py \
    --env MountainCar-v0 \
    --render \
    --memory_size 100000 \
    --frameskip 20 \
    --update_freq 1 \
    --lr 0.0001 \
    --double_dqn 1
