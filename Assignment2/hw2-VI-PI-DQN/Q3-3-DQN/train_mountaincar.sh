#!/bin/bash

python3 DQN_Implementation.py \
    --env MountainCar-v0 \
    --render \
    --memory_size 100000 \
    --frameskip 20 \
    --nos_updates 50 \
    --update_freq 1 \
    --lr 0.001
