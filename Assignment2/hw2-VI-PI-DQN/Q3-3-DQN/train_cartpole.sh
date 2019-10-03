#!/bin/bash

python3 DQN_Implementation.py \
    --env CartPole-v0 \
    --render \
    --memory_size 100000 \
    --frameskip 1 \
    --update_freq 1 \
    --lr 5e-5 \
    --epsilon 0.5 \
    --double_dqn 1
