#!/bin/bash

python3 DQN_Implementation.py \
    --env CartPole-v0 \
    --render \
    --memory_size 100000 \
    --frameskip 1 \
    --nos_updates 20 \
    --update_freq 10 \
    --lr 0.00005 \
    --epsilon 0.5
