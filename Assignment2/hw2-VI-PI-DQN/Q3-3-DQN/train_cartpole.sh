#!/bin/bash

python3 DQN_Implementation.py \
    --env CartPole-v0 \
    --render \
    --memory_size 50000 \
    --frameskip 1 \
    --nos_updates 10 \
    --update_freq 10 \
    --lr 0.0001 \
    --epsilon 0.5
