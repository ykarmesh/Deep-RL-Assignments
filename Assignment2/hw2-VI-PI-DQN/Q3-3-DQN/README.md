## README

### DQN Implementation

Run the following command to train CartPole DQN model.
```
python3 DQN_Implementation.py \
    --env CartPole-v0 \
    --render \
    --memory_size 100000 \
    --frameskip 1 \
    --update_freq 1 \
    --lr 5e-5 \
    --epsilon 0.5 \
    --double_dqn 0

```

Run the following command to train MountainCar DQN model.
```
python3 DQN_Implementation.py \
    --env MountainCar-v0 \
    --render \
    --memory_size 100000 \
    --frameskip 20 \
    --update_freq 1 \
    --lr 0.0001 \
    --double_dqn 0
```

### Double DQN Implementation

To train the model with the Double DQN implementation, run the same above commands, but set the flag `--double_dqn=1`.

### Video Rendering

There were issues with rendering the video during the training process. The model weights were instead saved at different intervals and the video rendering was done seperately. To render the video for a given model file, set the flags `--model` with the weights path and `--render=1`.
