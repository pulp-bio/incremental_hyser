#!/bin/sh

# launch of the experiment(s)

# CUDA_VISIBLE_DEVICES=3 nohup python -u experiment.py --m baseline -f retest_jun2024_baseline.pkl > retest_jun2024_baseline.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u experiment.py --m online -f retest_jun2024_online.pkl > retest_jun2024_online.out 2>&1 &


# SETTINGS FOR BASELINE:
# adam, 8 epochs, minibatch 32, lr 0.0001, weight_decay=0.0, slide 64, training randomized
