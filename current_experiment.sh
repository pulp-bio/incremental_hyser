#!/bin/sh

# command to launch the current experiment
# (to be edited each time)


# CUDA_VISIBLE_DEVICES=0 nohup python -u experiment.py --mode baseline -f baseline_feb11.pkl > out_baseline.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u experiment.py --mode online -f online_feb11_0002.pkl > out_online_0002.out 2>&1 &


# SETTINGS FOR BASELINE:
# adam, 8 epochs, minibatch 32, lr 0.0001, weight_decay=0.0, slide 63, training randomized
