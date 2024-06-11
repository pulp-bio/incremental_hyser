#!/bin/sh

# launch of the experiment(s)

CUDA_VISIBLE_DEVICES=3 nohup python -u experiment.py --m online -f fewchannels_04ch_mar25_online.pkl > out_fewchannels_04ch_mar25_online.out 2>&1 &

# SETTINGS FOR BASELINE:
# adam, 8 epochs, minibatch 32, lr 0.0001, weight_decay=0.0, slide 64, training randomized
