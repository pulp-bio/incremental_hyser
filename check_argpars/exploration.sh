#!/bin/sh

for i in 1 2 4; do
    for b in 1 2 4; do
        for o in sgd adam; do
            python attempt.py -i $i -b $b -o $o >> out.out 2>&1
        done
    done
done
