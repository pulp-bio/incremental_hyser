#!/usr/bin/env python

import sys
import argparse

from incremental_hyser.hyser import hyser as hy
from incremental_hyser.protocol.protocol import experiment_all_subjects



def parse_my_args(argv=None) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-channels', type=int, help="Number of subsampled input channels")
    parser.add_argument('-b', '--minibatch-size', type=int, help="Mini-batch size")
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'adam'], help="Optimizer: SGD or Adam")
    parser.add_argument('-d', '--results-directory', type=str, help="Destination directory of the results file")
    parser.add_argument('-f', '--results-filename', type=str, help="Name of the results file")
    args = parser.parse_args(argv)

    return args


def main() -> None:

    if __name__ != "__main__":
        # running as a module, so (re)add the first argument
        sys.argv = ['experiment.py'] + sys.argv

    print(sys.argv)
    args = parse_my_args()
    print(args)

    args = vars(args)
    input_channels = args['input_channels']
    minibatch_size = args['minibatch_size']
    optimizer_str = args['optimizer_str']
    results_directory = args['results_directory']
    results_filename = args['results_filename']
    experiment_all_subjects(
        input_channels=input_channels,
        minibatch_size=minibatch_size,
        optimizer_str=optimizer_str,
        results_directory=results_directory,
        results_filename=results_filename,
    )

    return


if __name__ == "__main__":
    main()
