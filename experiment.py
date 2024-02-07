#!/usr/bin/env python

import sys
import argparse

from incremental_hyser.hyser import hyser as hy
from incremental_hyser.protocol.protocol import experiment_all_subjects



def parse_my_args(argv=None) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--inchannels', type=int, help="Subsampled input channels")
    parser.add_argument('-b', '--minibatch', type=int, help="Mini-batch size")
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'adam'], help="Optimizer: SGD or Adam")
    parser.add_argument('-d', '--dirname', type=str, default="./results", help="Destination directory of results file")
    parser.add_argument('-f', '--filename', type=str, help="Name of the results file")
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
    # names changed here
    input_channels = args['inchannels']
    minibatch_size = args['minibatch']
    optimizer_str = args['optimizer']
    results_directory = args['dirname']
    results_filename = args['filename']
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
