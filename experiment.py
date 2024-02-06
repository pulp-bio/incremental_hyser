#!/usr/bin/env python

import sys
import argparse

from incremental_hyser.hyser import hyser as hy
from incremental_hyser.protocol.protocol import experiment_all_subjects



def parse_my_args(argv=None) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-channels", type=int, help="Number of subsampled input channels")
    parser.add_argument("-b", "--minibatch-size", type=int, help="Mini-batch size")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "adam"], help="Optimizer: SGD or Adam")
    parser.add_argument("-r", "--results-directory", type=str, help="Target directory of the results file")
    args = parser.parse_args(argv)

    return args


def main() -> None:

    if __name__ != "__main__":
        # running as a module, so add a dummy argument
        sys.argv = ['experiment.py'] + sys.argv

    print(sys.argv)
    args = parse_my_args()
    print(args)

    experiment_all_subjects()

    return


if __name__ == "__main__":
    main()
