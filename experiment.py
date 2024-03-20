#!/usr/bin/env python

import sys
import argparse

from incremental_hyser.hyser import hyser as hy
from incremental_hyser.protocol.protocol import experiment_all_subjects



def parse_my_args(argv=None) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # learning mode: offline baseline vs. online
    parser.add_argument(
        '-m', '--mode',
        nargs='?', type=str, choices=["baseline", "online"], required=True,
        help="Learning mode: offline baseline or online",
    )

    # name of the results' destination file
    parser.add_argument(
        '-f', '--filename',
        nargs="?", type=str,
        help="Name of the results file",
    )

    args = parser.parse_args(argv)

    return args


def main() -> None:

    if __name__ != "__main__":
        # running as a module, so (re)add the first argument
        sys.argv = ['experiment.py'] + sys.argv

    args = parse_my_args()

    # print a summary, also for checking
    print("\n\nsys.argv = \n")
    print(sys.argv)
    print("\n\nargs = \n")
    print(args)
    print("\n\n")

    # (NB: names changed)
    learning_mode = args.mode
    results_filename = args.filename
    experiment_all_subjects(
        learning_mode=learning_mode,
        results_filename=results_filename,
    )

    return


if __name__ == "__main__":
    main()
