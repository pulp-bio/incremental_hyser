import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--input-channels", type=int, help="Number of subsampled input channels")
parser.add_argument("-b", "--minibatch-size", type=int, help="Mini-batch size")
parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "adam"], help="Optimizer: SGD or Adam")
parser.add_argument("-r", "--results-directory", type=str, help="Target directory of the results file")

args = vars(parser.parse_args())

num_in = args['input_channels']
bs = args['minibatch_size']
optimizer_str = args['optimizer']

results_filename = f"results_in_{num_in}_batch_{bs}_opt_{optimizer_str}.pkl"
print(f"{results_filename}", flush=True)
