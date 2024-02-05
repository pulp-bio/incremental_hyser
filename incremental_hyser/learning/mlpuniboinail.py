from __future__ import annotations

import torch  # for tensors
from torch import nn
import torchinfo


"""

###############################################################################
--> FOR THE EXTENSION, THIS ONLY HOLDS FOR hidden=8!!!
--> decide heurostic for the other dataset(s): 8, +4, x2, **1.5, ...
###############################################################################


This module implements in PyTorch the 2-layer Multi-Layer Perceptron (MLP)
termed Neural Network (NN) and used in the paper:

B. Milosevic, E. Farella, S. Benatti,
Exploring Arm Posture and Temporal Variability in Myoelectric Hand Gesture
Recognition
https://doi.org/10.1109/BIOROB.2018.8487838

Differences:
- ReLU non-linear activation function is used instead of sigmoid;
- Batch-Normalization (BN) is added;
- the structure is adapted to be compliant with the DNN quantization tool
  quantlib (https://github.com/pulp-platform/quantlib): no biases, no final
  softmax inside the model, so that the model ends with a biasless linear.

"""


class MLPUniboINAIL(nn.Module):

    def __init__(self, num_input, num_hidden, num_output):
        super(MLPUniboINAIL, self).__init__()

        self.fc0 = nn.Linear(num_input, num_hidden, bias=False)
        self.fc0_bn = nn.BatchNorm1d(num_hidden)
        self.fc0_relu = nn.ReLU()
        self.fc1 = nn.Linear(num_hidden, num_output, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc0(x)
        x = self.fc0_bn(x)
        x = self.fc0_relu(x)
        y = self.fc1(x)
        return y


def summarize(
    model: nn.Module,
    num_input: int,
    verbose: 0 | 1 | 2 = 0,
) -> torchinfo.ModelStatistics:

    # set all parameters for torchsummary

    input_size = (num_input,)
    batch_dim = 0  # index of the batch dimension
    col_names = [
        'input_size',
        'output_size',
        'num_params',
        'params_percent',
        'kernel_size',
        'mult_adds',
        'trainable',
    ]
    device = 'cpu'
    mode = 'eval'
    row_settings = [
        'ascii_only',
        'depth',
        'var_names',
    ]

    # call the summary function

    model_stats = torchinfo.summary(
        model=model,
        input_size=input_size,
        batch_dim=batch_dim,
        col_names=col_names,
        device=device,
        mode=mode,
        row_settings=row_settings,
        verbose=verbose,
    )

    return model_stats


def main() -> None:

    # Display the summary of the MLP

    verbose = 1
    mlp_unibo_inail = MLPUniboINAIL()
    mlp_unibo_inail.eval()
    mlp_model_stats = summarize(mlp_unibo_inail, verbose=verbose)


if __name__ == '__main__':
    main()
