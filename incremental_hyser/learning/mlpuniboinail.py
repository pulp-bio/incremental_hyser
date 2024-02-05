from __future__ import annotations

import torch  # for tensors
from torch import nn
import torchinfo


"""

This module implements in PyTorch the ... TEMPONET...

...citation...

Differences:
- ...
- ...

"""


class TEMPONet(nn.Module):
    
    def __init__(self,
            num_ch_in: int,
            num_ch_out: int,
            pdrop_fc: int=0.5,
        ):
        super().__init__()
        
        self.num_ch_in = num_ch_in
        self.num_ch_out = num_ch_out
        
        self.b0_tcn0      = nn.Conv1d(self.Cx, 16, 3, dilation=2, padding=2, bias=False)
        self.b0_tcn0_BN   = nn.BatchNorm1d(16)
        self.b0_tcn0_ReLU = nn.ReLU()
        self.b0_tcn1      = nn.Conv1d(16, 16, 3, dilation=2, padding=2, bias=False)
        self.b0_tcn1_BN   = nn.BatchNorm1d(16)
        self.b0_tcn1_ReLU = nn.ReLU()
        self.b0_conv      = nn.Conv1d(16, 16, 5, stride=1, padding=2, bias=False)
        self.b0_conv_pool = torch.nn.AvgPool1d(2, stride=2, padding=0)
        self.b0_conv_BN   = nn.BatchNorm1d(16)
        self.b0_conv_ReLU = nn.ReLU()

        self.b1_tcn0      = nn.Conv1d(16, 32, 3, dilation=4, padding=4, bias=False)
        self.b1_tcn0_BN   = nn.BatchNorm1d(32)
        self.b1_tcn0_ReLU = nn.ReLU()
        self.b1_tcn1      = nn.Conv1d(32, 32, 3, dilation=4, padding=4, bias=False)
        self.b1_tcn1_BN   = nn.BatchNorm1d(32)
        self.b1_tcn1_ReLU = nn.ReLU()
        self.b1_conv      = nn.Conv1d(32, 32, 5, stride=2, padding=2, bias=False)
        self.b1_conv_pool = torch.nn.AvgPool1d(2, stride=2, padding=0)
        self.b1_conv_BN   = nn.BatchNorm1d(32)
        self.b1_conv_ReLU = nn.ReLU()

        self.b2_tcn0      = nn.Conv1d(32, 64, 3, dilation=8, padding=8, bias=False)
        self.b2_tcn0_BN   = nn.BatchNorm1d(64)
        self.b2_tcn0_ReLU = nn.ReLU()
        self.b2_tcn1      = nn.Conv1d(64, 64, 3, dilation=8, padding=8, bias=False)
        self.b2_tcn1_BN   = nn.BatchNorm1d(64)
        self.b2_tcn1_ReLU = nn.ReLU()
        self.b2_conv      = nn.Conv1d(64, 64, 5, stride=4, padding=2, bias=False)
        self.b2_conv_pool = torch.nn.AvgPool1d(2, stride=2, padding=0)
        self.b2_conv_BN   = nn.BatchNorm1d(64)
        self.b2_conv_ReLU = nn.ReLU()
        
        self.fc0         = nn.Linear(64 * 4, 64, bias=False)
        self.fc0_bn      = nn.BatchNorm1d(64)
        self.fc0_relu    = nn.ReLU()
        self.fc0_drop = nn.Dropout(0.5)

        self.fc0         = nn.Linear(64 * 4, 64, bias=False)
        self.fc0_bn      = nn.BatchNorm1d(64)
        self.fc0_relu    = nn.ReLU()
        self.fc0_drop = nn.Dropout(0.5)
        
     
        self.fc2 = nn.Linear(32, self.num_ch_out, bias=False)
        
    def forward(self, x):

        x = self.b0_tcn0_ReLU(self.b0_tcn0_BN(                  self.b0_tcn0(x) ))
        x = self.b0_tcn1_ReLU(self.b0_tcn1_BN(                  self.b0_tcn1(x) ))
        x = self.b0_conv_ReLU(self.b0_conv_BN(self.b0_conv_pool(self.b0_conv(x))))

        x = self.b1_tcn0_ReLU(self.b1_tcn0_BN(                  self.b1_tcn0(x) ))
        x = self.b1_tcn1_ReLU(self.b1_tcn1_BN(                  self.b1_tcn1(x) ))
        x = self.b1_conv_ReLU(self.b1_conv_BN(self.b1_conv_pool(self.b1_conv(x))))
        
        x = self.b2_tcn0_ReLU(self.b2_tcn0_BN(                  self.b2_tcn0(x) ))
        x = self.b2_tcn1_ReLU(self.b2_tcn1_BN(                  self.b2_tcn1(x) ))
        x = self.b2_conv_ReLU(self.b2_conv_BN(self.b2_conv_pool(self.b2_conv(x))))
        
        x = x.flatten(1)
        x = self.FC0_dropout(self.FC0_ReLU(self.FC0_BN(self.FC0(x))))
        x = self.FC1_dropout(self.FC1_ReLU(self.FC1_BN(self.FC1(x))))
        y = self.GwayFC(x)
        
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
    temponet = TEMPONet()
    temponet.eval()
    mlp_model_stats = summarize(temponet, verbose=verbose)


if __name__ == "__main__":
    main()
