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
            pdrop_fc: int=0.0,
        ):
        super().__init__()
        
        self.num_ch_in = num_ch_in
        self.num_ch_out = num_ch_out
        self.pdrop_fc = pdrop_fc
        
        self.b0_tcn0      = nn.Conv1d(self.num_ch_in, 16, 3, padding='same', bias=False)
        self.b0_tcn0_bn   = nn.BatchNorm1d(16)
        self.b0_tcn0_relu = nn.ReLU()
        self.b0_tcn1      = nn.Conv1d(16, 16, 3, padding='same', bias=False)
        self.b0_tcn1_bn   = nn.BatchNorm1d(16)
        self.b0_tcn1_relu = nn.ReLU()
        self.b0_conv      = nn.Conv1d(16, 16, 3, padding='same', bias=False)
        self.b0_conv_bn   = nn.BatchNorm1d(16)
        self.b0_conv_relu = nn.ReLU()
        self.b0_conv_pool = nn.AvgPool1d(2, stride=2, padding='valid')
        
        self.b1_tcn0      = nn.Conv1d(16, 32, 3, padding='same', bias=False)
        self.b1_tcn0_bn   = nn.BatchNorm1d(32)
        self.b1_tcn0_relu = nn.ReLU()
        self.b1_tcn1      = nn.Conv1d(32, 32, 3, padding='same', bias=False)
        self.b1_tcn1_bn   = nn.BatchNorm1d(32)
        self.b1_tcn1_relu = nn.ReLU()
        self.b1_conv      = nn.Conv1d(32, 32, 3, padding='same', bias=False)
        self.b1_conv_bn   = nn.BatchNorm1d(32)
        self.b1_conv_relu = nn.ReLU()
        self.b1_conv_pool = nn.AvgPool1d(2, stride=2, padding='valid')
        
        self.b2_tcn0      = nn.Conv1d(32, 64, 3, padding='same', bias=False)
        self.b2_tcn0_bn   = nn.BatchNorm1d(64)
        self.b2_tcn0_relu = nn.ReLU()
        self.b2_tcn1      = nn.Conv1d(64, 64, 3, padding='same', bias=False)
        self.b2_tcn1_bn   = nn.BatchNorm1d(64)
        self.b2_tcn1_relu = nn.ReLU()
        self.b2_conv      = nn.Conv1d(64, 64, 3, padding='same', bias=False)
        self.b2_conv_bn   = nn.BatchNorm1d(64)
        self.b2_conv_relu = nn.ReLU()
        self.b2_conv_pool = nn.AvgPool1d(2, stride=2, padding=0)
        
        self.fc0 = nn.Linear(64 * 4, 64, bias=False)
        self.fc0_bn = nn.BatchNorm1d(64)
        self.fc0_relu = nn.ReLU()
        self.fc0_drop = nn.Dropout(self.pdrop_fc)

        self.fc1 = nn.Linear(64, 64, bias=False)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(self.pdrop_fc)

        self.fc2 = nn.Linear(32, self.num_ch_out, bias=False)
        
    def forward(self, x):

        x =                   self.b0_tcn0_relu(self.b0_tcn0_bn(self.b0_tcn0(x)))
        x =                   self.b0_tcn1_relu(self.b0_tcn1_bn(self.b0_tcn1(x)))
        x = self.b0_conv_pool(self.b0_conv_relu(self.b0_conv_bn(self.b0_conv(x))))

        x =                   self.b1_tcn0_relu(self.b1_tcn0_bn(self.b1_tcn0(x)))
        x =                   self.b1_tcn1_relu(self.b1_tcn1_bn(self.b1_tcn1(x)))
        x = self.b1_conv_pool(self.b1_conv_relu(self.b1_conv_bn(self.b1_conv(x))))
        
        x =                   self.b2_tcn0_relu(self.b2_tcn0_bn(self.b2_tcn0(x)))
        x =                   self.b2_tcn1_relu(self.b2_tcn1_bn(self.b2_tcn1(x)))
        x = self.b2_conv_pool(self.b2_conv_relu(self.b2_conv_bn(self.b2_conv(x))))

        x = x.flatten(1)
        x = self.fc0_drop(self.fc0_relu(self.fc0_bn(self.fc0(x))))
        x = self.fc1_drop(self.fc1_relu(self.fc1_bn(self.fc1(x))))
        y = self.fc2(x)
        
        return y


class TinierNet8(nn.Module):
    
    def __init__(self,
            num_ch_in: int,
            num_ch_out: int,
            pdrop_fc: int=0.0,
        ):
        super().__init__()
        
        self.num_ch_in = num_ch_in
        self.num_ch_out = num_ch_out
        self.pdrop_fc = pdrop_fc
        
        self.b0_conv = nn.Conv1d(self.num_ch_in, 32, 3, padding='valid', stride=2, bias=False)
        self.b0_bn = nn.BatchNorm1d(32)
        self.b0_relu = nn.ReLU()

        self.b1_conv = nn.Conv1d(32, 32, 3, padding='valid', stride=2, bias=False)
        self.b1_bn = nn.BatchNorm1d(32)
        self.b1_relu = nn.ReLU()

        self.b2_conv = nn.Conv1d(32, 32, 3, padding='valid', stride=2, bias=False)
        self.b2_bn = nn.BatchNorm1d(32)
        self.b2_relu = nn.ReLU()

        self.b3_conv = nn.Conv1d(32, 32, 3, padding='valid', stride=2, bias=False)
        self.b3_bn = nn.BatchNorm1d(32)
        self.b3_relu = nn.ReLU()

        self.b4_conv = nn.Conv1d(32, 64, 3, padding='valid', stride=2, bias=False)
        self.b4_bn = nn.BatchNorm1d(64)
        self.b4_relu = nn.ReLU()

        self.b5_conv = nn.Conv1d(64, 64, 3, padding='valid', stride=2, bias=False)
        self.b5_bn = nn.BatchNorm1d(64)
        self.b5_relu = nn.ReLU()

        self.b6_conv = nn.Conv1d(64, 64, 3, padding='valid', stride=2, bias=False)
        self.b6_bn = nn.BatchNorm1d(64)
        self.b6_relu = nn.ReLU()

        #self.b7_conv = nn.Conv1d(64, 64, 3, padding='valid', stride=2, bias=False)
        #self.b7_bn = nn.BatchNorm1d(1)
        #self.b7_relu = nn.ReLU()
        
        self.fc0 = nn.Linear(64, 64, bias=False)
        self.fc0_bn = nn.BatchNorm1d(64)
        self.fc0_relu = nn.ReLU()
        self.fc0_drop = nn.Dropout(self.pdrop_fc)

        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc1_bn = nn.BatchNorm1d(32)
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(self.pdrop_fc)

        self.fc2 = nn.Linear(32, self.num_ch_out, bias=False)
        
    def forward(self, x):

        x = self.b0_relu(self.b0_bn(self.b0_conv(x)))
        x = self.b1_relu(self.b1_bn(self.b1_conv(x)))
        x = self.b2_relu(self.b2_bn(self.b2_conv(x)))
        x = self.b3_relu(self.b3_bn(self.b3_conv(x)))
        x = self.b4_relu(self.b4_bn(self.b4_conv(x)))
        x = self.b5_relu(self.b5_bn(self.b5_conv(x)))
        x = self.b6_relu(self.b6_bn(self.b6_conv(x)))
        #x = self.b7_relu(self.b7_bn(self.b7_conv(x)))

        x = x.flatten(1)
        x = self.fc0_drop(self.fc0_relu(self.fc0_bn(self.fc0(x))))
        x = self.fc1_drop(self.fc1_relu(self.fc1_bn(self.fc1(x))))
        y = self.fc2(x)
        
        return y


def summarize(
    model: nn.Module,
    num_ch_in: int,
    num_samples_in: int,
    verbose: 0 | 1 | 2 = 0,
) -> torchinfo.ModelStatistics:

    # set all parameters for torchsummary

    input_size = (num_ch_in, num_samples_in)
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

    verbose = 2
    tiniernet8 = TinierNet8(64, 5)
    tiniernet8.eval()
    summarize(tiniernet8, 64, 256, verbose=verbose)


if __name__ == "__main__":
    main()
