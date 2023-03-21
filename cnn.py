# contains code for a CNN model

import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=16,
            kernel_size=3
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            'same',
            bias=False
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    

class CNN(nn.Module):
    def __init__(
            self,
            channels
    ):
        super().__init__()

        self.convs = nn.Sequential(
            *[
            ConvBlock(channels[i - 1], channels[i]) for i in range(1, len(channels))
            ]
        )

        self.fc = nn.Sequential(
            *[
            nn.Flatten(),
            nn.Linear(23 * channels[-1], 82)
            ]
        )


    def forward(self, x):
        x = self.convs(x)

        return self.fc(x)