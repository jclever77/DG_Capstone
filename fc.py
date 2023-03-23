# containes code for a fully-connected network

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            drop_prob=0.5
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_dim,
            out_dim,
            bias=False
        )

        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(drop_prob, inplace=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        return self.relu(self.dropout(self.bn(self.linear(x))))
    

class FC(nn.Module):
    def __init__(
            self,
            dims
    ):
        super().__init__()

        self.hidden_layers = nn.Sequential(
            *[
            MLP(dims[i - 1], dims[i]) for i in range(1, len(dims))
            ]
        )

        self.final = nn.Linear(dims[-1], 82)


    def forward(self, x):
        out = self.hidden_layers(x)

        return self.final(out)