# containes custom PyTorch Dataset for loading yoga data

import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset


class YogaDataset(Dataset):
    def __init__(self, dataset, model_type):
        self.data = np.load(f'{dataset}.npy')
        self.labels = np.load(f'{dataset}_labels.npy')
        self.model_type = model_type


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).to(torch.float32)
        label = self.labels[idx]

        if self.model_type == 'fc':
            data = data.flatten()
        elif self.model_type == 'cnn':
            data = data.t()

        return data, label