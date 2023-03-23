# containes custom PyTorch Dataset for loading yoga data

import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset


class YogaDataset(Dataset):
    def __init__(self, folder, model_type):
        self.folder = folder
        self.filenames = os.listdir(folder)
        self.pose2idx = {pose: idx for pose, idx in zip(os.listdir('yoga'), range(len(os.listdir('yoga'))))}
        self.model_type = model_type


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.folder, filename)
        arr = torch.from_numpy(np.load(filepath, allow_pickle=True)).to(torch.float32)

        if self.model_type == 'fc':
            arr = arr.flatten()
        elif self.model_type == 'cnn':
            arr = arr.t()

        posename = re.split('[0-9]', filename, 1)[0]
        # label = torch.zeros(82, dtype=torch.long)
        # label[self.pose2idx[posename]] = 1

        return arr, self.pose2idx[posename]