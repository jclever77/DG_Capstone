# containes custom PyTorch Dataset for loading yoga data

import os
import numpy as np
from torch.utils.data import Dataset


class YogaDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = [os.path.join(folder, filename) for filename in os.listdir(folder)]


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        filepath = self.files[idx]
        arr = np.load(filepath, allow_pickle=True)

        return arr.transpose()