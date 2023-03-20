# containes custom PyTorch Dataset for loading yoga data

import os
import numpy as np
from torch.utils.data import Dataset


class YogaDataset(Dataset):
    def __init__(self, folder):
        data = []
        
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            pose_arr = np.load(filepath)
            for body in pose_arr:
                


    def __len__(self):
        pass


    def __getitem__(self, idx):
        pass