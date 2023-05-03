# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:23:07 2023

@author: myuey
"""
import numpy as np
import ml_scripts as mls
from torch.utils.data import Dataset
class VerifDataset(Dataset):
    def __init__(self, data_dir, verif_name):
        pass
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry = self.data[idx]
        label = self.labels[idx]
        return entry, label