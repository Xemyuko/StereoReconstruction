# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:02:07 2023

@author: myuey
"""
import numpy as np
from torch.utils.data import Dataset
class PairDataset(Dataset):
    def load_data(self,data_name, label_name):
        data = np.load(data_name, allow_pickle = True)
        labels = np.load(label_name, allow_pickle = True)
        return data,labels
    def __init__(self, data_dir, data_name, lbl_name):
        self.data_dir = data_dir
        self.data, self.labels = self.load_data(data_dir + data_name, data_dir + lbl_name)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry = self.data[idx]
        label = self.labels[idx]
        return entry, label