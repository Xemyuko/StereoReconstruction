# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:02:07 2023

@author: myuey
"""
import numpy as np
import ml_scripts as mls
from torch.utils.data import Dataset
class PairDataset(Dataset):
    
    def __init__(self, data_dir, train_pos_name, train_neg_name):
        self.data_dir = data_dir
        tp,tn = mls.load_dataset(data_dir + train_pos_name, data_dir + train_neg_name)
        self.data = np.concatenate((tp,tn))
        self.labels = np.concatenate((np.ones((tp.shape[2],)),np.zeros((tn.shape[2],))))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry = self.data[idx]
        label = self.labels[idx]
        return entry, label