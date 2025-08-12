# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:08:21 2025

@author: Admin
"""

import os
import gc
import ncc_core as ncc
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import scripts as scr
import random
from tqdm import tqdm
import time

import cv2
torch.cuda.empty_cache()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def spat_extract(img, dist):
    imshape = img.shape
    n = 0
    res = np.zeros((n,imshape[0],imshape[1]), dtype = img.dtype)
    for i in range(dist,imshape[0] - dist):
        for j in range(dist,imshape[1] - dist):
            pass
def ref_data_gen(imsL,imsR, kL, kR, R, t, F, offset = 10):
    ptsL, ptsR = ncc.cor_pts_pix(imsL, imsR, kL, kR, R, t, F, offset)
    
class PairDataset(Dataset):
    def __init__(self, data_in, data_target, transform=None):
        self.target_list = data_target
        self.train_list = data_in
        
        self.transform = transform
    def __len__(self):
        return len(self.target_list)
    def __getitem__(self, idx):
        imgTar = self.target_list[idx]
        imgData = self.train_list[idx]
        if self.transform:
            imgTar = self.transform(imgTar)
            imgData = self.transform(imgData)        
        return imgData, imgTar
    
classes = (0,1)

class CNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10,2)

    def forward(self, x):
        x = self.pool(TF.relu(self.conv1(x)))
        x = self.pool(TF.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = TF.relu(self.fc1(x))
        x = TF.relu(self.fc2(x))
        x = TF.relu(self.fc3(x))
        x = self.fc4(x)
        return x