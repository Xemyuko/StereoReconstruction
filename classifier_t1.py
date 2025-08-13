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
    n = 25
    res = np.zeros((n,imshape[0],imshape[1], 3), dtype = img.dtype)
    for i in range(dist,imshape[0] - dist):
        for j in range(dist,imshape[1] - dist):
            #assign central pixel
            res[0,i,j,:] = img[i,j,:]
            #assign cardinal directions, first layer, NSEW
            res[1,i,j,:] = img[i-1,j,:]
            res[2,i,j,:] = img[i+1,j,:]
            res[3,i,j,:] = img[i,j-1,:]
            res[4,i,j,:] = img[i,j+1,:]
            #first layer diagonals
            res[5,i,j,:] = img[i-1,j-1,:]
            res[6,i,j,:] = img[i+1,j+1,:]
            res[7,i,j,:] = img[i+1,j-1,:]
            res[8,i,j,:] = img[i-1,j+1,:]
            #second layer cardinals
            res[9,i,j,:] = img[i-2,j,:]
            res[10,i,j,:] = img[i+2,j,:]
            res[11,i,j,:] = img[i,j-2,:]
            res[12,i,j,:] = img[i,j+2,:]
            #second layer diagonals
            res[13,i,j,:] = img[i-2,j-2,:]
            res[14,i,j,:] = img[i+2,j-2,:]
            res[15,i,j,:] = img[i-2,j+2,:]
            res[16,i,j,:] = img[i+2,j+2,:]
            #second layer fills
            res[17,i,j,:] = img[i-2,j-1,:]
            res[18,i,j,:] = img[i+2,j-1,:]
            res[19,i,j,:] = img[i-1,j-2,:]
            res[20,i,j,:] = img[i-1,j+2,:]
            res[21,i,j,:] = img[i-2,j+1,:]
            res[22,i,j,:] = img[i+2,j+1,:]
            res[23,i,j,:] = img[i+1,j-2,:]
            res[24,i,j,:] = img[i+1,j+2,:]
            '''
            #third layer cardinals
            res[25,i,j] = img[i-3,j]
            res[26,i,j] = img[i+3,j]
            res[27,i,j] = img[i,j-3]
            res[28,i,j] = img[i,j+3]
            #third layer diagonals
            res[29,i,j] = img[i-3,j-3]
            res[30,i,j] = img[i+3,j-3]
            res[31,i,j] = img[i-3,j+3]
            res[32,i,j] = img[i+3,j+3]
            #third layer fills
            res[33,i,j] = img[i-3,j-2]
            res[34,i,j] = img[i+3,j-2]
            res[35,i,j] = img[i-2,j-3]
            res[36,i,j] = img[i-2,j+3]
            res[37,i,j] = img[i-3,j+2]
            res[38,i,j] = img[i+3,j+2]
            res[39,i,j] = img[i+2,j-3]
            res[40,i,j] = img[i+2,j+3]
            
            res[41,i,j] = img[i-3,j-1]
            res[42,i,j] = img[i+3,j-1]
            res[43,i,j] = img[i-1,j-3]
            res[44,i,j] = img[i-1,j+3]
            res[45,i,j] = img[i-3,j+1]
            res[46,i,j] = img[i+3,j+1]
            res[47,i,j] = img[i+1,j-3]
            res[48,i,j] = img[i+1,j+3]
            '''
            
    return res
            
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