# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:56:57 2025

@author: Admin
"""

import os
import gc

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
from skimage.metrics import structural_similarity
import cv2
from sewar.full_ref import msssim
torch.cuda.empty_cache()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class SingleImageSet(Dataset):
    def __init__(self, image1, transform=None):
        imData= []
        a = scr.multi_tile(np.dstack((image1,image1,image1)))
        for i in a:
            imData.append(i)
            
        self.img_list = imData
        
        self.transform = transform


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imgData = self.img_list[idx]
        if self.transform:
            imgData = self.transform(imgData)
        
        return imgData
class PairDatasetDir(Dataset):
    def __init__(self, data_path_in, data_path_target, transform=None):
        imgLi,imgRi = scr.load_images_1_dir(data_path_in, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
        imgLt,imgRt = scr.load_images_1_dir(data_path_target, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
        imData= []
        imTarget = []
        for im in imgLi:
            a = scr.multi_tile(np.dstack((im,im,im)))
            for i in a:
                imData.append(i)
        for im in imgRi:
            a = scr.multi_tile(np.dstack((im,im,im)))
            for i in a:
                imData.append(i)
        for im in imgLt:
            a = scr.multi_tile(np.dstack((im,im,im)))
            for i in a:
                imTarget.append(i)
        for im in imgRt:
            a = scr.multi_tile(np.dstack((im,im,im)))
            for i in a:
                imTarget.append(i)
        self.target_list = imTarget
        self.train_list = imData
        
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
    
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.dec1 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(256, 64)
        self.dec3 = self.upconv_block(128, 32)
        self.dec4 = nn.Conv2d(64, 3, kernel_size=1) # Final 1x1 convolution

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d1 = self.dec1(e4)
        d1 = torch.cat([e3, d1], dim=1)  # Skip connection
        d2 = self.dec2(d1)
        d2 = torch.cat([e2, d2], dim=1)  # Skip connection
        d3 = self.dec3(d2)
        d3 = torch.cat([e1, d3], dim=1)  # Skip connection
        d4 = self.dec4(d3)  # No activation in the final layer

        return torch.tanh(d4) # Tanh activation for output
    