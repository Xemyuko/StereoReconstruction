# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:29:48 2025

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
from torchview import draw_graph
import cv2
torch.cuda.empty_cache()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def tile_image(image):
    #cut image into 4 parts
    imshape = image.shape
    xhalf = int(imshape[1]/2)
    yhalf = int(imshape[0]/2)
    img1 = image[:yhalf,:xhalf,:]
    img2 = image[:yhalf,xhalf:imshape[1],:]
    img3 = image[yhalf:imshape[0],:xhalf,:]
    img4 = image[yhalf:imshape[0],xhalf:imshape[1],:]
    print(img1.shape)
    print(img2.shape)
    print(img3.shape)
    print(img4.shape)
    return img1,img2,img3,img4
def merge_tiles(img1,img2,img3,img4):
    pass    
def mark_box(imc, xOffset1 = 100, xOffset2 = 100, yOffset1 = 100, yOffset2 = 100, thick  = 10):
    color1 = (255,0,0)
    imshape = imc.shape
    xLim = imshape[1]
    yLim = imshape[0]
    imc = cv2.rectangle(imc, (xOffset1,yOffset1), (xLim - xOffset2,yLim - yOffset2), color1,thick) 
    return imc
def check_image():
    data_path = './test_data/eval1_data_in/'
    #data_path = './test_data/testset1/bulb/'
    imgL,imgR = scr.load_images_1_dir(data_path, 'cam1', 'cam2', ext = '.jpg', colorIm = True)  
    imc = np.dstack((imgL[0],imgL[0], imgL[0]))
    imc = mark_box(imc)
    print(imc.shape)
    t1,t2,t3,t4 = tile_image(imc)
    t11,t12,t13,t14 = tile_image(t1)
check_image()
class EvalDataset(Dataset):
    def __init__(self, folder0, transform=None):
        
        imgL,imgR = scr.load_images_1_dir(folder0, 'cam1', 'cam2', ext = '.jpg', colorIm = False) 
        imTest = []
        for imL in imgL:
            imTest.append(np.dstack((imL,imL, imL)))
        for imR in imgR:
            imTest.append(imR)
        self.test_img_list = imTest
        self.transform = transform
        

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        image = self.test_img_list[idx]

        if self.transform:
            image = self.transform(image)

        return image



class TrainDataset(Dataset):
    def __init__(self, data_path_in, data_path_target, transform=None):
        imgLi,imgRi = scr.load_images_1_dir(data_path_in, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
        imgLt,imgRt = scr.load_images_1_dir(data_path_target, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
        imData= []
        imTarget = []
        for imL in imgLi:
            imData.append(imL)
        for imR in imgRi:
            imData.append(imR)
        for imL in imgLt:
            imTarget.append(imL)
        for imR in imgRt:
            imTarget.append(imR)
        self.target_list = imTarget
        self.train_list = imData
        
        self.transform = transform


    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, idx):
        imgTar = self.target_list[idx]
        imgData = self.train_list_list[idx]
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
    
    
