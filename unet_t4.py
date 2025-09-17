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
class PairDatasetDirResize(Dataset):
    def __init__(self, data_path_in, data_path_target, transform=None):
        imgi = scr.load_all_imgs_1_dir(data_path_in, ext = '.jpg', convert_gray=False)
        imgt = scr.load_all_imgs_1_dir(data_path_target, ext = '.jpg', convert_gray=False)
        imData= []
        imTarget = []
        reshape_val = 704
        for i in imgi:
            imData.append(cv2.resize(i, dsize=(reshape_val,reshape_val), interpolation=cv2.INTER_CUBIC))
        for t in imgt:
            imTarget.append(cv2.resize(t, dsize=(reshape_val,reshape_val), interpolation=cv2.INTER_CUBIC))
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
    
class UNet1(nn.Module):
    def __init__(self):
        super(UNet1, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 512)
        self.enc6 = self.conv_block(512, 1024)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder

        self.dec1 = self.upconv_block(1024, 512)
        self.dec2 = self.upconv_block(1024, 256)
        self.dec3 = self.upconv_block(512,128)
        self.dec4 = self.upconv_block(256,64)
        self.dec5 = self.upconv_block(128, 32)
        self.dec6 = nn.Conv2d(64, 3, kernel_size=1) # Final 1x1 convolution

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

        e5 = self.enc5(self.pool(e4))

        e6 = self.enc6(self.pool(e5))

        # Decoder
        
        d1 = self.dec1(e6)

        d1 = torch.cat([e5, d1], dim=1)
        

        d2 = self.dec2(d1)

        d2 = torch.cat([e4, d2], dim=1)

        
        d3 = self.dec3(d2)

        d3 = torch.cat([e3, d3], dim=1)

        d4 = self.dec4(d3)

        d4 = torch.cat([e2, d4], dim=1)

        d5 = self.dec5(d4)

        d5 = torch.cat([e1, d5], dim=1)

        d6 = self.dec6(d5)

        return torch.tanh(d6) # Tanh activation for output
    
    
device = torch.device("cuda:0")
def run_model_train(train, ref, save_path):
    
    train_dataset = PairDatasetDir(train,ref, transform=test_transform)

    n_epochs = 20
    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = UNet1()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.enc1.register_forward_hook(get_activation('enc1'))


    learning_rate = 0.001  # Initial learning rate
    

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Learning rate scheduler (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, threshold=1e-7, threshold_mode='abs')


    # Function to train the model
    def train_model(model, trainloader, device, n_epochs, optimizer, criterion, scheduler):
        print("Training with Adam optimizer...")

        losses = []

        torch.cuda.empty_cache()
        gc.collect()
    
        for epoch in tqdm(range(n_epochs)):
            running_loss = 0.0
            for i, (noisy_images, clean_images) in enumerate(trainloader):
                noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

                outputs = model(noisy_images)
                loss = criterion(outputs, clean_images)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()

                del noisy_images, clean_images, outputs, loss
                torch.cuda.empty_cache()

            epoch_loss = running_loss / len(trainloader)
            losses.append(epoch_loss)
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

            scheduler.step(epoch_loss)
            #print(f"Scheduler state: {scheduler.state_dict()}")
        
            torch.cuda.empty_cache()
            gc.collect()

        print('Finished Training')
        return losses

    model = model.to(device)

    # Train the model
    losses = train_model(model, trainloader, device, n_epochs, optimizer, criterion, scheduler)
    #save model weights
    
    torch.save(model.state_dict(), save_path)
    
def denormalize(images):
    images = images * 0.5 + 0.5
    return images

run_model_train('./test_data/denoise_unet/sets/train1_in_325f/', 
                './test_data/denoise_unet/sets/train1_target/', './test_data/denoise_unet/t4_wts_20ep_set1.pth')