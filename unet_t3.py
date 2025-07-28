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
    return img1,img2,img3,img4
def merge_tiles(img1,img2,img3,img4):
    
    im_t = np.concatenate((img1,img2), axis = 1)
    im_b = np.concatenate((img3,img4), axis = 1)
    res = np.concatenate((im_t,im_b), axis = 0)
    
    
    return res
    
    
def mark_box(imc, xOffset1 = 100, xOffset2 = 100, yOffset1 = 100, yOffset2 = 100, thick  = 10):
    color1 = (255,0,0)
    imshape = imc.shape
    xLim = imshape[1]
    yLim = imshape[0]
    imc = cv2.rectangle(imc, (xOffset1,yOffset1), (xLim - xOffset2,yLim - yOffset2), color1,thick) 
    return imc
def check_image():
    data_path = './test_data/denoise_unet/set1/eval1_target/'
    imgL,imgR = scr.load_images_1_dir(data_path, 'cam1', 'cam2', ext = '.jpg', colorIm = True)  
    imc = np.dstack((imgL[0],imgL[0], imgL[0]))
    #imc = mark_box(imc, 300,300,0,200)
    print(imc.shape)
    plt.imshow(imc)
    plt.show()
    t1,t2,t3,t4 = tile_image(imc)
    
    plt.imshow(t1)
    plt.show()
    plt.imshow(t2)
    plt.show()
    plt.imshow(t3)
    plt.show()
    plt.imshow(t4)
    plt.show()
    im_m = merge_tiles(t1,t2,t3,t4)
    plt.imshow(im_m)
    plt.show()
    
check_image()
class PairDataset(Dataset):
    def __init__(self, data_path_in, data_path_target, transform=None):
        imgLi,imgRi = scr.load_images_1_dir(data_path_in, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
        imgLt,imgRt = scr.load_images_1_dir(data_path_target, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
        imData= []
        imTarget = []
        for imL in imgLi:
            entIm = np.dstack((imL,imL,imL))
            im1,im2,im3,im4 = tile_image(entIm)
            imData.append(im1)
            imData.append(im2)
            imData.append(im3)
            imData.append(im4)
        for imR in imgRi:
            
            imData.append(np.dstack((imR,imR, imR)))
        for imL in imgLt:
            
            imTarget.append(np.dstack((imL,imL, imL)))
        for imR in imgRt:
            
            imTarget.append(np.dstack((imR,imR, imR)))
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
    



def run_model():
    
    # Data Loaders
    train_dataset = PairDataset('./test_data/denoise_unet/set1/train1_in/','./test_data/denoise_unet/set1/train1_target/', transform=test_transform)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = PairDataset('./test_data/denoise_unet/set1/eval1_in/','./test_data/denoise_unet/set1/eval1_target/', transform=test_transform)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)



    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Instantiate the model
    model = UNetAutoencoder()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.enc1.register_forward_hook(get_activation('enc1'))


    learning_rate = 0.001  # Initial learning rate
    n_epochs = 10

    # Move model to device
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

    # Get some test images
    dataiter = iter(testloader)
    images, targets = next(dataiter)
    images = images.to(device)

    with torch.no_grad():  # Don't calculate gradients during evaluation
        denoised_images = model(images)
        # Access the activations from the 'activation' dictionary
        enc1_activations = activation['enc1']
    '''
    # Visualize activations of the first image in the batch
    activations = enc1_activations[0]  # First image in batch
    num_channels = activations.shape[0]

    # Create a grid for visualization
    ncols = 8  # Adjust as needed
    nrows = num_channels // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            ax.imshow(activations[i].cpu().numpy(), cmap='viridis')  # Use a colormap
            ax.axis('off')

    plt.suptitle("Activations of 'enc1' for a Single Image")
    plt.show()
    '''
    denoised_images = model(images)

    # Get denoised outputs
    denoised_images = model(images)

    # Denormalize images for display
    def denormalize(images):
        images = images * 0.5 + 0.5
        return images

    denoised_images = denormalize(denoised_images.cpu())
    images = denormalize(images.cpu())

    n = 5
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original Image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Denoised Image
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(np.transpose(denoised_images[i].detach(), (1, 2, 0)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        #target image
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(targets[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()