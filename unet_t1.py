# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:03:46 2025

@author: myuey
"""

import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
import scripts as scr
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn.functional import relu
import copy
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models

def sp_noise(image,prob, mode = False):
    '''
    
    prob: Probability of the noise
    mode: salt&pepper or only pepper - simulates darkening from low exposure time
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                if mode:
                    output[i][j] = 1 
                else:
                    output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output




def test_noi_gen():
    #load image
    #load image pair
    folder1 = './test_data/250221_Cudatest/pos4/'
    #folder1 = './test_data/testset1/bulb-multi/b1/'
    imgLc,imgRc = scr.load_images_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg', colorIm = True)
    imgL,imgR = scr.load_images_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg', colorIm = False)

    n1 = sp_noise(imgLc[0],0.1, True)
    scr.display_stereo(imgLc[0], n1)
    n2 = sp_noise(imgLc[0],0.2, True)
    scr.display_stereo(imgLc[0], n2)
    n3 = sp_noise(imgLc[0],0.3, True)
    scr.display_stereo(imgLc[0][18:1070,2:1454,:], n3[18:1070,2:1454,:])
    print(n1.shape)

test_noi_gen()
def noi_data_gen(count):
    # input: 1452x1052x3
    folder1 = './test_data/250221_Cudatest/pos4/'
    imgLc,imgRc = scr.load_images_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg', colorIm = True)    

    imData = []
    imTarget = []
    for imL in tqdm(imgLc):
        imData.append(sp_noise(imL,0.1, True)[18:1070,2:1454,:])
        imTarget.append(imL[18:1070,2:1454,:])
        imData.append(sp_noise(imL,0.2, True)[18:1070,2:1454,:])
        imTarget.append(imL[18:1070,2:1454,:])
        imData.append(sp_noise(imL,0.3, True)[18:1070,2:1454,:])
        imTarget.append(imL[18:1070,2:1454,:])
        
    
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 1052x1452x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 1050x1450x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 1048x1448x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 524x724x64

        # input: 724x524x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 522x722x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 520x720x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 260x360x128

        # input: 360x260x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 258x358x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 256x356x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 128x178x256

        # input: 178x128x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 126x176x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 124x174x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 62x87x512

        # input: 87x62x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 60x85x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 58x83x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
        
        
    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out

class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = noi_data_gen(count = count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]
def get_data_loaders():
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    train_set = SimDataset(100, transform = trans)
    val_set = SimDataset(20, transform = trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders
def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
    
def train_model(model, optimizer, scheduler, num_epochs=25):
    dataloaders = get_data_loaders()
    device = torch.device("cuda:0")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run(UNet):
    num_class = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=60)

    model.eval()  # Set model to the evaluation mode

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    ])
    
    
    
    '''
    # # Create another simulation dataset for test
    test_dataset = SimDataset(3, transform = trans)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

    # Get the first batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    pred_rgb = [masks_to_colorimg(x) for x in pred]

    plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

    '''
