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

class ProcessImageSet(Dataset):
    def __init__(self, img_list, transform=None):
        reshape_val = 704
        imData= []
        
        for i in img_list:


            img_in = cv2.resize(i, dsize=(reshape_val,reshape_val), interpolation=cv2.INTER_CUBIC)

        imData.append(img_in)
            
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
    def __init__(self, data_path_in, data_path_target, reshape_val = 704, transform=None):
        imgi = scr.load_imgs(data_path_in, ext = '.jpg', convert_gray=False)
        imgt = scr.load_imgs(data_path_target, ext = '.jpg', convert_gray=False)
        imData= []
        imTarget = []
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
    
class UNetT4(nn.Module):
    def __init__(self):
        super(UNetT4, self).__init__()

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

def run_model_train(train, ref, save_path, n_epochs = 20, n_save = 20):
    
    train_dataset = PairDatasetDir(train,ref, reshape_val = 400,transform=test_transform)

    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = UNetT4()

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
'''
run_model_train('./test_data/denoise_unet/sets/statue-fb-t2-train1/', 
                './test_data/denoise_unet/sets/statue-fb-target1/', 
                './test_data/denoise_unet/unet_t4_40ep_statue_fb_t2_352.pth', n_epochs = 40)
'''



def run_model_process(image, model):
    model.to(device)
    img_in = []
    img_in.append(image)
    imageset = ProcessImageSet(img_in,transform=test_transform)
    images_dataloader = DataLoader(imageset, batch_size=16, shuffle=False)
    dataiter = iter(images_dataloader)
    images_in = next(dataiter)
    images = images_in.to(device)
    denoised_images = model(images)
    denoised_images = denormalize(denoised_images.cpu())
    res=np.asarray(np.transpose(denoised_images[0].detach(), (1, 2, 0)))
    proc = cv2.normalize(res, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    fin_im = cv2.resize(proc, dsize=(2848,2848), interpolation=cv2.INTER_CUBIC)
    return fin_im

def t1():
    #process 1 image using resized images
    #load image
    
    input_folder = "./test_data/denoise_unet/sets/eval_in_t1/"
    #input_folder = "./test_data/denoise_unet/sets/eval_in_t2/"
    #input_folder = "./test_data/denoise_unet/sets/eval_in_t3/"
    target_folder = "./test_data/denoise_unet/sets/eval_target/"
    input_imgs = scr.load_imgs(input_folder)
    target_imgs = scr.load_imgs(target_folder)
    img_ind = 4
    img = input_imgs[img_ind]
    print(img.shape)
    model = UNetT4()
    model.load_state_dict(torch.load('./test_data/denoise_unet/unet_t4_150ep_bs_fb_t1.pth', weights_only = True))
    #model.load_state_dict(torch.load('./test_data/denoise_unet/unet_t4_80ep_bs_t2.pth', weights_only = True))
    #model.load_state_dict(torch.load('./test_data/denoise_unet/unet_t4_40ep_bs_t3.pth', weights_only = True))
    #pass image through nn
    img_chk = run_model_process(img, model)
    
    #load target
    targ = target_imgs[img_ind]
    
    scr.display_stereo(img,targ, 'Input', 'Target')
    scr.display_stereo(img,img_chk, 'Input', 'Output')
    scr.display_stereo(img_chk,targ, 'Output', 'Target')
    #run SSIM compare on image and target
    score, diff = scr.ssim_compare(img_chk,targ)
    #ms_score = msssim(img_chk,targ)
    scr.dptle(diff, 'Diff Map - SSIM: ' + str(round(score,5)), cmap = 'gray')
    scr.display_4_comp(img,img_chk,diff,targ,"Input","Output",'Diff Map - SSIM: ' + str(round(score,5)),"Target")
  
    img_chk2 = scr.boost_zone(img,1000, 1, 1, 1, 1)
    
    
    score2, diff2 = scr.ssim_compare(img_chk2,targ)
    
    scr.dptle(diff2, 'Diff Map - SSIM: ' + str(round(score2,5)), cmap = 'gray')
    scr.display_4_comp(img,img_chk2,diff2,targ,"Input","Output",'Diff Map - SSIM: '+ str(round(score2,5)),"Target" )


def t2():
    #process folder of images and save them for reconstruction
    model = UNetT4()
    model.load_state_dict(torch.load('./test_data/denoise_unet/unet_t4_150ep_bs_fb_t1.pth', weights_only = True))
    #load images
    data_path_in = 'C:/Users/Admin/Documents/251017_blockball/block16500/'
    imgL,imgR = scr.load_imagesLR(data_path_in, 'cam1', 'cam2', ext = '.jpg')
    imgLP = []
    imgRP = []
    #pass through nn
    for a in tqdm(imgL):
        imgLP.append(run_model_process(a,model))
    for b in tqdm(imgR):
        imgRP.append(run_model_process(b,model))
    #filename templates
    left_nm = "cam1_proc_pattern_"
    right_nm = "cam2_proc_pattern_"
    #save images
    output_path = 'C:/Users/Admin/Documents/251017_blockball/blockproc/'
    for i in range(len(imgLP)):
        cv2.imwrite(output_path + left_nm + str(i)+'.jpg', imgLP[i])
    for j in range(len(imgRP)):
        cv2.imwrite(output_path + right_nm + str(j)+'.jpg', imgRP[j])

