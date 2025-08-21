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
from skimage.metrics import structural_similarity
import cv2
from sewar.full_ref import msssim
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
    res = []
    res.append(img1)
    res.append(img2)
    res.append(img3)
    res.append(img4)
    return res
def merge_tiles(img1,img2,img3,img4):
    
    im_t = np.concatenate((img1,img2), axis = 1)
    im_b = np.concatenate((img3,img4), axis = 1)
    res = np.concatenate((im_t,im_b), axis = 0)
    
    
    return res
def multi_tile(image):
    t1 = tile_image(image)
    res = []
    for i in t1:
        
        a = tile_image(i)
        for j in a:
            res.append(j)
    return res    
def merge_multi(im_list):
    a1 = merge_tiles(im_list[0],im_list[1],im_list[2],im_list[3])     
    a2 = merge_tiles(im_list[4],im_list[5],im_list[6],im_list[7])     
    a3 = merge_tiles(im_list[8],im_list[9],im_list[10],im_list[11])     
    a4 = merge_tiles(im_list[12],im_list[13],im_list[14],im_list[15])     
    return merge_tiles(a1,a2,a3,a4)
def mark_box(imc, xOffset1 = 100, xOffset2 = 100, yOffset1 = 100, yOffset2 = 100, thick  = 10):
    color1 = (255,0,0)
    imshape = imc.shape
    xLim = imshape[1]
    yLim = imshape[0]
    imc = cv2.rectangle(imc, (xOffset1,yOffset1), (xLim - xOffset2,yLim - yOffset2), color1,thick) 
    return imc
def check_image():
    data_path = './test_data/denoise_unet/sets/eval1_target/'
    imgL,imgR = scr.load_images_1_dir(data_path, 'cam1', 'cam2', ext = '.jpg', colorIm = True)  
    imc = np.dstack((imgL[0],imgL[0], imgL[0]))
    #imc = mark_box(imc, 300,300,0,200)
    print(imc.shape)
    plt.imshow(imc)
    plt.show()
    res = multi_tile(imc)
    
    for i in res:
        plt.imshow(i)
        plt.show()
    
    r2 = merge_multi(res)
    plt.imshow(r2)
    plt.show()
class SingleImageSet(Dataset):
    def __init__(self, image1, transform=None):
        imData= []
        a = multi_tile(np.dstack((image1,image1,image1)))
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
            a = multi_tile(np.dstack((im,im,im)))
            for i in a:
                imData.append(i)
        for im in imgRi:
            a = multi_tile(np.dstack((im,im,im)))
            for i in a:
                imData.append(i)
        for im in imgLt:
            a = multi_tile(np.dstack((im,im,im)))
            for i in a:
                imTarget.append(i)
        for im in imgRt:
            a = multi_tile(np.dstack((im,im,im)))
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
    

device = torch.device("cuda:0")
#device = torch.device("cpu")
def run_model_train():
    
    train_dataset = PairDatasetDir('./test_data/denoise_unet/sets/train1_in_625f/','./test_data/denoise_unet/sets/train1_target/', transform=test_transform)

    n_epochs = 20
    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = UNetAutoencoder()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.enc1.register_forward_hook(get_activation('enc1'))


    learning_rate = 0.001  # Initial learning rate
    

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
    #save model weights
    save_path = './test_data/denoise_unet/unet_t3_weights_20ep_set1.pth'
    torch.save(model.state_dict(), save_path)
    
    
def denormalize(images):
    images = images * 0.5 + 0.5
    return images
def run_model_test():
    test_dataset = PairDatasetDir('./test_data/denoise_unet/sets/eval1_in/','./test_data/denoise_unet/sets/eval1_target/', transform=test_transform)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = UNetAutoencoder()
    model.load_state_dict(torch.load('./test_data/denoise_unet/unet_t3_weights_20ep_set1.pth'))
    model.to(device)
    dataiter = iter(testloader)
    images, targets = next(dataiter)
    images = images.to(device)

    denoised_images = model(images)

    # Get denoised outputs
    denoised_images = model(images)



    denoised_images = denormalize(denoised_images.cpu())
    images = denormalize(images.cpu())

    n = 10
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
        plt.imshow(np.transpose(denormalize(targets[i]), (1,2,0)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    im1 = []
    f1 = []
    t1 = []
    for i in range(16):
        im1.append(np.transpose(images[i], (1, 2, 0)))
        f1.append(np.transpose(denoised_images[i].detach(), (1, 2, 0)))
        t1.append(np.transpose(denormalize(targets[i]), (1, 2, 0)))
    im2 = merge_multi(im1)
    f2 = merge_multi(f1)
    t2 = merge_multi(t1)
    scr.display_stereo(im2,t2, 'Input', 'Target')
    scr.display_stereo(im2,f2, 'Input', 'Output')
    scr.display_stereo(f2,t2, 'Output', 'Target')


def process_list(image_list, model):
    
    model.to(device)
    pro_list = []
    for im in tqdm(image_list):
        #load images
        imageset = SingleImageSet(im,transform=test_transform)
        images_dataloader = DataLoader(imageset, batch_size=16, shuffle=False)
        dataiter = iter(images_dataloader)
        images_in = next(dataiter)
        images = images_in.to(device)
        denoised_images = model(images)
        
        
        
        denoised_images = denormalize(denoised_images.cpu())
        res = []
        for i in range(len(denoised_images)):
            res.append(np.transpose(denoised_images[i].detach(), (1, 2, 0)))
        proc = cv2.normalize(merge_multi(res), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        pro_list.append(proc)
    return pro_list    
def run_model_process(image, model):
    model.to(device)

    imageset = SingleImageSet(image,transform=test_transform)
    images_dataloader = DataLoader(imageset, batch_size=16, shuffle=False)
    dataiter = iter(images_dataloader)
    images_in = next(dataiter)
    images = images_in.to(device)
    denoised_images = model(images)
    
    


    denoised_images = denormalize(denoised_images.cpu())
    res = []
    for i in range(len(denoised_images)):
        res.append(np.transpose(denoised_images[i].detach(), (1, 2, 0)))
    proc = cv2.normalize(merge_multi(res), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    return proc
def ssim_compare(im1,im2):
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(im1, im2, full=True)
    diff = (diff * 255).astype("uint8")
    return score,diff
def verify_images(folder,ext = ''):
    res = []
    
    for file in os.listdir(folder):
        if file.endswith(ext):
            res.append(file)
    res.sort()
    for i in range(len(res)):
        print(res[i])
        img = cv2.imread(folder + res[i])
    print('Image check complete for: ' + folder)
     
    
def t1():
    #load image
    input_folder = "./test_data/denoise_unet/sets/eval1_in/"
    target_folder = "./test_data/denoise_unet/sets/eval1_target/"
    input_imgs = scr.load_all_imgs_1_dir(input_folder)
    target_imgs = scr.load_all_imgs_1_dir(target_folder)
    img_ind = 3
    img = input_imgs[img_ind]
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    model = UNetAutoencoder()
    model.load_state_dict(torch.load('./test_data/denoise_unet/t3_wts_80ep_set4_allbl.pth', weights_only = True))
    #model.load_state_dict(torch.load('./test_data/denoise_unet/t3_wts_20ep_set1_625f.pth', weights_only = True))
    #pass image through nn
    img_chk = run_model_process(img2, model)
    
    #load target
    targ = target_imgs[img_ind]
    
    scr.display_stereo(img,targ, 'Input', 'Target')
    scr.display_stereo(img,img_chk, 'Input', 'Output')
    scr.display_stereo(img_chk,targ, 'Output', 'Target')
    #run SSIM compare on image and target
    score, diff = ssim_compare(img_chk,targ)
    #ms_score = msssim(img_chk,targ)
    scr.dptle(diff, 'Diff Map - SSIM: ' + str(round(score,5)), cmap = 'gray')
    scr.display_4_comp(img,img_chk,targ,diff,"Input","Output","Target",'Diff Map - SSIM: ' + str(round(score,5)))
    
    img_chk2 = scr.boost_zone(img, 50, 1, 1, 1, 1)
    
    
    score2, diff2 = ssim_compare(img_chk2,targ)
    
    scr.dptle(diff2, 'Diff Map - SSIM: ' + str(round(score2,5)), cmap = 'gray')
    scr.display_4_comp(img,img_chk2,targ,diff2,"Input","Output","Target",'Diff Map - SSIM: ' + str(round(score2,5)))

  
def t2():
    model = UNetAutoencoder()
    model.load_state_dict(torch.load('./test_data/denoise_unet/unet_t3_weights_30ep_set1.pth'))
    #load images
    data_path_in = './test_data/denoise_unet/trec_inputs1/'
    imgL,imgR = scr.load_images_1_dir(data_path_in, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
    #pass through nn
    imgL = process_list(imgL, model)
    imgR = process_list(imgR, model)
    #filename templates
    left_nm = "cam1_proc_pattern_"
    right_nm = "cam2_proc_pattern_"
    #save images
    output_path = './test_data/denoise_unet/trec_outputs1/'
    for i in range(len(imgL)):
        cv2.imwrite(output_path + left_nm + str(i)+'.jpg', imgL[i])
    for j in range(len(imgR)):
        cv2.imwrite(output_path + right_nm + str(j)+'.jpg', imgR[j])

def t2_b():
    #load images
    data_path_in = './test_data/denoise_unet/trec_inputs1/'
    imgL,imgR = scr.load_images_1_dir(data_path_in, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
    imgL = scr.boost_list(imgL, 40, 1, 1, 1, 1)
    imgR = scr.boost_list(imgR, 40, 1, 1, 1, 1)
    #filename templates
    left_nm = "cam1_proc_pattern_"
    right_nm = "cam2_proc_pattern_"
    #save images
    output_path = './test_data/denoise_unet/trec_outputs1b/'
    for i in range(len(imgL)):
        cv2.imwrite(output_path + left_nm + str(i)+'.jpg', imgL[i])
    for j in range(len(imgR)):
        cv2.imwrite(output_path + right_nm + str(j)+'.jpg', imgR[j])

def calcF():
    #load images
    data_path_in = './test_data/denoise_unet/trec_outputs1/'
    imgL,imgR = scr.load_images_1_dir(data_path_in, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
    f = scr.find_f_mat(imgL[0], imgR[0])
    np.savetxt("./test_data/denoise_unet/matrices/" + "fund.txt", f, header = "3\n3")
    
        
def t3():
    model = UNetAutoencoder()
    model.load_state_dict(torch.load('./test_data/denoise_unet/t3_wts_30ep_set1_625f.pth'))
    #load images
    data_path_in = './test_data/denoise_unet/trec_inputs1/'
    data_path_ref = './test_data/denoise_unet/trec_reference1/'
    imgL,imgR = scr.load_images_1_dir(data_path_in, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
    imgLref,imgRref = scr.load_images_1_dir(data_path_in, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
    #pass through nn
    imgLP = process_list(imgL, model)
    imgRP = process_list(imgR, model)
    comp_score_list_L = []
    diff_list_L = []
    for i in range(len(imgLP)):
        s,d = ssim_compare(imgLP[i], imgLref[i])
        comp_score_list_L.append(s)
        diff_list_L.append(d)
    comp_score_list_R = []
    diff_list_R = []
    for i in range(len(imgRP)):
        s,d = ssim_compare(imgRP[i], imgRref[i])
        comp_score_list_R.append(s)
        diff_list_R.append(d)
    scrL_arr = np.asarray(comp_score_list_L)
    scrR_arr = np.asarray(comp_score_list_R)
    
    print(np.average(scrL_arr))
    print(np.average(scrR_arr))
    print((np.average(scrL_arr)+np.average(scrR_arr))/2)

    
t3()