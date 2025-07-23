# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:10:13 2025

@author: myuey
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

from torchview import draw_graph
torch.cuda.empty_cache()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


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


class TestImageDataset(Dataset):
    def __init__(self, folder0, transform=None):
        
        imgLc1,imgRc1 = scr.load_images_1_dir(folder0, 'cam1', 'cam2', ext = '.jpg', colorIm = True) 
        imTest = []
        for imL in imgLc1:
            imTest.append(imL[18:1070,2:1454,:])
        self.test_img_list = imTest
        self.transform = transform
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.test_img_list[idx]

        if self.transform:
            image = self.transform(image)

        return image

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        imgLc,imgRc = scr.load_images_1_dir(data_path, 'cam1', 'cam2', ext = '.jpg', colorIm = True)  
        imData= []
        imTarget = []
        for imL in imgLc:
            imData.append(sp_noise(imL,0.3, True)[18:1070,2:1454,:])
            imTarget.append(imL[18:1070,2:1454,:])
        self.clean_list = imTarget
        self.noisy_list = imData
        self.transform = transform


    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, idx):
        clean_image = self.clean_list[idx]
        noisy_image = self.noisy_list[idx]
        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)
        
        return noisy_image, clean_image

    


# Data Loaders
train_dataset = ImageDataset('./test_data/250221_Cudatest/pos7/', transform=test_transform)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Create a DataLoader for your ImageDataset
test_dataset = TestImageDataset('./test_data/250221_Cudatest/pos8/', transform=test_transform)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


batch_size = 32

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Instantiate the model
model = UNetAutoencoder()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.enc1.register_forward_hook(get_activation('enc1'))


learning_rate = 0.001  # Initial learning rate
n_epochs = 50 # Train for more epochs

# Move model to device
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Learning rate scheduler (ReduceLROnPlateau)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True, threshold=1e-7, threshold_mode='abs')


# Function to train the model
def train_model(model, trainloader, device, n_epochs, optimizer, criterion, scheduler):
    print("Training with Adam optimizer...")

    losses = []

    torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(n_epochs):
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
        print(f"Scheduler state: {scheduler.state_dict()}")

        torch.cuda.empty_cache()
        gc.collect()

    print('Finished Training')
    return losses

model = model.to(device)

# Train the model
losses = train_model(model, trainloader, device, n_epochs, optimizer, criterion, scheduler)

# Get some test images
dataiter = iter(testloader)
images = next(dataiter)
images = images.to(device)

with torch.no_grad():  # Don't calculate gradients during evaluation
    denoised_images = model(images)
    # Access the activations from the 'activation' dictionary
    enc1_activations = activation['enc1']

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
plt.suptitle("Denoised Images with U-Net")
for i in range(n):
    # Original Image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.transpose(images[i], (1, 2, 0)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Denoised Image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.transpose(denoised_images[i].detach(), (1, 2, 0)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()