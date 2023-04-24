# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:25:18 2023

@author: myuey
"""
import scripts as scr
import ml_scripts as mls
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # 28x28x1 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3)
        self.d1 = nn.Linear(8 * 30, 128)
        self.d2 = nn.linear(128,64)
        self.d3 = nn.Linear(64, 2)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = nn.functional.relu(x)

        x = x.flatten(start_dim = 1)

        x = self.d1(x)
        x = nn.functional.relu(x)
        x = self.d2(x)
        x = nn.functional.relu(x)
        logits = self.d3(x)
        out = nn.functional.softmax(logits, dim=1)
        return out


def run_nn():
    BATCH_SIZE = 8

    ## transformations
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    tpos, tneg, verif = mls.load_dataset()
    trainset = np.concatenate((tpos,tneg))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(verif, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)
    learning_rate = 0.001
    num_epochs = 5
    print(trainset[0].shape)
    print(iter(trainloader))
    print(torch.cuda.is_available())
    device = torch.device("cuda:0")
    
run_nn()
    