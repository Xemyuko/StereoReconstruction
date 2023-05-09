# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:25:18 2023

@author: myuey
"""

import torch 
import torch.nn as nn
import trainpairdataset as pada
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class modelA(nn.Module):
    def __init__(self):
        super(modelA, self).__init__()

        # 28x28x1 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.d1 = nn.Linear(34*28, 64)
        self.d2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        
        x = x.flatten(start_dim = 1)
        x = self.d1(x)
        x = nn.functional.relu(x)
        logits = self.d2(x)
        out = nn.functional.softmax(logits, dim=1)
        return out

BATCH_SIZE = 1

## transformations
transform = transforms.Compose([transforms.ToTensor()])
trainset = pada.PairDataset("","trainPosA.npy","trainNegA.npy")
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
for data, labels in trainloader:
    print("Batch dimensions:", data.shape)
    print("Label dimensions:", labels.shape)
    break    
learning_rate = 0.001
num_epochs = 5

device = torch.device("cuda:0")
model = modelA()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model = model.train()

    ## training step
    for i, (data, labels) in enumerate(trainloader):
        
        data = data.to(device)
        labels = labels.to(device)

        ## forward + backprop + loss
        logits = model(data)
        print(logits.dtype)
        print(labels.dtype)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()

        ## update model params
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(logits, labels, BATCH_SIZE)
    
    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc/i)) 