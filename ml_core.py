# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:25:18 2023

@author: myuey
"""

import torch 
import torch.nn as nn
import pairdataset as pada
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
class modelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(30, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 120)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(120, 20)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
class modelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(30, 90)
        self.relu = nn.ReLU()
        self.output = nn.Linear(90, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x    

BATCH_SIZE = 10

## transformations
transform = transforms.Compose([transforms.ToTensor()])
train_set = pada.PairDataset("","train.npy","train_labels.npy")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
for data, labels in train_loader:
    print("Batch dimensions:", data.shape)
    print("Label dimensions:", labels.shape)
    break    
verif_set = pada.PairDataset("","verif.npy","verif_labels.npy")
verif_loader = DataLoader(verif_set, batch_size=BATCH_SIZE, shuffle=True)
learning_rate = 0.001
num_epochs = 50

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
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        ## forward + backprop + loss
        output = model(data)
        labels = labels.long()
        loss = criterion(output, labels.view(len(labels),1))
        optimizer.zero_grad()
        loss.backward()

        ## update model params
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(output, labels, BATCH_SIZE)
    
    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc/i)) 