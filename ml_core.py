# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:25:18 2023

@author: myuey
"""
import copy
import torch 
import torch.nn as nn
import pairdataset as pada
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
class modelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(30, 60)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(60, 120)
        self.fill_layer = nn.Linear(120,120)
        self.layer3 = nn.Linear(120, 20)
        self.output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid() 
    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.fill_layer(x))
        x = self.act(self.layer3(x))
        x = self.output(x)
        x = self.sigmoid(x)
        return x
class modelB(nn.Module):
    def __init__(self, input_size = 30, hidden_size = 60, num_classes = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()                   
    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.sigmoid(self.fc2(out)) #sigmoid as we use BCELoss
        return out
class modelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 32, kernel_size = 3, padding = 1)
        self.a1 = nn.ReLU()
        self.c2 = nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1)
        self.a2 = nn.ReLU()
        self.m1 = nn.MaxPool2d(2,2)
        
        self.c3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.a3 = nn.ReLU()
        self.c4 = nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1)
        self.a4 = nn.ReLU()
        self.m2 = nn.MaxPool2d(2,2)
        
        self.c5 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.a5 = nn.ReLU()
        self.c6 = nn.Conv2d(256 ,8, kernel_size = 3, stride = 1, padding = 1)
        self.a6 = nn.ReLU()
        self.m3 = nn.MaxPool2d(2,2)
        
        self.f1 = nn.Flatten()
        self.li_1 = nn.Linear(12,128)
        self.a7 = nn.ReLU()
        self.li_2 = nn.Linear(128,6)
        self.a8 = nn.ReLU()
        self.li_3 = nn.Linear(6,1)
    def forward(self, x):
        x = self.a1(self.c1(x))
        
        x = self.a2(self.c2(x))
        x = self.m1(x)
        x = self.a3(self.c3(x))
        x = self.a4(self.c4(x))
        x = self.m2(x)
        x = self.a5(self.c5(x))
        x = self.a6(self.c6(x))
        x = self.m3(x)
        x = x.view(4,12)
        x = self.f1(x)
        x = self.li_1(x)
        x = self.a7(x)
        x = self.li_2(x)
        x = self.a8(x)
        x = self.li_3(x)
        return x

class modelD(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 32, kernel_size = 3, padding = 1)
        self.a1 = nn.ReLU()
        self.c2 = nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1)
        self.a2 = nn.ReLU()
        self.m1 = nn.MaxPool2d(2,2)
        self.f1 = nn.Flatten()
        self.li_1 = nn.Linear(2160,128)
        self.a7 = nn.ReLU()
        self.li_2 = nn.Linear(128,32)
        self.a8 = nn.ReLU()
        self.li_3 = nn.Linear(32,1)
    def forward(self, x):
        x = self.a1(self.c1(x))
        
        x = self.a2(self.c2(x))
        x = self.m1(x)
        x = x.view(4,16,-1)
        x = self.f1(x)
        x = self.li_1(x)
        x = self.a7(x)
        x = self.li_2(x)
        x = self.a8(x)
        x = self.li_3(x)
        return x
BATCH_SIZE = 4

## transformations
transform = transforms.Compose([transforms.ToTensor()])
train_set = pada.PairDataset("","train.npy","train_labels.npy")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
a = None
for data, labels in train_loader:
    print("Batch dimensions:", data.shape)
    a = data.shape
    print("Label dimensions:", labels.shape)
    break    
verif_set = pada.PairDataset("","verif.npy","verif_labels.npy")
verif_loader = DataLoader(verif_set, batch_size=BATCH_SIZE, shuffle=True)
counter = 0
for data2, labels2 in train_loader:
    if(data2.shape != a):
        counter+=1
        print(counter)
        print(data2.shape)

learning_rate = 0.001
num_epochs = 500

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

