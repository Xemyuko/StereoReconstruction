# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:53:43 2023

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
class ClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
BATCH_SIZE = 4
train_set = pada.PairDataset("","train.npy","train_labels.npy")
train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
a = None
for data, labels in train_dl:
    print("Batch dimensions:", data.shape)
    a = data.shape
    print("Label dimensions:", labels.shape)
    break    
val_set = pada.PairDataset("","verif.npy","verif_labels.npy")
val_dl = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
counter = 0
for data2, labels2 in train_dl:
    if(data2.shape != a):
        counter+=1
        print(counter)
        print(data2.shape)


class ModelA(nn.Module):
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
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader):
    
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history
num_epochs = 30
lr = 0.001   
model = ModelA() 
history = fit(num_epochs, lr, model, train_dl, val_dl)