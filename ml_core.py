# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:25:18 2023

@author: myuey
"""

import torch 
import torch.nn as nn
import pairdataset as pada
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

device = torch.device("cuda:0")
def run_training(train_dataset, BATCH_SIZE, model):
    #set batch size and load data
    
    
    
    num_epochs = 2
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    def get_accuracy(logit, target, batch_size):
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects/batch_size
        return accuracy.item()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        train_acc = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(inputs)
            labels = labels.long()
            loss = criterion(output, labels.view(len(labels),1))
            loss.backward()
            optimizer.step()

            
            running_loss += loss.item()
            train_acc += get_accuracy(output, labels, BATCH_SIZE)
        # print statistics
        #model.eval()
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(epoch, running_loss / i, train_acc/i))
      
    PATH = './pair_model.pth'
    torch.save(model.state_dict(), PATH)
def check_model(model, PATH, test_loader):
    
    
    model.to(device)
    model.load_state_dict(torch.load(PATH))

    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(len(predicted)):
                if(predicted[i].item() == labels[i].item()):
                    correct+=1
            break
    print(correct)
    print(total)        
    print(f'Accuracy of the network on the test data: {100 * correct // total} %')

dataset = pada.PairDataset("","train.npy","train_labels.npy")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
BATCH_SIZE = 4
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
run_training(train_dataset, BATCH_SIZE, modelA)    