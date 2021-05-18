import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

train_labels = train_df['label'].values
train_images = (train_df.iloc[:,1:].values).astype('float32')
test_images  = (test_df.values).astype('float32')

train_images, val_images, train_labels, val_labels = train_test_split(train_images,
        train_labels, stratify=train_labels, random_state=123, test_size=0.2)

print(train_images.shape,val_images.shape)
print(train_labels.shape,val_labels.shape)

train_images = train_images.reshape(train_images.shape[0], 28, 28)
val_images   = val_images.reshape(val_images.shape[0], 28, 28)
test_images  = test_images.reshape(test_images.shape[0], 28, 28)

train_images_tensor = torch.tensor(train_images)/255.
train_labels_tensor = torch.tensor(train_labels)
val_images_tensor   = torch.tensor(val_images)/255.
val_labels_tensor   = torch.tensor(val_labels)

#Make a training dataset object
train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
val_tensor   = TensorDataset(val_images_tensor, val_labels_tensor)


train_loader = DataLoader(train_tensor, batch_size=16, num_workers=0, shuffle=True)
val_loader   = DataLoader(val_tensor,   batch_size=16, num_workers=0, shuffle=True)


image, label = next(iter(train_loader))
print(image.shape)
print(label.shape)
#print(label)

#for batch_idx, (data, target) in enumerate(train_loader):
#    print(data.shape, target.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_block = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_block = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128*7*7, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x

conv_model = Net()
conv_model.train()

optimizer = optim.Adam(params=conv_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

#image, label = next(iter(train_loader))
#print(conv_model(image))


exp_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

if torch.cuda.is_available():
    conv_model = conv_model.cuda()
    optimizer  = conv_model.cuda()

def train_model(num_epoch):
    conv_model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.unsqueeze(1)
        data , target = data, target

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = conv_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx+1)%100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}, LR: {:.6f}'.format(
                num_epoch, (batch_idx+1)*len(data), len(train_loader.dataset),
                100. * (batch_idx+1)/len(train_loader), loss.data.item(),
                optimizer.param_groups[0]['lr']))

def evaluate(data_loader):
    conv_model.eval()
    loss = 0 
    correct = 0 

    for data, target in data_loader:
        data = data.unsqueeze(1)
        data, target = data, target

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = conv_model(data)

        loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    loss /= len(data_loader.dataset)
    print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


 
#train_model(1)












