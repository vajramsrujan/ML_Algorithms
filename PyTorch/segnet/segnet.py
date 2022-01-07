#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:28:23 2022

@author: srujanvajram
"""

# Imports
import numpy as np

import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
from SegNetDataSet import SegNetDataSet

# Simple CNN
class SegNet(nn.Module):
    
    def __init__(self, in_channels=3):
        super(SegNet, self).__init__()
        
        # -------------------------# 
        # Encoder block 1
        self.encoder_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn1 = nn.BatchNorm2d(64)
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn2 = nn.BatchNorm2d(64)
        self.encoder_mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # -------------------------# 
        # Encoder block 2
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn3 = nn.BatchNorm2d(128)
        self.encoder_conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn4 = nn.BatchNorm2d(128)
        self.encoder_mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # -------------------------# 
        # Encoder block 3
        self.encoder_conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn5 = nn.BatchNorm2d(256)
        self.encoder_conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn6 = nn.BatchNorm2d(256)
        self.encoder_mp3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # ============================ # 
        # Decoder block 1
        self.decoder_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn2 = nn.BatchNorm2d(256)
        self.decoder_mup1 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        # -------------------------# 
        # Decoder block 2
        self.decoder_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn3 = nn.BatchNorm2d(128)
        self.decoder_conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn4 = nn.BatchNorm2d(128)
        self.decoder_mup2 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        # -------------------------# 
        # Decoder block 3
        self.decoder_conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn5 = nn.BatchNorm2d(64)
        self.decoder_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn6 = nn.BatchNorm2d(64)
        self.decoder_mup3 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        # -------------------------# 
        # Prediction layer
        

    def forward(self, x):
        # Encoder block 1 pass
        x = F.relu( self.encoder_bn1( self.encoder_conv1(x) ) )
        x = F.relu( self.encoder_bn2( self.encoder_conv2(x) ) )
        x = self.encoder_mp1(x)
        
        # Encoder block 2 pass
        x = F.relu( self.encoder_bn3( self.encoder_conv3(x) ) )
        x = F.relu( self.encoder_bn4( self.encoder_conv4(x) ) )
        x = self.encoder_mp2(x)
        
        # Encoder block 3 pass
        x = F.relu( self.encoder_bn5( self.encoder_conv5(x) ) )
        x = F.relu( self.encoder_bn6( self.encoder_conv6(x) ) )
        x = self.encoder_mp3(x)
        
        # Decoder block 1 pass
        x = F.relu( self.decoder_bn1( self.decoder_conv1(x) ) )
        x = F.relu( self.decoder_bn2( self.decoder_conv2(x) ) )
        x = self.decoder_mup1(x)
        
        # Decoder block 2 pass
        x = F.relu( self.decoder_bn3( self.decoder_conv3(x) ) )
        x = F.relu( self.decoder_bn4( self.decoder_conv4(x) ) )
        x = self.decoder_mup2(x)
        
        # Decoder block 3 pass
        x = F.relu( self.decoder_bn5( self.decoder_conv5(x) ) )
        x = F.relu( self.decoder_bn6( self.decoder_conv6(x) ) )
        x = self.decoder_mup3(x)
        
        return x

torch.cuda.empty_cache()

# Compose transformations 
my_transforms = transforms.Compose([
    transforms.Resize((512,512)),   
    transforms.ToTensor(),          
    ])   

# Load custom dataset
dataset = SegNetDataSet(r'C:\Users\vajra\Documents\GitHub\ML_playground\PyTorch\segnet\archive', transforms = my_transforms)

# Produce test and train sets
train_set, test_set = torch.utils.data.random_split(dataset, [329, 37]) # 20k to train, 5k to test

# Compute mean and std of test data
# train_loader = DataLoader(dataset=train_set, batch_size=len(train_set))
# data, targets = next(iter(train_loader))
# means = data[:, 0, :, :].mean(), data[:, 1, :, :].mean(), data[:, 2, :, :].mean()
# stds = data[:, 0, :, :].std(), data[:, 1, :, :].std(), data[:, 2, :, :].std()

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 3
learning_rate = 0.001
batch_size = 32
num_epochs = 2

# Initialize network
model = SegNet(in_channels=3).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
losses = np.zeros((1,num_epochs)).flatten()

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses[epoch] = loss.item()
        
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")