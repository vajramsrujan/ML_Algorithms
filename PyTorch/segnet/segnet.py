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

# ============================================================================= # 
#  SegNet
class SegNet(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        
        # -------------------------# 
        # Encoder block 1
        self.encoder_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn1 = nn.BatchNorm2d(64)
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn2 = nn.BatchNorm2d(64)
        
        self.encoder_mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        
        # -------------------------# 
        # Encoder block 2
        self.encoder_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn3 = nn.BatchNorm2d(128)
        self.encoder_conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn4 = nn.BatchNorm2d(128)
        
        self.encoder_mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        
        # -------------------------# 
        # Encoder block 3
        self.encoder_conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn5 = nn.BatchNorm2d(256)
        self.encoder_conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn6 = nn.BatchNorm2d(256)
        
        self.encoder_mp3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        
        # ============================ # 
        # Decoder block 1
        self.decoder_mup1 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.decoder_conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn1 = nn.BatchNorm2d(128)
        self.decoder_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn2 = nn.BatchNorm2d(128)
        
        # -------------------------# 
        # Decoder block 2
        self.decoder_mup2 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.decoder_conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn3 = nn.BatchNorm2d(64)
        self.decoder_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn4 = nn.BatchNorm2d(64)
        
        # -------------------------# 
        # Decoder block 3
        self.decoder_mup3 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.decoder_conv5 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn5 = nn.BatchNorm2d(num_classes)
        self.decoder_conv6 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn6 = nn.BatchNorm2d(num_classes)
        
        # -------------------------# 
        # Prediction layer
        

    def forward(self, x):
        # Encoder block 1 pass
        x = F.relu( self.encoder_bn1( self.encoder_conv1(x) ) )
        x = F.relu( self.encoder_bn2( self.encoder_conv2(x) ) )
        x, indices_mp1 = self.encoder_mp1(x)
        
        # Encoder block 2 pass
        x = F.relu( self.encoder_bn3( self.encoder_conv3(x) ) )
        x = F.relu( self.encoder_bn4( self.encoder_conv4(x) ) )
        x, indices_mp2 = self.encoder_mp2(x)
        
        # Encoder block 3 pass
        x = F.relu( self.encoder_bn5( self.encoder_conv5(x) ) )
        x = F.relu( self.encoder_bn6( self.encoder_conv6(x) ) )
        x, indices_mp3 = self.encoder_mp3(x)
        
        # Decoder block 1 pass
        x = self.decoder_mup1(x, indices_mp3)
        x = F.relu( self.decoder_bn1( self.decoder_conv1(x) ) )
        x = F.relu( self.decoder_bn2( self.decoder_conv2(x) ) )
        
        # Decoder block 2 pass
        x = self.decoder_mup2(x, indices_mp2)
        x = F.relu( self.decoder_bn3( self.decoder_conv3(x) ) )
        x = F.relu( self.decoder_bn4( self.decoder_conv4(x) ) )
        
        # Decoder block 3 pass
        x = self.decoder_mup3(x, indices_mp1)
        x = F.relu( self.decoder_bn5( self.decoder_conv5(x) ) )
        x = F.relu( self.decoder_bn6( self.decoder_conv6(x) ) )
        
        # x = torch.squeeze(x)
        
        return x

# ============================================================================= # 
# # Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    
    model.eval()
    image_accuracies = []
    
    with torch.no_grad():
        for index, (data, target) in enumerate(loader):
            print("checking batch " + str(index))
            data = data.to(device=device)
    
            scores = model(data)
            
            for i in range(scores.shape[0]):
                score = scores[i, :, :, :]
                
                score = torch.argmax(score.squeeze(), dim=0).cpu().detach().numpy()
                true_label = target[i, :, :].numpy()
                matching = score == true_label
                
                accuracy = matching.sum() / len(matching.flatten())
                image_accuracies.append(accuracy*100)
                
            
    model.train()
    return np.array(image_accuracies)

# ============================================================================= # 

torch.cuda.empty_cache()

# Compose transformations 
data_transforms = transforms.Compose([
    transforms.Resize((256,256)),   
    transforms.ToTensor(),  
    transforms.Normalize( mean = [0.1600, 0.1959, 0.2559], 
                          std=[0.2209, 0.2456, 0.2530] )
    ])   

target_transforms = transforms.Compose([
    transforms.Resize((256,256)),   
    ])   

# Hyperparameters
in_channels = 3
learning_rate = 0.01
batch_size = 16
num_epochs = 10
num_classes = 3

# Load custom dataset
dataset = SegNetDataSet(r'C:\Users\vajra\Documents\GitHub\ML_playground\PyTorch\segnet\archive', 
                        data_transforms=data_transforms, target_transforms=target_transforms)

# Produce test and train sets
train_set, test_set = torch.utils.data.random_split(dataset, [329, 37]) # 90% 10% split between train and test 

# # Compute mean and std of test data
# train_loader = DataLoader(dataset=train_set, batch_size=len(train_set))
# data, targets = next(iter(train_loader))
# means = data[:, 0, :, :].mean(), data[:, 1, :, :].mean(), data[:, 2, :, :].mean()
# stds = data[:, 0, :, :].std(), data[:, 1, :, :].std(), data[:, 2, :, :].std()

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize network
model = SegNet(in_channels=in_channels, num_classes=num_classes).to(device)

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
            
# ============================================================================= # 

training_accuracies = check_accuracy(train_loader, model)
testing_accuracies = check_accuracy(test_loader, model)

print(f"Accuracy on training set " + str(training_accuracies.mean()))
print(f"Accuracy on test set " + str(testing_accuracies.mean()))