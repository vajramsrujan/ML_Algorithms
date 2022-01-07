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

# Simple CNN
class SegNet(nn.Module):
    
    def __init__(self, in_channels=1, num_classes=10):
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
        self.decoder_mp1 = nn.max_unpool2d(kernel_size=(2, 2), stride=(2, 2))
        # -------------------------# 
        # Decoder block 2
        self.decoder_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn3 = nn.BatchNorm2d(128)
        self.decoder_conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn4 = nn.BatchNorm2d(128)
        self.decoder_mp2 = nn.max_unpool2d(kernel_size=(2, 2), stride=(2, 2))
        # -------------------------# 
        # Decoder block 3
        self.decoder_conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn5 = nn.BatchNorm2d(64)
        self.decoder_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn6 = nn.BatchNorm2d(64)
        self.decoder_mp3 = nn.max_unpool2d(kernel_size=(2, 2), stride=(2, 2))
        # -------------------------# 
        # Prediction layer
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x