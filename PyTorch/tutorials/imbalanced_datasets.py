#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:40:10 2022

@author: srujanvajram
"""

import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader 
import torchvision.transforms as transforms
import torch.nn as nn

# To combat imbalanced datasets we can either 
# 1. Oversample from the less populated class
# 2. Weight the classes

# WEIGHTING CLASSES 
# The loss is multiplied by 50 for the second class assuming two classes
loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1,50]))

# OVERSAMPLING 
def get_loader(root, batch): 
    
    my_transforms = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        ])
    
    # Can use this to assign a loader to a directory with image folders
    dataset = datasets.ImageFolder(root = root, transform=my_transforms())
    class_weights = [1, 50]
    # We need to assign each sample in the dataset the respective class weight
    sample_weights = [0] * len(dataset)
    
    for index, (data, label) in enumerate(dataset): 
        sample_weights[index] = class_weights[label]
        
    # Now we can create the sampler 
    # This is an object we can pass intowhen creating the DataLoader
    # Note: replacement=True will ensure we are equally sampling from the classes (i.e oversample from the sparse classes)
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True)
    
    # This loader will ensure that the images are passed through the nextwork with the attached weights
    loader = DataLoader(dataset, 64, sampler=sampler)
    
    return loader
 
    
    

