#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 22:20:26 2021

@author: srujanvajram
"""

import torch
import torchvision.transforms as transforms 
from torchvision.utils import save_image
from CustomDataSet import CustomDataSet

# Compose transformations 
my_transforms = transforms.Compose([
    transforms.ToPILImage(),                    # Convert target image to PIL format
    transforms.Resize((256,256)),               
    transforms.RandomHorizontalFlip(p=0.5),     # 50% probability of horizontal flip 
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),                      # Convert image to tensor 
    transforms.Normalize(mean = [0,0,0], std=[1,1,1]) # for each image will do (value in channel - mean in channel) / std in channel)
    # The current values right now will do nothing (identity), but in practice we need to first compute these values and then 
    # place them in place of what we have above. 
    ])                                          

# We can then pass this transform composition to the dataset

dataset = CustomDataSet('csv_file.csv', 'root_path_to_images', transfrom = my_transforms)

