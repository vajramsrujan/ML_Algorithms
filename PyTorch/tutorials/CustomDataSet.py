#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 15:16:12 2021

@author: srujanvajram
"""

import os 
import pandas as pd
import torch 
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms

class CustomDataSet(Dataset): 
    
    def __init__(self, csv, root, transform=None): 
        self.annotations = pd.read_csv(csv)
        self.root_dir = root
        self.transform = transform 
        
    def __len__(self): 
        return len(self.annotations)
        
    def __getitem__(self, index): 
        img_path = os.path.join(self.root, self.annotations.iloc[index, 0])
        y_label = torch.tensor( int(  self.annotations.iloc[index, 1]  ) )
        image = io.imread(img_path)
        
        if self.transform: 
            image = self.transform(image)
        
        return (image, y_label)
    
    
# If we want to use this custon dataset class in another file wed 
# first import it 
# from CustomDataSet import CustomDataSet

dataset = CustomDataSet('csv_file.csv', 'root_directory_name', transfrom = transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000]) # 20k to train, 5k to test
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)


    
        