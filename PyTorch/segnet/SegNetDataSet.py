# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:00:37 2022

@author: vajra
"""

import glob
import os 
import pandas as pd
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SegNetDataSet(Dataset):
    
    def __init__(self, directory, transforms):
        super(SegNetDataSet, self).__init__()
        
        self.data = glob.glob(os.path.join(directory,'images','*.png'))
        self.targets = glob.glob(os.path.join(directory,'masks','*.png'))
        self.transforms = transforms
        # transforms.Normalize(mean = [(0.1564,0.1926,0.2511)], std=[(0.2174,0.2430,0.2511)]), 
        
    def __getitem__(self, index):
            data_path = self.data[index]
            target_path = self.targets[index]
            
            data =  Image.open(data_path) 
            target = Image.open(target_path) 
            if self.transforms: 
                data = self.transforms(data)
                target = self.transforms(target)
                
            return data, target

    def __len__(self):
        return len(self.data)