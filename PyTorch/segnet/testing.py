#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:00:55 2022

@author: srujanvajram
"""

import numpy as np
from PIL import Image 
from matplotlib import pyplot as plt
import torch

# for i in range(1,367): 
    
#     image = Image.open('archive/masks/' + str(i).zfill(3) + '.png').convert('L')
#     image.save('archive/flattened_masks/' + str(i).zfill(3) + '.png')

image = Image.open('archive/masks/' + str(4).zfill(3) + '.png').convert('L')
im2arr = np.array(image)