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
    
#     image = Image.open('archive/images/' + str(i).zfill(3) + '.png').convert('L')
#     image.save('archive/greyscale_images/' + str(i).zfill(3) + '.png')

image = Image.open('archive/masks/' + "036" + '.png').convert('RGB')
im2arr = np.array(image)

im2arr[im2arr==1]=128
im2arr[im2arr==2]=255

arr2im = Image.fromarray(im2arr).convert('RGB')
arr2im.save("036_coloredmask.png")