#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:00:55 2022

@author: srujanvajram
"""

import numpy as np
from PIL import Image 
from matplotlib import pyplot as plt

for i in range(1,367): 
    
    image = Image.open('archive/masks/' + str(i).zfill(3) + '.png').convert('L')
    im2arr = np.array(image)
    im2arr[im2arr > 0] = 255
    arr2im = Image.fromarray(im2arr)
    arr2im.save('archive/binary_masks/' + str(i).zfill(3) + '.png')


