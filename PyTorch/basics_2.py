# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:59:03 2021

@author: vajra
"""

# Tensor indexing

import torch

batch_size = 10
features = 25
x = torch.rand(batch_size,features)
 
x_row_1 = x[0]
x_col_1 = x[:,0]
x_slice = x[2, 0:10] # Grabs first 10 features from 2nd row
 
# Advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)]) # Get values smaller than 2 or greater than 8
print(x[x.remainder(2) == 0]) # Only choose elements whose remainder with 2 is zero

# Index finding and replacement
# If x is greater than 5, replace it with x, otherwise, replace with x*2
print(torch.where(x > 5, x, x*2))

# Uniqueness (set)
x = torch.tensor([1,1,2,3,4,4,4,5,6])
print(x.unique())

# Check dimension 
print(x.ndimension())

# Print number of elements in tensor
print(x.numel())

# Tensor reshaping
x = torch.arange(9)
x_3by3 = x.reshape(3,3) # Reshapes 
 
# Concatenation
x1 = torch.rand(2,3)
x2 = torch.rand(2,3)
x3 = torch.cat((x1,x2), dim=0) # concatenates rows
# dim=1 will concatenate column wise

# Rearrange dimensions
x = torch.rand(64,2,5)
z = x.permute(0,2,1)
# Numebrs refer to the indices of the dimensions. By
# rearranging these indices, we are swapping the dimensions
# (i.e '2' refers to the last dimension, and by putting it in the second
# position in permute, we are swapping the last dimension with the second last
print(z.shape)

 