# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:28:20 2021

@author: vajra
"""

import torch
import numpy as np

# Create 2D tensors 
x = torch.empty(3,3)
y = torch.ones(3,3)
z = torch.rand(3,3)
a = torch.ones(2,2, dtype=torch.float16)
i = torch.eye(3,3) # Identity matrix
a = torch.arange(start=0, end=5, step=1)
a = torch.linspace(start=0.1, end=1, steps=10)

# Convert tensors to other data types
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())   # int 16
print(tensor.long())    # int 64
print(tensor.half())    # float 16
print(tensor.float())   # flaot 32
print(tensor.double())  # float 64

# Create tensor from list
x = torch.tensor([2.5, 3.2, 1.7])

# Add and subtract tensors with element wise matrix addition
y = torch.rand(3,3)
z = torch.rand(3,3)
m = y + z
# or
y+=z
# or 
y.add_(z) # same as y += z

m = y - z 

# Multiply tensors element wise
m = y * z 
m = y / z
# in place
y.mul_(z)
y.div_(z)

# Dot product (must be 1D)
x = torch.tensor([1,2])
y = torch.tensor([1,2])
z = torch.dot(x,y)

# Matrix exponentiation
tensor = torch.ones(2,2)*3
tensor.matrix_power(3)

# Broadcasting
x = torch.rand(5,5)
y = torch.rand(1,5)
z = x - y # The 1,5 will expand via duplication to 5,5

# You can slice tensors like matrices
m = y[:,0]

# If you want to extract the actual value from the tensor
m = y[1,1].item()

# If you want to flatten tensor into 1D
x = torch.rand(4,4)
y = x.view(-1) # Flattens x into 1D 16 element tensor
y = x.view(16,1) # Converts to shape 16,1
y = x.view(-1, 8) # The -1 will will become the appropriate number for (? , 8) 

# You can extend this flattening operation to specific 
# dimensions of n dimensional matrices
x = torch.rand(64,2,5)
z = x.view(64, -1) # Will flatten 2,5 to 10
# now z.shape = 64, 10

# Other useful operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
mean_x = torch.mean(x.float(), dim=0) # mean only works on floats
z = torch.eq(x,y) # Returns tensor list of booleans of true or false depending on equality btwn tensors
sorted_y, indices = torch.sort(x, dim=0, descending=False) # Sort tensor 
z = torch.clamp(x, min=0, max=10) # Any value lower than zero set to zero, same for 10 (greater than 10 set to 10)

# Convert tensor to numpy
# IMPORTANT 
# This type of conversion will TIE TOEGETHER the 
# tensor and numpy if they are on the savme device (ex: GPU)
# i.e changing tesor x will change numpy array y
x = torch.ones(5)
y = x.numpy()
# Will again tie them together (i.e NOT unique)
x = np.ones(6)
y = torch.from_numpy(x)

# You need to keep track of where the tensors are being computed
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Declare where you are creating the tensor
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    # Move y to the device prior to computing with x
    y = y.to(device)
    z = x + y # Compute
    # Before you can convert back to numpy array, you need to 
    # Move it back onto the CPU
    z = z.to("cpu")
    z.numpy()


