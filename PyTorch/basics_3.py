# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:57:19 2021

@author: vajra
"""

import torch
import numpy as np

# If you need to optimize a variable, you need
# to pass the requires_grad flag
x = torch.ones(3, requires_grad=True) # Needed for calculating gradients

# Say we have a function
y = (x**2).sum()
y.backward()
print(x.grad)
# y.backward() will compute dy/dx 
# This derivitive computation will then be stored in x.grad
# for each input of x

# If in a loop where y.backwards() is being called
# You need to make sure the gradients are reset
# To reset the stored gradients
x.grad.zero_()

 


