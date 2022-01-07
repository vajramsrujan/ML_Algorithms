#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 15:21:59 2022

@author: srujanvajram
"""

# ============================================================================ #

# 1. Remember to toggle model.eval() before checking accuracy (put it in the check_accuracy function)
# it will prevent dropout/regulrization at test time

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    
    # !!!
    model.eval()
    # !!!

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples

# ============================================================================ #

# 2. Do not forget optimizer.zero_grad() in backprop 
# Otherwise it will be taking an accumulation of all past gradients with the current computed gradient 

# backward
optimizer.zero_grad()
loss.backward()

# ============================================================================ #

# 3. Try to overfit one batch before starting training
# This is a good way to bug test the network , if its training on only one a batch for many epochs,
# we expect to see overfitting

# Use
data, targets = next(iter(train_loader))

# Then train only this data,target batch in the epoch loop