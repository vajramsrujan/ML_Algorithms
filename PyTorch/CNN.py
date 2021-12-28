# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:29:59 2021

@author: vajra
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connceted NN

# Inherit from parent nn class
class CNN(nn.Module):
    
    def __init__(self, in_channels = 1, num_classes):
        
        '''
            input_size: Starting layer size
            num_classes: ending layer size
        '''
        # Call initialization method of parent class
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels=1, out_channels=8, 
                               kernel_size=(3,3), stride=(1,1), padding=(1,1)) # Layer 1
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.conv2 = nn.Conv2d(n_channels=8, out_channels=16, 
                               kernel_size=(3,3), stride=(1,1), padding=(1,1)) # Layer 1
        
        self.fc1 = nn.Linear(50, num_classes) # Final layer
        
    def forward(self, x):
        
        '''
            Forward passes layer 'x' through the network 
        '''
        
        # Notice we can pass x in as an argument to the layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = NN(784, 10)
# The MNST database has 28*28 = 784 flattened image size
# So the below tensor represents 64 such images
x = torch.randn(64,784) 
# You can pass a tensor directly into a model.
# It will output the last layer
print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
epochs = 1

# Load data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform=transforms.ToTensor(), 
               download=True) 
# train flag to mark dataset as training set
# transform to tensors in case the data we laod is in numpy format
# downlaod flag to download the data in case it doesnt exist in the folder already 

# This loader will prepare the data  (ex, batches and shuffles it) 
# prior to feeding it into the model
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Same preparations for test dataset 
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform=transforms.ToTensor(), 
               download=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
# Sends the model to the available device for computation
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(epochs):
    for batch_index, (data, targets) in enumerate(train_loader):
        # Send the data and truth data to available device
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Flatten data to shape 64, 784
        # One batch of 64 images, each row corresponsing to 1 image
        data = data.reshape(data.shape[0], -1)
        
        # Forward pass
        scores = model(data) # Gets the final output layer scores
        loss = criterion(scores, targets) # Computes the loss function score
        
        # backprop 
        optimizer.zero_grad() # First, reset the gradients so that the previous gradients arent stored
        loss.backward() # Computes the gradients 
        
        # Gradient descent
        optimizer.step()

# Check model accuracy 

def check_accuracy(loader, model): 
    num_correct = 0
    num_samples = 0
    model.eval()
    
    # Letting pytorch know that it does not need 
    # to compute gradients 
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(float(num_correct/num_samples)*100)

    model.train()
    return 

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
