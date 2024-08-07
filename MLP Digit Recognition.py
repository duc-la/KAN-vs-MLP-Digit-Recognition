import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import copy
import time

# Ensure the root path is a string
root = './data'
if not isinstance(root, str):
    raise ValueError("Root path should be a string")

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

class MNISTDatasetLoader(Dataset):
    def __init__(self, csv_file, root, transform=None):
        self.inputData = pd.read_csv(os.path.join(root, csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.inputData)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        RowSample = self.inputData.iloc[idx, :]
        sample = {'x': np.reshape(np.array(RowSample[1:]), (28, 28)), 'label': RowSample[0]}

        if self.transform:
            sample['x'] = self.transform(sample['x'])
        return sample

class MNISTDatasetLoaderTest(Dataset):
    def __init__(self, csv_file, root, transform=None):
        self.inputData = pd.read_csv(os.path.join(root, csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.inputData)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        RowSample = self.inputData.iloc[idx, :]
        sample = {'x': np.reshape(np.array(RowSample[:]), (28, 28))}

        if self.transform:
            sample['x'] = self.transform(sample['x'])
        return sample

class ThreeLayerLinear(nn.Module):
    def __init__(self):
        super(ThreeLayerLinear, self).__init__()
        self.layer1 = nn.Linear(784, 500)
        self.layer2 = nn.Linear(500, 300)
        self.layer3 = nn.Linear(300, 200)
        self.layer4 = nn.Linear(200, 100)
        self.layer5 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        #x = self.layer3(x)
        return x

def train_model(model, train_loader, epochs=50, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.view(-1, 28*28)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

network = ThreeLayerLinear()
train_model(network, trainloader)
