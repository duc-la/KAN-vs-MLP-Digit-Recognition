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
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'Convolutional-KANs-master-master')))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'Convolutional-KANs-master-master', 'kan_convolutional')))

from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional.KANLinear import KANLinear

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

class KANNetwork(nn.Module):
    def __init__(self, device: str = 'cuda'):
        super().__init__()


        self.kan1 = KANLinear(
            625,
            10,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
        )