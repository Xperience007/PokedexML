import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

#Transforms to apply to data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Grab the data and split it into testing and training sets
data_path = "C:\\Users\\tntbi\\Downloads\\data"
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])

#Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=108, kernel_size=5)
        self.fc1 = nn.Linear(in_features=(108 * 5 * 5), out_features=540)
        self.fc2 = nn.Linear(in_features=540, out_features=378)
        self.fc3 = nn.Linear(in_features=378, out_features=151)