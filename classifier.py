import torch
import torch.nn as nn
import torch.nn.functional as F

#CNN structure
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=108, kernel_size=5)
        self.fc1 = nn.Linear(in_features=(108 * 115 * 115), out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=185)
        self.fc3 = nn.Linear(in_features=185, out_features=151)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x