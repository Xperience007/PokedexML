import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from classifier import NeuralNet

#Transforms to apply to data
transform = transforms.Compose([
    transforms.Resize((475, 475)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#Grab the data and split it into testing and training sets
data_path = "C:\\Users\\tntbi\\Downloads\\data"
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])

#Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

image, label = dataset[0]
print(image.shape)

#Set up optimizer, loss, and neural network
net = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#Epochs to run for
epochs = 30

#Device
device = torch.device("cuda")

for epoch in range(epochs):
    avg_loss = 0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training Batches")):
        inputs, labels = batch

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = nn.functional.cross_entropy(outputs, labels)
        loss.backward()

        avg_loss += loss.item()

        optimizer.step()
    
    print(f'Loss: {avg_loss / len(train_dataloader)}')
    print(f"Epoch: {epoch+1}")

    torch.save(net.state_dict(), f"./classification/epoch_{epoch+1}.pth")