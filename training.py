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
learning_rate = 1e-4
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=0.05)

#Epochs to run for
epochs = 30

#Device
device = torch.device("cuda")
print(torch.cuda.is_available())
net = net.to(device)

net = net.train()
for epoch in range(epochs):
    avg_loss = 0
    inputs = []
    targets = []
    features = []
    max_grad_norm = 0
    avg_grad_norm = 0
    avg_correct = 0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training Batches")):
        try:
            inputs = batch[0].to(device) 
            targets = batch[1].to(device)

            optimizer.zero_grad()

            features = net(inputs)
            pred = torch.argmax(F.softmax(features), dim=1)
            truth = torch.argmax(targets, dim=1)
            correct = (pred == truth).sum().item() / truth.shape[0]

            loss = nn.functional.cross_entropy(features, targets)
            loss.backward()

            avg_loss += loss.item()
            avg_correct += correct

            optimizer.step()
        except Exception as e:
            pass
    
    print(f'Loss: {avg_loss / len(train_dataloader)}')
    print(f"Avg Train Correct: {avg_correct / len(train_dataloader)}")
    print(f"Epoch: {epoch+1}")

    torch.save(net.state_dict(), f"./classification/epoch_{epoch+1}.pth")