import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from classifier import NeuralNet
import warnings

warnings.simplefilter("ignore", UserWarning)

class Startup():
    #Transforms to apply to data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    #Grab the data and split it into testing and training sets
    data_path = "C:\\Users\\tntbi\\Downloads\\data"
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    class_names = dataset.classes
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])

def train():
    torch.backends.cudnn.benchmark = True

    #Create data loader
    train_dataloader = DataLoader(Startup.train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    #Set up optimizer, loss, and neural network
    net = NeuralNet()
    learning_rate = 1e-4
    optimizer = optim.AdamW(params=net.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, cooldown=1, threshold=0.01, mode="min", factor=0.5)

    #Epochs to run for
    epochs = 30

    #Device
    device = torch.device("cuda")
    net = net.to(device)

    net = net.train()
    for epoch in range(epochs):
        avg_loss = 0
        inputs = []
        targets = []
        features = []
        avg_correct = 0

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training Batches")):
            #Moving data to device
            inputs = batch[0].to(device) 
            targets = batch[1].to(device)

            optimizer.zero_grad()

            #Pass inputs into the model and calculate loss
            features = net(inputs)
            pred = torch.argmax(features, dim=1)
            truth = targets
            correct = (pred == truth).sum().item() / truth.shape[0]

            #Backpropigation
            loss = nn.functional.cross_entropy(features, targets, label_smoothing=0.1)
            loss.backward()

            avg_loss += loss.item()
            avg_correct += correct

            #Clipping gradients
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

            #Optimizer step
            optimizer.step()
        
        scheduler.step(avg_loss / len(train_dataloader))
        
        print(f'Loss: {avg_loss / len(train_dataloader)}')
        print(f"Avg Train Correct: {avg_correct / len(train_dataloader)}")
        print(f"Epoch: {epoch+1}")

        torch.save(net.state_dict(), f"./classification/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()