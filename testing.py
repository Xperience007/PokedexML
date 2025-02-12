import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from classifier import NeuralNet
from training import Startup

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = Startup.transform(image)
    image = image.unsqueeze(0)
    return image

def test():
    #Create data loader
    test_dataloader = DataLoader(Startup.test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    net = NeuralNet()

    device = torch.device("cuda")
    net.load_state_dict(torch.load("./classification/epoch_30.pth", weights_only=True))
    net = net.to(device)
    net = net.eval()

    image_paths = ["C:\\Users\\tntbi\\Downloads\\molt.jpg"]
    images = [load_image(img) for img in image_paths]
    class_names = Startup.class_names

    avg_correct = 0

    with torch.no_grad():
        # For testing batches
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing Batches")):
            #Moving data to device
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            #Pass inputs into the model and calculate loss
            features = net(inputs)
            pred = torch.argmax(features, dim=1)
            correct = (pred == targets).sum().item() / targets.shape[0]

            avg_correct += correct

        test_correct = avg_correct/len(test_dataloader)
        print(f"Avg Test Correct: {test_correct}")

        # For testing one picture
        # for image in images:
        #     image = image.to(device)
        #     features = net(image)
        #     _, predicted = torch.max(features, 1)
        #     print(f'Prediction: {class_names[predicted.item()]}')


if __name__ == "__main__":
    test()