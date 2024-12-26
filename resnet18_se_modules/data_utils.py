import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 三个通道
    ])

    trainset = datasets.SVHN(root='../data', split='train', download=True, transform=transform)
    testset = datasets.SVHN(root='../data', split='test', download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
