import torch
from torch import nn
from torchvision import models

class ResNetWithSE(nn.Module):
    def __init__(self):
        super(ResNetWithSE, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # 去掉初始的 MaxPool 层
        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # SVHN 有 10 个类别
        )

        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        x = self.resnet.layer1(x)
        x = self.se1(x)

        x = self.resnet.layer2(x)
        x = self.se2(x)

        x = self.resnet.layer3(x)
        x = self.se3(x)

        x = self.resnet.layer4(x)
        x = self.se4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        y = self.fc1(y)
        y = torch.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)  # Reshape to (b, c, 1, 1)
        return x * y
