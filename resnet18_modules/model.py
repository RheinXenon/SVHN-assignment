import torch
from torch import nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 使用 torchvision 提供的预训练 ResNet18 模型
        self.resnet = models.resnet18(pretrained=True)
        
        # 修改第一个卷积层和删除 maxpool
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # 删除 MaxPool 层
        
        # 修改全连接层以适应 SVHN 数据集 (10 类别)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # 10 个类别
        )

    def forward(self, x):
        return self.resnet(x)