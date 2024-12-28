import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN1(nn.Module):
    def __init__(self):
        super(SimpleCNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=256)
        # 输出层：输入特征数为 256，输出特征数为 10（对应数字 0-9 的分类）
        self.fc2 = nn.Linear(in_features=256, out_features=10)
        # Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        # 展平操作，将多维特征图展开成一维向量
        x = x.view(-1, 128 * 4 * 4)
        # 全连接层，ReLU 激活，Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 输出层
        x = self.fc2(x)
        return x
