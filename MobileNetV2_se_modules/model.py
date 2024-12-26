import torch
from torch import nn
from torchvision import models

class MobileNetV2Custom(nn.Module):
    def __init__(self):
        super(MobileNetV2Custom, self).__init__()
        # Load pre-trained MobileNetV2 model
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        
        # Modify the first convolutional layer
        self.mobilenet_v2.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the classifier and replace it with a custom one
        num_features = self.mobilenet_v2.last_channel
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # SVHN has 10 classes
        )

    def forward(self, x):
        x = self.mobilenet_v2(x)
        return x