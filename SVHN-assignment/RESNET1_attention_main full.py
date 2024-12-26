import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

# 初始化 TensorBoard
writer = SummaryWriter(log_dir='./logs')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 三个通道
])

# 下载并加载 SVHN 数据集
trainset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
testset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# 定义 SEBlock
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

# 定义 ResNet + SEBlock 模型
class ResNetWithSE(nn.Module):
    def __init__(self):
        super(ResNetWithSE, self).__init__()
        # 使用 ResNet18 作为骨干网络
        self.resnet = models.resnet18(pretrained=True)
        
        # 替换 ResNet 的第一个卷积层和最后的全连接层以适应 SVHN 数据
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # 去掉初始的 MaxPool 层，因为输入图像已经很小
        num_features = self.resnet.fc.in_features

        # 替换 ResNet 的全连接层
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # SVHN 有 10 个类别
        )

        # 在 ResNet 的某些卷积层后加入 SEBlock
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        x = self.resnet.layer1(x)
        x = self.se1(x)  # Apply SEBlock after layer1

        x = self.resnet.layer2(x)
        x = self.se2(x)  # Apply SEBlock after layer2

        x = self.resnet.layer3(x)
        x = self.se3(x)  # Apply SEBlock after layer3

        x = self.resnet.layer4(x)
        x = self.se4(x)  # Apply SEBlock after layer4

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
model = ResNetWithSE().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 使用学习率衰减（每10个epoch降低学习率）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

script_name = os.path.basename(__file__).split('.')[0]
save_dir = './models'
os.makedirs(save_dir, exist_ok=True)

# 训练模型
for epoch in range(30):
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    y_true_train = []
    y_pred_train = []
    
    model.train()
    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}")
    for i, data in progress_bar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # For F1 calculation
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

        progress_bar.set_postfix(loss=(running_loss / (i + 1)))
        
        writer.add_scalar('Training Loss', loss.item(), epoch * len(trainloader) + i)

    train_acc = 100 * train_correct / train_total
    train_f1 = f1_score(y_true_train, y_pred_train, average='macro')
    avg_train_loss = running_loss / len(trainloader)

    # 验证集评估
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            # For F1 calculation
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(y_true_val, y_pred_val, average='macro')
    avg_val_loss = val_loss / len(testloader)

    # Logging to TensorBoard
    writer.add_scalar('Average Training Loss per Epoch', avg_train_loss, epoch)
    writer.add_scalar('Training Accuracy', train_acc, epoch)
    writer.add_scalar('Training F1 Score', train_f1, epoch)

    writer.add_scalar('Validation Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation Accuracy', val_acc, epoch)
    writer.add_scalar('Validation F1 Score', val_f1, epoch)

    scheduler.step()  # Update the learning rate

    if (epoch + 1) % 5 == 0:  # 每 5 个 epoch 保存一次
        model_path = os.path.join(save_dir, f"{script_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到 {model_path}")

print('训练结束')

# 测试模型
correct = 0
total = 0
model.eval()
with torch.no_grad():
    progress_bar = tqdm(testloader, desc="测试集")
    for data in progress_bar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'在测试集上测试得到的准确度: {100 * correct / total:.2f} %')

writer.close()
