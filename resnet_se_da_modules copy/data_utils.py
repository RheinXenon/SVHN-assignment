import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128):
    # 数据增强的变换（仅应用于训练集）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),         # 随机裁剪到32x32，四周填充4个像素
        transforms.RandomHorizontalFlip(),            # 随机水平翻转
        transforms.ColorJitter(brightness=0.2,        # 随机调整亮度、对比度、饱和度
                               contrast=0.2,
                               saturation=0.2),
        transforms.ToTensor(),                        # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])

    # 测试集的变换（仅标准化）
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),                  # 调整到32x32尺寸
        transforms.ToTensor(),                        # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])

    # 加载训练集和测试集
    trainset = datasets.SVHN(root='../data', split='train', download=True, transform=transform_train)
    testset = datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)

    # 创建数据加载器
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader
