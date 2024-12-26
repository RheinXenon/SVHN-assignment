import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# # 数据预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))])

# # 下载并加载SVHN数据集
# trainset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
# testset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
train_dataset = datasets.SVHN(root='./data', split='train', download=True)

images = train_dataset.data[:10]  # shape: (10, 3, 32, 32)
labels = train_dataset.labels[:10]

images = np.transpose(images, (0, 2, 3, 1))

# Plot the images
fig, axs = plt.subplots(2, 5, figsize=(12, 6))
axs = axs.ravel()

for i in range(10):
    axs[i].imshow(images[i])
    axs[i].set_title(f"Label: {labels[i]}")
    axs[i].axis('off')

plt.tight_layout()
plt.show()
