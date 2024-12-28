import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

# 下载并加载 SVHN 数据集
train_dataset = datasets.SVHN(root='./data', split='train', download=True)

# 获取前 32 张图片和标签
images = train_dataset.data[:32]  # shape: (32, 3, 32, 32)
labels = train_dataset.labels[:32]

# 调整图片的形状为 (32, 32, 32, 3)，以便用于可视化
images = np.transpose(images, (0, 2, 3, 1))

# 创建一个网格布局，6 行 6 列（或其他合适布局）
fig, axs = plt.subplots(4, 8, figsize=(16, 8))  # 4 行 8 列，展示 32 张图片
axs = axs.ravel()

# 绘制每张图片
for i in range(32):
    axs[i].imshow(images[i])
    axs[i].set_title(f"Label: {labels[i]}")
    axs[i].axis('off')  # 关闭坐标轴

plt.tight_layout()
plt.show()
