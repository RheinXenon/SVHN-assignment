import torch
import random
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(testloader, model, device, num_images=32):
    """
    可视化测试集上随机的图像，显示真实标签和预测标签
    """
    model.eval()  # 设置模型为评估模式
    images, labels = next(iter(testloader))  # 获取一个批次的测试数据
    images, labels = images.to(device), labels.to(device)

    # 获取模型预测
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # 获取预测类别

    # 随机选取 num_images 张图片
    indices = random.sample(range(images.size(0)), num_images)
    selected_images = images[indices]
    selected_labels = labels[indices]
    selected_preds = preds[indices]

    # 将 Tensor 转为 NumPy 数据
    selected_images = selected_images.cpu().numpy()
    selected_labels = selected_labels.cpu().numpy()
    selected_preds = selected_preds.cpu().numpy()

    # 可视化
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        ax = plt.subplot(4, 8, i + 1)
        img = selected_images[i].transpose((1, 2, 0))  # 将图像维度调整为 HWC
        img = np.clip(img, 0, 1)  # 确保像素值在 [0, 1] 范围内
        plt.imshow(img)
        plt.title(f"True: {selected_labels[i]}\nPred: {selected_preds[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()