import torch
import os
from data_utils import get_data_loaders
from train import train_model
from test import test_model
from model import ResNetWithSE
from visualization import visualize_predictions
from torchsummary import summary

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_data_loaders(batch_size=128)

    save_dir = "../models"
    os.makedirs(save_dir, exist_ok=True)

    # 训练模式
    # model = ResNetWithSE().to(device)
    # train_model(model,trainloader, testloader, device, num_epochs=30, save_dir=save_dir)



    # 评估测试模式
    # test_model(testloader, model, device)

    # 测试可视化模式
    model = ResNetWithSE().to(device)
    model_path = "../models/resnet18_se_da_epoch30.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    summary(model, (3, 32, 32))
    print(f"模型加载：{model_path}")

    test_accuracy = test_model(testloader, model, device)

    visualize_predictions(testloader, model, device, 32)
