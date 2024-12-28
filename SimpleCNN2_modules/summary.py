import torch
import os
from data_utils import get_data_loaders
from train import train_model
from test import test_model
from model import SimpleCNN2
from visualization import visualize_predictions
from torchsummary import summary
from colorama import Fore, Style, init

# 初始化 colorama
init(autoreset=True)

def colored_summary(model, input_size, device):
    """
    This function extends torchsummary.summary to add colors to its output.
    """
    from io import StringIO
    import sys

    # Capture the output of torchsummary.summary
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    summary(model, input_size, device=device)

    # Retrieve the captured output
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Add colors to the output
    colored_output = ""
    for line in output.splitlines():
        if "Layer (type)" in line:
            colored_output += Fore.YELLOW + line + Style.RESET_ALL + "\n"
        elif "Output Shape" in line:
            colored_output += Fore.CYAN + line + Style.RESET_ALL + "\n"
        elif "Param #" in line:
            colored_output += Fore.MAGENTA + line + Style.RESET_ALL + "\n"
        else:
            colored_output += line + "\n"

    print(colored_output)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_data_loaders(batch_size=128)

    save_dir = "../models"
    os.makedirs(save_dir, exist_ok=True)

    # 训练模式
    # model = SimpleCNN2().to(device)
    # train_model(model, trainloader, testloader, device, num_epochs=30, save_dir=save_dir)

    # 评估测试模式
    # test_model(testloader, model, device)

    # 测试可视化模式
    model = SimpleCNN2().to(device)
    model_path = "../models/CNN2_epoch30.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 彩色 summary
    colored_summary(model, (3, 32, 32), str(device))
    # print(f"模型加载：{model_path}")

    # test_accuracy = test_model(testloader, model, device)

    # visualize_predictions(testloader, model, device, 32)
