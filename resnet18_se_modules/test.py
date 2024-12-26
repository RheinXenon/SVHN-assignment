import torch
from tqdm import tqdm

def test_model(testloader, model, device):
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
