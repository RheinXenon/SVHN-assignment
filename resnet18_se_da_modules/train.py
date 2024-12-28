import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

def train_model(model,trainloader, testloader, device, num_epochs=30, save_dir='./models'):
    writer = SummaryWriter(log_dir='./logs')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        y_true_train = []
        y_pred_train = []

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

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/resnet18_se_da_epoch{epoch+1}.pth")
            print(f"模型已保存到 {save_dir}/resnet18_se_da_epoch{epoch+1}.pth")
    writer.close()
