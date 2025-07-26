import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torchvision import models, transforms
import time
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import random

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 数据路径配置
data_dir = 'E:/人工智能基础A/hw2/完整数据集/1512427/完整jpg'
pt_data_dir = 'E:/人工智能基础A/hw2/转换后_torch张量'

# 图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PTDataset(data.Dataset):
    def __init__(self, samples, classes, class_to_idx, transform=None):
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        tensor = torch.load(file_path)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


def prepare_datasets():
    split_dir = os.path.join(pt_data_dir, 'split')
    train_samples = []
    train_path = os.path.join(split_dir, 'train')
    classes = sorted(os.listdir(train_path))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(train_path, cls)
        for pt_file in os.listdir(cls_dir):
            if pt_file.endswith('.pt'):
                train_samples.append((
                    os.path.join(cls_dir, pt_file),
                    class_to_idx[cls]
                ))

    test_samples = []
    test_path = os.path.join(split_dir, 'test')
    for cls in classes:
        cls_dir = os.path.join(test_path, cls)
        for pt_file in os.listdir(cls_dir):
            if pt_file.endswith('.pt'):
                test_samples.append((
                    os.path.join(cls_dir, pt_file),
                    class_to_idx[cls]
                ))

    print(f"训练集大小: {len(train_samples)}, 测试集大小: {len(test_samples)}")

    train_dataset = PTDataset(train_samples, classes, class_to_idx)
    test_dataset = PTDataset(test_samples, classes, class_to_idx)

    return train_dataset, test_dataset, classes, class_to_idx


# 创建DenseNet121模型的函数
def create_model(num_classes):
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    log_data = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        log_data.append({
            'epoch': epoch,
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'train_acc': train_accs[-1].item(),
            'val_acc': val_accs[-1].item()
        })

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    log_df = pd.DataFrame(log_data)
    log_df.to_csv('training_log.csv', index=False)

    return model, best_acc, train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, dataloader, device, classes):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    n_classes = len(classes)
    y_test_bin = label_binarize(y_true, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, precision, recall, f1, conf_matrix


def plot_confusion_matrix(conf_matrix, classes, title='Confusion matrix'):
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def convert_jpg_to_pt():
    print("开始将JPG图像转换为PT文件...")
    total_images = 0
    split_path = os.path.join(data_dir, 'split')
    pt_split_path = os.path.join(pt_data_dir, 'split')
    os.makedirs(pt_split_path, exist_ok=True)

    for dataset_type in ['train', 'test']:
        dataset_path = os.path.join(split_path, dataset_type)
        pt_dataset_path = os.path.join(pt_split_path, dataset_type)
        os.makedirs(pt_dataset_path, exist_ok=True)

        for cls in os.listdir(dataset_path):
            cls_path = os.path.join(dataset_path, cls)
            pt_cls_path = os.path.join(pt_dataset_path, cls)
            os.makedirs(pt_cls_path, exist_ok=True)

            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.jpg', '.jpeg')):
                    img_path = os.path.join(cls_path, img_name)
                    pt_name = os.path.splitext(img_name)[0] + '.pt'
                    pt_path = os.path.join(pt_cls_path, pt_name)

                    if not os.path.exists(pt_path):
                        try:
                            img = Image.open(img_path).convert('RGB')
                            tensor = transform(img)
                            torch.save(tensor, pt_path)
                            total_images += 1
                        except Exception as e:
                            print(f"无法处理图像 {img_path}: {e}")

    print(f"完成转换，共处理 {total_images} 张图像")


def perform_random_search(train_dataset, test_dataset, classes, num_classes):
    # 定义参数搜索空间
    optimizers = [optim.SGD, optim.Adam, optim.AdamW]
    lrs = [0.009, 0.006, 0.003]
    best_val_acc = 0.0
    best_optimizer = None
    best_lr = None
    best_model = None
    best_train_losses = []
    best_val_losses = []
    best_train_accs = []
    best_val_accs = []

    #搜索9次（3个优化器 × 3个学习率）
    for i in range(len(optimizers)):
        for j in range(len(lrs)):
            optimizer_choice = optimizers[i]
            lr = lrs[j]
            print(f"\n第 {i*3 + j + 1} 次尝试")
            print(f"选择优化器: {optimizer_choice.__name__}, 学习率: {lr}")

            # 创建模型和优化器
            model = create_model(num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optimizer_choice(model.parameters(), lr=lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            # 数据加载器
            train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
            dataloaders = {'train': train_loader, 'val': test_loader}

            # 训练模型
            model, val_acc, train_losses, val_losses, train_accs, val_accs = train_model(
                model, criterion, optimizer, scheduler, dataloaders, num_epochs=10)

            # 更新最佳参数
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_optimizer = optimizer_choice
                best_lr = lr
                best_model = model
                best_train_losses = train_losses
                best_val_losses = val_losses
                best_train_accs = train_accs
                best_val_accs = val_accs

    return best_model, best_val_acc, best_optimizer, best_lr, best_train_losses, best_val_losses, best_train_accs, best_val_accs


if __name__ == '__main__':
    # 转换JPG到PT文件
    convert_jpg_to_pt()

    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 准备数据集
    train_dataset, test_dataset, classes, class_to_idx = prepare_datasets()
    num_classes = len(classes)

    # 定义数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    dataloaders = {'train': train_loader, 'val': test_loader}

    # 执行随机搜索
    best_model, best_val_acc, best_optimizer, best_lr, best_train_losses, best_val_losses, best_train_accs, best_val_accs = perform_random_search(
        train_dataset, test_dataset, classes, num_classes)

    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    print(f"最佳优化器: {best_optimizer.__name__}")
    print(f"最佳学习率: {best_lr}")

    # 评估模型
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(best_model, test_loader, device, classes)

    print(f"\n评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(conf_matrix, classes, title='Confusion Matrix')

    # 绘制训练和验证的损失/准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(best_train_losses, label='Train Loss')
    plt.plot(best_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(best_train_accs, label='Train Accuracy')
    plt.plot(best_val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 保存模型
    torch.save(best_model.state_dict(), 'best_model.pth')