import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(data_dir, test_size=0.2, random_state=42):
    # 创建分割目录结构
    split_dir = os.path.join(data_dir, 'split')
    train_dir = os.path.join(split_dir, 'train')
    test_dir = os.path.join(split_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 在分割目录中创建类别子目录
    for cls in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, cls)) and not cls.startswith('split'):
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # 对每个类别进行分层划分
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path) or cls.startswith('split'):
            continue

        # 获取类别所有文件
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        files = np.array(files)

        # 划分数据
        train_files, test_files = train_test_split(
            files, test_size=test_size, random_state=random_state
        )

        # 复制文件到对应的分割目录
        # 复制训练集
        for f in train_files:
            src = os.path.join(cls_path, f)
            dst = os.path.join(train_dir, cls, f)
            shutil.copy(src, dst)
        # 复制测试集
        for f in test_files:
            src = os.path.join(cls_path, f)
            dst = os.path.join(test_dir, cls, f)
            shutil.copy(src, dst)


if __name__ == '__main__':
    data_dir = 'E:/人工智能基础A/hw2/完整数据集/1512427/完整jpg'  # 原始数据路径
    split_dataset(data_dir)
    print("数据集划分完成！")