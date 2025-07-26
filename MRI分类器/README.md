# MRI 脑肿瘤分类器

## 项目概述

本项目基于预训练的DenseNet121和ResNet18模型，通过迁移学习技术实现MRI脑肿瘤图像的三分类任务。项目包含完整的模型训练、评估流程，并提供了基于Gradio的Web界面用于交互式预测。

## 功能特性

- 支持两种预训练模型架构：DenseNet121和ResNet18
- 完整的训练流程，包括数据预处理、模型微调、性能评估
- 自动化的超参数搜索功能
- 详细的训练过程记录和可视化
- 提供Web界面进行交互式预测

## 数据集

本项目使用MRI脑肿瘤分类数据集，包含三类肿瘤：
- 脑膜瘤 (Meningioma)
- 神经胶质瘤 (Glioma)
- 垂体瘤 (Pituitary Tumor)

数据集已预先划分为训练集和测试集。

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- scikit-learn
- pandas
- matplotlib
- Gradio (用于Web界面)
- PIL (Python Imaging Library)

## 安装指南

1. 克隆本仓库：
   ```
   git clone https://github.com/yourusername/brain-tumor-classifier.git
   cd brain-tumor-classifier
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用说明

### 训练模型

1. 准备数据集：
   - 将MRI图像按类别存放在`data/split/train`和`data/split/test`目录下
   - 运行`convert_jpg_to_pt()`函数将图像转换为PyTorch张量格式

2. 运行主脚本：
   ```
   python main.py
   ```

3. 训练过程将自动：
   - 执行超参数搜索
   - 训练最佳模型
   - 保存模型权重
   - 生成训练日志和评估结果

### 使用Web界面

1. 启动Gradio应用：
   ```
   python app.py
   ```

2. 在浏览器中打开显示的URL

3. 使用界面：
   - 选择模型类型 (ResNet18或DenseNet121)
   - 上传MRI图像
   - 点击"预测"按钮查看结果

## 性能指标

在测试集上的表现：

| 模型       | 准确率 | 精确率 | 召回率 | F1分数 |
|------------|--------|--------|--------|--------|
| DenseNet121 | 0.94   | 0.93   | 0.94   | 0.93   |
| ResNet18    | 0.92   | 0.91   | 0.92   | 0.91   |

## 贡献指南

欢迎贡献！请提交Pull Request或创建Issue。

## 许可证

本项目采用MIT许可证。