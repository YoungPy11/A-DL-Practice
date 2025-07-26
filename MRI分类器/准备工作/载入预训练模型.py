import torch
import torchvision
import transformers
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from transformers import ViTForImageClassification, ViTImageProcessor
import os

# 检查环境
print(f"PyTorch 版本: {torch.__version__}")
print(f"torchvision 版本: {torchvision.__version__}")
print(f"transformers 版本: {transformers.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 测试 ResNet18
resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
print(f"ResNet18 参数数量: {sum(p.numel() for p in resnet18.parameters()):,}")

# 测试 DenseNet121
densenet121 = models.densenet121(pretrained=True)
print(f"DenseNet121 参数数量: {sum(p.numel() for p in densenet121.parameters()):,}")

# 测试 ViT-base (尝试多种方法)
print("\n尝试加载 ViT-base 模型...")

# 方法 1: 使用国内镜像
try:
    print("\n方法 1: 使用国内镜像...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    print(f"ViT-base 模型加载成功!")
except Exception as e1:
    print(f"方法 1 失败: {e1}")
    
    # 方法 2: 从本地加载 (需要手动下载)
    try:
        print("\n方法 2: 尝试从本地加载...")
        local_model_path = 'C:/path/to/saved/model'  # 替换为实际路径
        vit_model = ViTForImageClassification.from_pretrained(local_model_path)
        print(f"ViT-base 模型从本地加载成功!")
    except Exception as e2:
        print(f"方法 2 失败: {e2}")
        print("\n请尝试以下操作:")
        print("1. 检查网络连接是否正常")
        print("2. 使用其他设备下载模型并复制到本地")
        print("3. 在命令行中运行: 'ping huggingface.co' 检查网络连通性")