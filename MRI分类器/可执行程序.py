import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 定义模型创建函数
def create_resnet18(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_densenet121(num_classes):
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


# 加载训练好的模型权重
def load_model(model_name, num_classes):
    model = None
    if model_name == "ResNet18":
        model = create_resnet18(num_classes)
        model_path = r"./Resnet18/best_model_resnet18.pth"  # 使用正确路径
    elif model_name == "DenseNet121":
        model = create_densenet121(num_classes)
        model_path = r"./DenseNet121/网格3d.pth"  # 使用正确路径
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# 图像预处理
def preprocess_image(image):
    # 将 NumPy 数组转换为 PIL 图像
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# 预测函数
def predict_image(model_name, image):
    num_classes = 3  # 假设是三分类问题
    model = load_model(model_name, num_classes)

    processed_image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    class_labels = ["脑膜瘤", "神经胶质瘤", "垂体瘤"]
    return class_labels[predicted_class]


# 创建 Gradio 界面
with gr.Blocks(title="MRI 图像分类") as demo:
    gr.Markdown("## MRI 图像分类器")

    with gr.Tab("图像预测"):
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    choices=["ResNet18", "DenseNet121"],
                    label="选择模型",
                    value="ResNet18"
                )
                image_input = gr.Image(label="上传 MRI 图像")
                predict_btn = gr.Button("预测", variant="primary")
            with gr.Column(scale=1):
                result_output = gr.Label(label="预测结果")

        predict_btn.click(
            fn=predict_image,
            inputs=[model_choice, image_input],
            outputs=result_output
        )

    gr.Markdown("### 使用说明")
    gr.Markdown("""
    1. 使用此界面，您可以选择已训练好的 ResNet18 或 DenseNet121 模型。
    2. 上传一张 MRI 图像。
    3. 点击"预测"按钮，查看预测结果。
    """)

# 启动应用
if __name__ == "__main__":
    demo.launch()