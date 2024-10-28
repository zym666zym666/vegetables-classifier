import torch
from torchvision import transforms, models
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt

# 使用内嵌绘图
%matplotlib inline

# 加载模型和类别索引
def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 设置为评估模式
    return model

def load_class_indices(json_path):
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    return class_indices

# 预测函数
def predict(image, model, class_indices, device):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # 将图像张量移动到 GPU

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # 计算每个类别的概率
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    predicted_class_name = class_indices[str(predicted_class)]
    return predicted_class_name, probabilities.squeeze().cpu().numpy()  # 将概率移动回 CPU

# 主程序
def main(image_files):
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型和类别映射
    model_path = 'best_model.pth'
    json_path = 'class_indices.json'
    
    class_indices = load_class_indices(json_path)
    num_classes = len(class_indices)
    model = load_model(model_path, num_classes)
    model.to(device)  # 将模型移动到 GPU

    # 处理每张图片并绘制概率图
    for image_file in image_files:
        image_path = f'./{image_file}'  # 确保路径正确
        image = Image.open(image_path).convert('RGB')
        
        # 预测
        predicted_class, probabilities = predict(image, model, class_indices, device)
        
        # 输出预测类别和概率
        print(f"Predicted Class for {image_file}: {predicted_class}")
        print(f"Probabilities: {probabilities}")

        # 将概率转换为 DataFrame
        prob_df = pd.DataFrame(probabilities, index=list(class_indices.values()), columns=["Probability"])

        # 画图
        fig, ax = plt.subplots(figsize=(12, 6))  # 创建子图
        prob_df.sort_values(by="Probability", ascending=False).plot(kind='bar', ax=ax)  # 使用 ax 参数指定坐标轴
        ax.set_title(f"Prediction Probabilities for {image_file}", fontsize=16)
        ax.set_xlabel("Predicted Class", fontsize=12, labelpad=15)
        ax.set_ylabel("Probability", fontsize=12, labelpad=15)
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)  # 调整布局
        plt.show()  # 在本地显示图像

if __name__ == "__main__":
    # 替换为你想处理的图像文件名列表
    main(['carrot.jpg', 'onion.jpg'])  # 处理多张图片，例如 'carrot.jpg' 和 'onion.jpg'
