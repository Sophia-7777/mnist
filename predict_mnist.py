# predict.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# ------------------------------
# 1. 定义神经网络（MLP）
# ------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# ------------------------------
# 2. 加载训练好的模型
# ------------------------------
model = MLP()
model_path = "mnist_mlp.pth"  # 假设你已经保存过模型
if not os.path.exists(model_path):
    print(f"Error: 模型文件 {model_path} 不存在，请先训练并保存模型")
    sys.exit(1)

# 加载模型参数，并设置为评估模式
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ------------------------------
# 3. 图像预处理
# ------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保是单通道
    transforms.Resize((28, 28)),                  # 调整为28x28
    transforms.ToTensor(),
])

# ------------------------------
# 4. 预测函数（带置信度）
# ------------------------------
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    with torch.no_grad():
        output = model(image)               # 输出 logits
        probs = torch.softmax(output, dim=1)  # 转成概率
        confidence, predicted = torch.max(probs, dim=1)  # 最大概率及对应类别
    return predicted.item(), confidence.item()

# ------------------------------
# 5. 命令行使用
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: 图片文件 {img_path} 不存在")
        sys.exit(1)

    predicted, confidence = predict(img_path)
    print(f"预测结果: {predicted}, 置信度: {confidence*100:.2f}%")
