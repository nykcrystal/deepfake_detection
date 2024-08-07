# -*- coding: utf-8 -*-
from flask import Flask, request,request,render_template
from flask import jsonify
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

model = timm.create_model('xception', pretrained=False)
# 加载本地预训练模型
model.load_state_dict(torch.load('xception_weights.pth'))
# 打印模型结构，查找最后一层
# print("Model structure:\n", model)

# 获取最后一层的输入特征数
num_features = model.get_classifier().in_features

# 修改模型的最后一层以适应二分类任务
model.fc = nn.Linear(num_features, 1)  # 二分类任务：1 输出神经元
model.sigmoid = nn.Sigmoid()  # 添加 Sigmoid 激活函数

model.eval()  # 将模型设置为评估模式

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image



def get_image_filenames(folder_path):
    # 获取文件夹中的所有文件和子文件夹
    files = os.listdir(folder_path)

    # 过滤出所有图片文件，假设图片文件的扩展名是 jpg, jpeg, png, gif
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    image_filenames = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

    return image_filenames


app = Flask(__name__)

@app.route('/upload')
def index():
    return render_template('index.html')
    
@app.route('/upload', methods=['POST'])
def run_detect():
    '''
    image_filenames = get_image_filenames('/root/home/PycharmProjects/deepfake_detection/test_images')
    image_list = []
    image_test_result = {}

    for image_file_name in image_filenames:
        image_file_name = 'test_images/' + image_file_name
        image_list.append(image_file_name)
    ''' 
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"errro": "No selected file"})
    if file and file.filename.lower().endswith(('png','jpg','jpeg','gif')):
        try:
            # 将文件保存到服务器
            #filepath = os.path.join(app.config['/root/home/PycharmProjects/deepfake_detection/'],file.filename)
            #file.save(filepath)

            image = Image.open(file).convert('RGB')
            image = preprocess(image)
            image = image.unsqueeze(0)

            with torch.no_grad():
                output = model(image)

            probability = model.sigmoid(output).item()
            # print('Probability:', probability)
            threshold = 0.5
            if probability > threshold:
                 return jsonify({"probablity": '{:.2%}'.format(1 - abs(probability - 0.5) / 0.1)},{"msg":'AI-generated or deepfaked'})
            else:
                return jsonify({"probablity": '{:.2%}'.format(file.filename, 1 - abs(probability - 0.5) / 0.1)},{"msg":"real"})

        except Exception as e:
            return jsonify({"erro":"File is not a image file"})


app.run(host="0.0.0.0", port=6004, debug=True)

