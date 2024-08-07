# -*- coding: utf-8 -*-
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json
from flask import Flask, jsonify
import cv2

# 加载预训练的 Xception 模型
model = timm.create_model('xception', pretrained=True)
torch.save(model.state_dict(),'xception_weights.pth')
model.load_state_dict(torch.load('xception_weights.pth'))
# 打印模型结构，查找最后一层
#print("Model structure:\n", model)

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

def getImg_from_video():
    # 加载预训练的人脸检测 Haar Cascade 分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # 打开视频文件
    video_path = '/root/home/PycharmProjects/deepfake_detection/video/input.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # 创建用于保存截取人脸的目录
    output_dir = '/root/home/PycharmProjects/deepfake_detection/test_images'
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    # 将帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 处理每个检测到的人脸
        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y + h, x:x + w]
        # 保存人脸图像到本地
            face_filename = os.path.join(output_dir, f'frame_{frame_count}_face_{i}.jpg')
            cv2.imwrite(face_filename, face_img)
        frame_count += 1
        if frame_count==2:
            break
    # 释放视频文件并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
  
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 确保图像是 RGB 模式
    image = preprocess(image)
    image = image.unsqueeze(0)  # 增加一个批次维度
    return image

def get_image_filenames(folder_path):
    # 获取文件夹中的所有文件和子文件夹
    files = os.listdir(folder_path)
    
    # 过滤出所有图片文件，假设图片文件的扩展名是 jpg, jpeg, png, gif
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif','.jpeg'}
    image_filenames = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
    
    return image_filenames
    
# 加载并预处理图像
getImg_from_video()
image_filenames = get_image_filenames('/root/home/PycharmProjects/deepfake_detection/test_images')
image_list = []
image_test_result = {}
i = 1
for image_file_name in image_filenames:
    image_file_name = 'test_images/'+image_file_name
    image_list.append(image_file_name)

for image_name in image_list:
    image = load_and_preprocess_image('/root/home/PycharmProjects/deepfake_detection/'+image_name)
# 进行预测
    with torch.no_grad():
      output = model(image)

# 将输出转换为概率
    probability = model.sigmoid(output).item()

#print('Probability:', probability)
    threshold = 0.5
    if probability > threshold:
      print('{}   The image is  {:.2%}  likely AI-generated or deepfaked.'.format(image_name,1 - abs(probability - 0.5)))
      image_test_result.update({f"msg{i}":'{}   The image is  {:.2%}  likely AI-generated or deepfaked.'.format(image_name, 1 - abs(probability - 0.5))})
      i = i+1
    else:
      print('{}   The image is {:.2%}  likely real.'.format(image_name,1 - abs(probability - 0.5)))
      image_test_result.update({f"msg{i}":'{}   The image is {:.2%}  likely real.'.format(image_name, 1 - abs(probability - 0.5))})
      i = i+1
json_data = json.dumps(image_test_result)
print(json_data)


