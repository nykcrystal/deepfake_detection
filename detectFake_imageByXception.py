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

# ����Ԥѵ���� Xception ģ��
model = timm.create_model('xception', pretrained=True)
torch.save(model.state_dict(),'xception_weights.pth')
model.load_state_dict(torch.load('xception_weights.pth'))
# ��ӡģ�ͽṹ���������һ��
#print("Model structure:\n", model)

# ��ȡ���һ�������������
num_features = model.get_classifier().in_features

# �޸�ģ�͵����һ������Ӧ����������
model.fc = nn.Linear(num_features, 1)  # ����������1 �����Ԫ
model.sigmoid = nn.Sigmoid()  # ��� Sigmoid �����

model.eval()  # ��ģ������Ϊ����ģʽ

# ͼ��Ԥ����
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def getImg_from_video():
    # ����Ԥѵ����������� Haar Cascade ������
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # ����Ƶ�ļ�
    video_path = '/root/home/PycharmProjects/deepfake_detection/video/input.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # �������ڱ����ȡ������Ŀ¼
    output_dir = '/root/home/PycharmProjects/deepfake_detection/test_images'
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    # ��֡ת��Ϊ�Ҷ�ͼ��
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # �������
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # ����ÿ����⵽������
        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y + h, x:x + w]
        # ��������ͼ�񵽱���
            face_filename = os.path.join(output_dir, f'frame_{frame_count}_face_{i}.jpg')
            cv2.imwrite(face_filename, face_img)
        frame_count += 1
        if frame_count==2:
            break
    # �ͷ���Ƶ�ļ����ر����д���
    cap.release()
    cv2.destroyAllWindows()
  
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # ȷ��ͼ���� RGB ģʽ
    image = preprocess(image)
    image = image.unsqueeze(0)  # ����һ������ά��
    return image

def get_image_filenames(folder_path):
    # ��ȡ�ļ����е������ļ������ļ���
    files = os.listdir(folder_path)
    
    # ���˳�����ͼƬ�ļ�������ͼƬ�ļ�����չ���� jpg, jpeg, png, gif
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif','.jpeg'}
    image_filenames = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
    
    return image_filenames
    
# ���ز�Ԥ����ͼ��
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
# ����Ԥ��
    with torch.no_grad():
      output = model(image)

# �����ת��Ϊ����
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


