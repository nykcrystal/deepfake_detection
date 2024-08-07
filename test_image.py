#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @Contact  : huangsy1314@163.com
# @Website  : https://huangshiyu13.github.io
# @File    : test_image.py

# 测试图片
import sys
from myutils import check_file
from PIL import Image
import numpy as np
import torch
import os
from params import padding_image, img_mean, img_std, resize, DeepFakeModel
from Mytimm.models import create_deepfake_model
import cv2
from PIL import Image, ImageDraw, ImageFont

def test_img(model_path, img_files):
    assert all(check_file(img_file) for img_file in img_files), 'file not exist!'

    use_cuda = True
    use_half = True
    print('To load model from {}'.format(model_path))
    model = create_deepfake_model(
        'efficientnet_deepfake',
        num_classes=2,
        in_chans=12,
        checkpoint_path=model_path,
        strict=False)
    print('Model loaded!')
    model = DeepFakeModel(model)
    if use_cuda:
        model.cuda()
        if use_half:
            model.half()
    model.eval()
    for img_file in img_files:
        img = np.transpose(padding_image(resize(np.array(Image.open(img_file).convert('RGB'), np.uint8))), (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = img.sub_(img_mean).div_(img_std)
        if use_cuda:
            img = img.cuda()
            if use_half:
                img = img.half()
        img = [torch.cat([img, img.clone(), img.clone(), img.clone()], dim=0)]
        with torch.no_grad():
            scores = model(torch.stack(img, dim=0))
        scores = scores.cpu().numpy()[:, 0].tolist()
        #image = Image.open('/root/home/PycharmProjects/deepfake_detection/'+img_file)
        #image.show()
        if scores[0]>0.05:
          print('{}\'s fake score:{}   fake photo!'.format(img_file, scores[0]))
        elif scores[0]<0.05:
          print('{}\'s fake score:{}   real photo!'.format(img_file, scores[0])) 
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #add_text_to_image('/root/home/PycharmProjects/deepfake_detection/'+img_file,'fake score:',(50,50),font,36,'./test_images/output/{}'.format(img_file))


def debug_test_img(model_path, img_files):
    print('To load model from {}'.format('model_path'))
    create_deepfake_model(
        'efficientnet_deepfake',
        num_classes=2,
        in_chans=12,
        strict=False)
    print('Model loaded!')

def add_text_to_image(image_path, text, position, font_path, font_size, output_path):
    # 打开图片
    image = Image.open(image_path)
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    # 指定字体和大小
    font = ImageFont.truetype(font_path, font_size)
    # 添加文字
    draw.text(position, text, font=font, fill="white")
    # 保存修改后的图片
    image.save(output_path)
    # 显示图片
    image.show()
    
def get_image_filenames(folder_path):
    # 获取文件夹中的所有文件和子文件夹
    files = os.listdir(folder_path)
    
    # 过滤出所有图片文件，假设图片文件的扩展名是 jpg, jpeg, png, gif
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif','.jpeg'}
    image_filenames = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
    
    return image_filenames

if __name__ == '__main__':
    model_path = './models/model_half.pth.tar'
    #if len(sys.argv) <= 1:
        #print('Please input your images. e.g. python test_images.py image_path1 image_path2')
        #exit()
    #img_files = sys.argv[1:]
    image_filenames = get_image_filenames('/root/home/PycharmProjects/deepfake_detection/test_images')
    image_list = []
    for image_file_name in image_filenames:
      image_file_name = 'test_images/'+image_file_name
      image_list.append(image_file_name)
    print(image_list)
    test_img(model_path, image_list)
