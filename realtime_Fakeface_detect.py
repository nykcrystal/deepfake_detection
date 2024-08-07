# -*- coding: utf-8 -*-
import cv2
import face_recognition

# 打开视频文件
video_path = '/root/home/PycharmProjects/deepfake_detection/video/input.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
previous_faces_encodings = []  # 前一张人脸

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测当前帧中的人脸
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # 检测是否发生换脸
    face_changed = False
    if previous_faces_encodings:
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(previous_faces_encodings, face_encoding)  # 调用face_recogniton库检测前一张人脸和现在这张人脸有没有变化
            if not any(matches):  # 匹配的特征点没有任何相似性
                face_changed = True
                break

    # 如果检测到换脸，在帧上标记
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if face_changed:
            color = (0, 0, 255)  # Red for face changed
            label = "Face Changed"
        else:
            color = (0, 255, 0)  # Green for no change
            label = "No Change"

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 显示当前帧
    cv2.imshow('Face Swap Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_faces_encodings = face_encodings
    frame_count += 1

# 释放视频文件并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
