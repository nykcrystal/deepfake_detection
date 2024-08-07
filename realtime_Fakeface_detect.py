# -*- coding: utf-8 -*-
import cv2
import face_recognition

# ����Ƶ�ļ�
video_path = '/root/home/PycharmProjects/deepfake_detection/video/input.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
previous_faces_encodings = []  # ǰһ������

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ��⵱ǰ֡�е�����
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # ����Ƿ�������
    face_changed = False
    if previous_faces_encodings:
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(previous_faces_encodings, face_encoding)  # ����face_recogniton����ǰһ����������������������û�б仯
            if not any(matches):  # ƥ���������û���κ�������
                face_changed = True
                break

    # �����⵽��������֡�ϱ��
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if face_changed:
            color = (0, 0, 255)  # Red for face changed
            label = "Face Changed"
        else:
            color = (0, 255, 0)  # Green for no change
            label = "No Change"

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # ��ʾ��ǰ֡
    cv2.imshow('Face Swap Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_faces_encodings = face_encodings
    frame_count += 1

# �ͷ���Ƶ�ļ����ر����д���
cap.release()
cv2.destroyAllWindows()
