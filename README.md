# Object-Detection-Yolov8

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# Using YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure this path is correct and the weights file is not corrupted

# Address of a video
cap = cv2.VideoCapture(r'C:\Users\Anjani\PycharmProjects\pythonProject4\video.mp4\video.mp4')  # Replace with the path to your video file


cv2.namedWindow('YOLOv8 Object Detection & Tracking', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('YOLOv8 Object Detection & Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


detected_objects = deque(maxlen=100)

# Defining drawing boxes
def draw_boxes(results, frame):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break


    # Get predictions
    results = model(frame)

    # Draw boxes on the frame
    draw_boxes(results, frame)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Object Detection & Tracking', frame)

    # Press q for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
