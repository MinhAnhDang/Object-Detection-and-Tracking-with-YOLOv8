import os
import cv2
from ultralytics import YOLO
import random

detection_threshold = 0.5
video_path = os.path.join('.', 'data', 'people.mp4')
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap_out_path = os.path.join('.', 'out.mp4')
cap_out = cv2.VideoWriter(cap_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))
model = YOLO('yolov8n.pt')
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
while ret:
    results = model(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)
    cap_out.write(frame)
    # cv2.imshow("frame", frame)
    # cv2.waitKey(25)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
