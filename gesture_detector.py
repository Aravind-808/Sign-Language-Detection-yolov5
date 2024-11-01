import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

videocamera = cv2.VideoCapture(0)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rajes\Project_1\yolov5\runs\train\exp16\weights\best.pt', force_reload=True)
print(model.names)

def detect_gesture(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)
    predicted = []

    for *xyxy, conf, cl in results.xyxy[0]:
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, xyxy)
            predicted.append((x1, y1, x2, y2, conf.item(), int(cl.item())))

    return predicted

while videocamera.isOpened():
    ret, frame = videocamera.read()
    if not ret:
        print("Frame not read")
        break
    frame = cv2.flip(frame,1)
    frame_resized = cv2.resize(frame, (640, 640))

    predicted = detect_gesture(frame_resized)

    for (x1, y1, x2, y2, conf, cl) in predicted:
        label = f'Alphabet detected: {model.names[cl]}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

videocamera.release()
cv2.destroyAllWindows()
