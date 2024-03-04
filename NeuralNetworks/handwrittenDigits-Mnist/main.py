import cv2
import numpy as np
from torchvision import transforms
import torch
from model import Net
from matplotlib import pyplot as plt
transform = transforms.Compose([
    transforms.ToTensor(),
])
network = Net()
model_path = './results/sgdmodel.pth'
network_state_dict = torch.load(model_path)
network.load_state_dict(network_state_dict)
hsv_value = np.load('hsv_value.npy')
print(hsv_value)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
kernel = np.ones((5, 5), np.int8)
noise_thresh = 800
show_labels=False
while True:
    _, frame = cap.read()
    canvas = np.zeros_like(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = hsv_value[0]
    upper_range = hsv_value[1]
    mask = cv2.inRange(frame, lower_range, upper_range)
    cv2.imshow("",mask)





    
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > noise_thresh:
            x1, y1, w, h = cv2.boundingRect(contour)
            padding = 10
            x1 -= padding
            y1 -= padding
            w += 2 * padding
            h += 2 * padding
            x1 = max(0, x1)
            y1 = max(0, y1)
            canvas = cv2.rectangle(canvas, (x1, y1), (x1 + w, y1 + h), [0, 255, 0], 4)
            roi = frame[y1:y1 + h, x1:x1 + w]
            gray_roi = cv2.inRange(roi, lower_range, upper_range)
            resized_roi = cv2.resize(gray_roi, (28, 28))
            tensor = transform(resized_roi)
            resized_tensor = torch.Tensor.view(tensor, (1, 1, 28, 28))
            if(show_labels):
                network.eval()
                with torch.no_grad():
                    output = network(resized_tensor)
                    pred = output.argmax(dim=1, keepdim=True)
                    pred = str(int(pred))
                cv2.putText(canvas, pred, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 255, 255], 2)
    combined = cv2.add(canvas, frame)
    stacked = np.hstack((frame, combined))

    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))
    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite('./digits/digit.png', stacked)
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        show_labels = not show_labels
cv2.destroyAllWindows()
cap.release()

