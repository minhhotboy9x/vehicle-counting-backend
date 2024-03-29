from ultralytics import YOLO
import cv2
import time

model = YOLO('models/v8n_relu_DETRAC2_quantized.torchscript')


video_path = 'imgs/MVI_39211.mp4'  # Đường dẫn của video của bạn
cap = cv2.VideoCapture(video_path)

sum_frame = 0

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or sum_frame==500:
        cap.release()
        break
    else:
        sum_frame += 1
        # Convert frame to byte stream
        result = model(frame, device='cpu')
        # frame = result[0].plot()
        print(sum_frame )

end_time = time.time()

inference_time = (end_time - start_time)
print(sum_frame/inference_time)
