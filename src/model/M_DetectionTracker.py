from ultralytics import YOLO
import os
import cv2

class DetectionTracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    def change_cam(self, new_cam_id):
        pass

    # Function to generate frames from video
    def generate_frames(self, cam_id):
        video_path = f'./imgs/{cam_id}.mp4'
        if os.path.exists(video_path):
            print("Đường dẫn video hợp lệ.")
        else:
            print("Đường dẫn video không tồn tại hoặc không hợp lệ.")
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(video_path)
            else:
                # Perform object detection
                results = self.model(frame)
                frame = results[0].plot()  # Render detected objects on the frame

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()