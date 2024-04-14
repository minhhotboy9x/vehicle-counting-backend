from ultralytics import YOLO
import os
import cv2
import os
import torch
from random import randint
from config import FRAME_WIDTH, FRAME_HEIGHT
from model.sort import *
import supervision as sv
from collections import defaultdict, deque

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        box_center = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)
        if categories is not None:
            label += f'({int(categories[i])})'  # Thêm category vào label nếu tồn tại
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 253), 1)
        cv2.rectangle(img, (x1, y1 - 13), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.3, [255, 255, 255], 1)

def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)

class DetectionTracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(frame_rate=25)

    def update_model(self, new_model_path):
        self.model = YOLO(new_model_path)

    def reset_track(self):
        KalmanBoxTracker.count = 0

    # def detect_track(self, frame):
    #     r = self.model(frame, verbose=False, device=0)[0]
    #     box = r.boxes.xyxy
    #     conf = r.boxes.conf.unsqueeze(1)
    #     cls = r.boxes.cls.unsqueeze(1)
    #     dets = torch.cat((box, conf, cls), dim=1).cpu().numpy()
    #     tracked_dets = self.tracker.update(dets)
    #     if len(tracked_dets) > 0:
    #         bbox_xyxy = tracked_dets[:,:4]
    #         identities = tracked_dets[:, 8]
    #         categories = tracked_dets[:, 4]
    #         draw_boxes(frame, bbox_xyxy, identities, categories)
    #     return frame
    


    def detect_track(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = sv.Detections(xyxy=results.boxes.xyxy.cpu().numpy(),
                                confidence=results.boxes.conf.cpu().numpy(),
                                class_id=results.boxes.cls.cpu().numpy().astype(int))

        detections = detections[np.isin(detections.class_id, [2, 1, 0])]

        detections = self.tracker.update_with_detections(detections=detections)
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_padding=5)
        points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER)
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(), detections=detections)
        
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame

    # Function to generate frames from video
    def generate_frames(self, cam_id):
        self.reset_track()

        video_path = f'./imgs/{cam_id}.mp4'
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(video_path)
            else:
                # Perform object detection
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame = self.detect_track(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    