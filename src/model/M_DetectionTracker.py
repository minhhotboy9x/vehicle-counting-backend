from ultralytics import YOLO
import os
import cv2
from model.M_Boundary import Boundary
from model.M_Roi import Roi
from model.utils import transform_points, calculate_distance
import numpy as np
from config import FRAME_WIDTH, FRAME_HEIGHT, JETSON_URL, FPS
import supervision as sv
from supervision.draw.utils import draw_text
from supervision.geometry.core import Point
from supervision.draw.color import Color
from pymongo import UpdateOne
from collections import defaultdict, deque
import warnings
import requests
import base64
import json
import time


class DetectionTracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(frame_rate=FPS)
        self.tracker.reset()
        # coordinate for speed estimation
        self.coordinates = []
        self.bounding_box_annotator = sv.BoxCornerAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(text_padding=5, 
                                            text_position=sv.Position.BOTTOM_CENTER,
                                            color_lookup=sv.ColorLookup.TRACK)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=FPS*2, 
                                    position=sv.Position.BOTTOM_CENTER,
                                    color_lookup=sv.ColorLookup.TRACK)

    def update_model(self, new_model_path):
        self.model = YOLO(new_model_path)

    def reset_track(self):
        self.coordinates = []
        self.tracker.reset()
        self.bounding_box_annotator = sv.BoxCornerAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(text_padding=5, 
                                            text_position=sv.Position.BOTTOM_CENTER,
                                            color_lookup=sv.ColorLookup.TRACK)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=FPS*2, 
                                    position=sv.Position.BOTTOM_CENTER,
                                    color_lookup=sv.ColorLookup.TRACK)

    def count_draw(self, frame, detections):
        line_counter_db_upd = []
        select_detection = np.array([False for _ in detections] )
        detection_poly_points = []
        
        # roi trigger
        for roi_counter in Roi.polygon_counters:
            tmp = roi_counter.trigger(detections).astype(bool)
            tmp_detection = detections[tmp]
            if tmp.shape[0] > 0:
                select_detection |= tmp
                # thứ tự của triggered detectionƠ
                detection_poly_points.append(tmp_detection.get_anchors_coordinates(
                                        anchor=sv.Position.BOTTOM_CENTER))
            else:
                detection_poly_points.append(np.array([]))

        # perspective transform
        for i in range(len(detection_poly_points)):
            detection_poly_points[i] = transform_points(Roi.transformers[i], detection_poly_points[i])
        
        if select_detection.shape[0] > 0:
            detections = detections[select_detection]
        # line_counter
        for line_counter in Boundary.line_counters:
            line_counter.trigger(detections=detections)
            line_counter_db_upd.append(UpdateOne( 
                {'id': line_counter.id},
                {'$set': {
                    'in': line_counter.in_count,
                    'out': line_counter.out_count
                }}))

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        
        # save tracker_id each roi
        for i in range(len(Roi.polygon_counters)):
            self.coordinates.append(defaultdict(lambda: deque(maxlen=FPS)))
            for tracker_id, [x, y] in zip(detections.tracker_id, detection_poly_points[i]):
                self.coordinates[i][tracker_id].append([x, y])
        
        # calculate speed
        for i in range(len(Roi.polygon_counters)):
            for tracker_id in detections.tracker_id:
                if len(self.coordinates[i][tracker_id]) > FPS//2:
                    coordinate_start = self.coordinates[i][tracker_id][-1]
                    coordinate_end = self.coordinates[i][tracker_id][0]
                    distance = calculate_distance(coordinate_start, coordinate_end)
                    time = len(self.coordinates[i][tracker_id]) / FPS
                    speed = distance / time * 3.6
                    k = [index for index, label in enumerate(labels) if label.startswith(f"#{tracker_id}")][0]
                    labels[k] += f" {int(speed)} km/h"
        
        Boundary.update_many(line_counter_db_upd)
        
        annotated_frame = frame.copy()

        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections)
        
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)

        # draw line
        for line_annotator, line_counter in zip(Boundary.line_annotators, Boundary.line_counters):
            annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)

        # draw roi
        for roi_annotator in Roi.polygon_annotators:
            annotated_frame = roi_annotator.annotate(annotated_frame)
        
        return annotated_frame

    def detect_track(self, frame, cam_id):
        start_time = time.time()
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections(xyxy=results.boxes.xyxy.cpu().numpy(),
                                confidence=results.boxes.conf.cpu().numpy(),
                                class_id=results.boxes.cls.cpu().numpy().astype(int))

        detections = detections[np.isin(detections.class_id, [2, 1, 0, 5, 6])]
        detections = self.tracker.update_with_detections(detections=detections)
        end_time = time.time()
        real_fps = int(1/(end_time - start_time))
        frame = draw_text(scene=frame, text=f"FPS: {real_fps}", text_anchor=Point(50, 20), text_padding=5, background_color=Color.WHITE)
        return self.count_draw(frame, detections)

    def get_track(self, frame, results):
        if not results['boxes']:
            empty_boxes = np.empty((0, 4))
            empty_confidence = np.empty(0)
            empty_class_ids = np.empty(0, dtype=int)
            detections = sv.Detections(xyxy=empty_boxes,
                                       confidence=empty_confidence,
                                       class_id=empty_class_ids)
        else:
            detections = sv.Detections(xyxy=np.array(results['boxes']),
                                    confidence=np.array(results['scores']),
                                    class_id=np.array(results['class_ids']).astype(int))

        detections = detections[np.isin(detections.class_id, [2, 1, 0, 5, 6])]
        try:
            detections = self.tracker.update_with_detections(detections=detections)
        except ValueError as e:
            print(detections) 
        return self.count_draw(frame, detections)

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
                frame = self.detect_track(frame, cam_id)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
        
    def get_jetson_frames(self, cam_id):
        start_part = b'--frame\r\nContent-Type: application/json\r\n\r\n'.decode("utf-8")
        end_part = b'endpart\r\n'.decode("utf-8")
        url = JETSON_URL + f'detecting/{cam_id}'
        print(f'-----call cam ID {cam_id}-------')
        with requests.get(url, stream=True) as response:
            if response.status_code == 200:
                start_time = time.time()
                for line in response.iter_lines():
                    if line:
                        json_part = json.loads(line.decode('utf-8').replace('data: ', ''))
                        frame_data = base64.b64decode(json_part['img'])
                        frame_np = np.frombuffer(frame_data, dtype=np.uint8)
                        # Decode the numpy array as an image
                        frame_decode = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        end_time = time.time()
                        real_fps = int(1/(end_time - start_time))
                        start_time = time.time()

                        frame_decode = draw_text(scene=frame_decode, text=f"FPS: {real_fps}", text_anchor=Point(50, 20), text_padding=5, background_color=Color.WHITE)
                        frame_decode = self.get_track(frame_decode, json_part) 
                        ret, frame_buffer = cv2.imencode('.jpg', frame_decode)
                        frame = frame_buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')