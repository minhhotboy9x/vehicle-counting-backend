from flask import Blueprint, Response, request, redirect, url_for, jsonify

from model.M_DetectionTracker import DetectionTracker


streaming_bp = Blueprint('streaming', __name__)

det_tracker = DetectionTracker('models/yolov8s.pt')


@streaming_bp.route('/streaming/<int:cam_id>')
def stream_cam(cam_id):
    return Response(det_tracker.generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')