from flask import Blueprint, Response, request, redirect, url_for, jsonify

from model.M_DetectionTracker import DetectionTracker


streaming_bp = Blueprint('streaming', __name__)

det_tracker = DetectionTracker('models/yolov8n.pt')

@streaming_bp.route('/change_cam', methods=['POST'])
def change_cam():
    data = request.get_json()
    new_cam_id = int(data['cam_id'])
    det_tracker.change_cam(new_cam_id)
    video_feed_url = det_tracker.get_video_feed_url(new_cam_id)
    return jsonify({'videoFeedUrl': video_feed_url})

@streaming_bp.route('/streaming/<int:cam_id>')
def stream_cam(cam_id):
    return Response(det_tracker.generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')