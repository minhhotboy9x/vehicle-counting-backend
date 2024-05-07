from flask import Blueprint, Response, request, redirect, url_for, jsonify

from model.M_DetectionTracker import DetectionTracker


streaming_bp = Blueprint('streaming', __name__)

det_tracker = DetectionTracker('models/yolov8n.pt')

@streaming_bp.route('/update_model')
def update_model():
    model = request.args['model']
    new_model_path = f'./models/{model}'
    det_tracker.update_model(new_model_path)
    return "Model updated successfully!"

@streaming_bp.route('/streaming/<int:cam_id>')
def stream_cam(cam_id):
    # return Response(det_tracker.generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(det_tracker.get_jetson_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')