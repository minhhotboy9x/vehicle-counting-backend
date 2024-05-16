from flask import Blueprint, request, jsonify
from model.M_Boundary import Boundary
from model.M_Roi import Roi

roiboundary_bp = Blueprint('roi_boundary', __name__)

def init_roiboundary_bp(mongo):
    Boundary.init_mongo(mongo)
    Roi.init_mongo(mongo)

    # Boundary endpoint

    @roiboundary_bp.route('/testboundary', methods=['POST'])
    def testboundary():
        data = request.json
        return jsonify({"message": "Received data successfully", "data": data})

    @roiboundary_bp.route('/insert_boundary', methods=['POST'])
    def insert_boundary():
        # Lấy dữ liệu từ request
        data = request.json
        # Kiểm tra nếu id đã được cung cấp trong dữ liệu
        if data.get('id') is None:
            return jsonify({'error': 'Missing id in request data'}), 400
        Boundary.insert(data)
        return jsonify({'message': 'Insert or update successful'}), 200
    
    @roiboundary_bp.route('/update_boundary', methods=['POST'])
    def update_boundary():
        data = request.json
        id = data.get('id')
        if id is None:
            return jsonify({'error': 'Missing id in request data'}), 400
        update_data = {}
        for key in data:
            if key != 'id':
                update_data[key] = data[key]

        if Boundary.update(id, **update_data):
            return jsonify({'message': 'Update successful'}), 200
        else:
            return jsonify({'error': 'Failed to update data'}), 400
    
    def get_boundary_and_counter(query):
        # Thực hiện truy vấn
        results = Boundary.find(query)
        # Tạo một danh sách dưới dạng từ điển
        boundaries_list = [{'id': boundary['id'], 
                    'camId': boundary['camId'], 
                    'in': boundary.get('in') if boundary.get('in') is not None else 0,
                    'out': boundary.get('out') if boundary.get('out') is not None else 0,
                    'pointL': boundary['pointL'], 
                    'pointR': boundary['pointR'], 
                    'pointDirect': boundary['pointDirect']} 
                    for boundary in results]
        Boundary.get_line_annotators(boundaries_list)
        return boundaries_list

    @roiboundary_bp.route('/delete_boundary', methods=['POST'])
    def delete_boundary():
        # Lấy dữ liệu từ request
        data = request.json
        id = data.get('id')
        if id is None:
            return jsonify({'error': 'Missing id in request data'}), 400
        query = {'id': id}
        # Thực hiện xóa
        result = Boundary.delete(query)
        if result.acknowledged:
            camId = data.get('camId')
            get_boundary_and_counter({'camId': camId})
            return jsonify({'message': 'Delete successful'}), 200
        else:
            return jsonify({'error': 'Failed to delete data'}), 400
    

    @roiboundary_bp.route('/get_boundaries', methods=['POST'])
    def get_boundaries():
        data = request.json
        query = {key: value for key, value in data.items() if value is not None}
        boundaries_list = get_boundary_and_counter(query)
        # Chuyển đổi danh sách thành JSON và trả về
        return jsonify({'boundaries': boundaries_list}), 200
    
    @roiboundary_bp.route('/get_boundary_property', methods=['POST'])
    def get_boundary_property():
        data = request.json
        query = {key: value for key, value in data.items() if value is not None}
        results = Boundary.find(query)
        # Tạo một danh sách dưới dạng từ điển
        boundaries_list = [{'id': boundary['id'], 
                    'camId': boundary['camId'], 
                    'in': boundary.get('in') if boundary.get('in') is not None else 0,
                    'out': boundary.get('out') if boundary.get('out') is not None else 0,
                    'pointL': boundary['pointL'], 
                    'pointR': boundary['pointR'], 
                    'pointDirect': boundary['pointDirect']} 
                    for boundary in results]
        # Chuyển đổi danh sách thành JSON và trả về
        return jsonify({'boundaries': boundaries_list}), 200


    @roiboundary_bp.route('/update_insert_boundary', methods=['POST'])
    def update_insert_boundary():
        data = request.json
        id = data.get('id')
        if id is None:
            return jsonify({'error': 'Missing id in request data'}), 400
        update_data = {}
        for key in data:
            if key != 'id':
                update_data[key] = data[key]
        if Boundary.update_or_insert(id, **update_data):
            camId = data.get("camId")
            get_boundary_and_counter({"camId": camId})
            return jsonify({'message': 'Update Insert successful'}), 200
        else:
            return jsonify({'error': 'Failed to update Insert data'}), 400
    
    # Roi endpoint

    @roiboundary_bp.route('/testroi', methods=['POST'])
    def testroi():
        data = request.json
        return jsonify({"message": "Received data successfully", "data": data})
    
    def get_roi_and_counter(query):
        # Thực hiện truy vấn
        results = Roi.find(query)
        # Tạo một danh sách dưới dạng từ điển
        rois_list = [{'id': roi['id'], 
                    'camId': roi['camId'], 
                    'points': roi['points'],
                    'mapping points': roi['mapping points']} 
                    for roi in results]
        Roi.get_polygon_annotators(rois_list)
        return rois_list

    @roiboundary_bp.route('/insert_roi', methods=['POST'])
    def insert_roi():
        # Lấy dữ liệu từ request
        data = request.json
        # Kiểm tra nếu id đã được cung cấp trong dữ liệu
        if data.get('id') is None:
            return jsonify({'error': 'Missing id in request data'}), 400
        Roi.insert(data)
        return jsonify({'message': 'Insert or update successful'}), 200
    
    @roiboundary_bp.route('/update_roi', methods=['POST'])
    def update_roi():
        data = request.json
        id = data.get('id')
        if id is None:
            return jsonify({'error': 'Missing id in request data'}), 400
        update_data = {}
        for key in data:
            if key != 'id':
                update_data[key] = data[key]

        if Roi.update(id, **update_data):
            return jsonify({'message': 'Update successful'}), 200
        else:
            return jsonify({'error': 'Failed to update data'}), 400
    
    @roiboundary_bp.route('/get_rois', methods=['POST'])
    def get_rois():
        data = request.json
        query = {key: value for key, value in data.items() if value is not None}
        rois_list = get_roi_and_counter(query)
        return jsonify({'rois': rois_list}), 200

    @roiboundary_bp.route('/update_insert_roi', methods=['POST'])
    def update_insert_roi():
        data = request.json
        id = data.get('id')
        if id is None:
            return jsonify({'error': 'Missing id in request data'}), 400
        update_data = {}
        for key in data:
            if key != 'id':
                update_data[key] = data[key]
        if Roi.update_or_insert(id, **update_data):
            camId = data.get("camId")
            get_roi_and_counter({"camId": camId})
            return jsonify({'message': 'Update Insert successful'}), 200
        else:
            return jsonify({'error': 'Failed to update Insert data'}), 400

    @roiboundary_bp.route('/delete_roi', methods=['POST'])
    def delete_roi():
        # Lấy dữ liệu từ request
        data = request.json
        id = data.get('id')
        if id is None:
            return jsonify({'error': 'Missing id in request data'}), 400
        query = {'id': id}
        # Thực hiện xóa
        result = Roi.delete(query)
        if result.acknowledged:
            camId = data.get('camId')
            get_roi_and_counter({'camId': camId})
            return jsonify({'message': 'Delete successful'}), 200
        else:
            return jsonify({'error': 'Failed to delete data'}), 400
        
    @roiboundary_bp.route('/get_roi_property', methods=['POST'])
    def get_roi_property():
        data = request.json
        query = {key: value for key, value in data.items() if value is not None}
        results = Roi.find(query)
        rois_list = [{'id': roi['id'], 
                    'camId': roi['camId'], 
                    'speed limit': roi["speed limit"],
                    'mapping points': roi['mapping points']} 
                    for roi in results]
        # print(rois_list, query)
        # Chuyển đổi danh sách thành JSON và trả về
        return jsonify({'rois': rois_list}), 200