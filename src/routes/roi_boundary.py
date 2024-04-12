from flask import Blueprint, request, jsonify
from model.M_Boundary import Boundary

roiboundary_bp = Blueprint('roi_boundary', __name__)

def init_roiboundary_bp(mongo):
    Boundary.init_mongo(mongo)
    @roiboundary_bp.route('/testboundary', methods=['POST'])
    def testboundary():
        data = request.json
        return jsonify({"message": "Received data successfully", "data": data})

    @roiboundary_bp.route('/insert_boundary', methods=['POST'])
    def insert_boundary():
        # Lấy dữ liệu từ request
        data = request.json
        # Kiểm tra nếu id đã được cung cấp trong dữ liệu
        if id is None:
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

        if result.deleted_count > 0:
            return jsonify({'message': 'Delete successful'}), 200
        else:
            return jsonify({'error': 'Failed to delete data'}), 400
    

    @roiboundary_bp.route('/get_boundaries', methods=['POST'])
    def get_boundaries():
        data = request.json
        query = {key: value for key, value in data.items() if value is not None}
        # Thực hiện truy vấn
        results = Boundary.find(query)
        # Tạo một danh sách dưới dạng từ điển
        boundaries_list = [{'id': boundary['id'], 
                            'camId': boundary['camId'], 
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
            return jsonify({'message': 'Update Insert successful'}), 200
        else:
            return jsonify({'error': 'Failed to update Insert data'}), 400