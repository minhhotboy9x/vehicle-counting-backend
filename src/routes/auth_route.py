from flask import Blueprint, request, jsonify
from service.auth_login import register_user, authenticate_user

auth_route = Blueprint('auth', __name__)

# Route to handle registration
@auth_route.route('/register', methods=['POST'])
def register():
    # Get data from request
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Call register_user function from auth_service
    success, message = register_user(email, password)

    if success:
        return jsonify({'message': message}), 200
    else:
        return jsonify({'message': message}), 400

@auth_route.route('/login', methods=['POST'])
def login():
    # Get data from request
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Call authenticate_user function from auth_service
    success, message = authenticate_user(email, password)

    if success:
        print(f'{email} logged in')
        return jsonify({'message': message}), 200
    else:
        print('login failed')
        return jsonify({'message': message}), 401