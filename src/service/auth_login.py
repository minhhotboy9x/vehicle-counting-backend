from pymongo import MongoClient
from bson.objectid import ObjectId
from config import STRING_CONNECTION, DATABASE, ACCOUNT_COLLECTION
import bcrypt
import json

# Function to register user
def register_user(email, password):

    # Connect to MongoDB
    client = MongoClient(STRING_CONNECTION)
    db = client[DATABASE]
    collection = db[ACCOUNT_COLLECTION]
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    # Check if email exists
    if collection.find_one({'email': email}):
        return False, 'Email already exists'
    
    # Insert new user with hashed password
    result = collection.insert_one({
        'email': email,
        'password': hashed_password
    })

    client.close()
    return True, 'Registration successful'


# Function to authenticate user
def authenticate_user(email, provided_password):
    # Connect to MongoDB
    client = MongoClient(STRING_CONNECTION)
    db = client[DATABASE]
    collection = db[ACCOUNT_COLLECTION]

    # Retrieve user document from database
    user = collection.find_one({'email': email})

    # Check if user exists and verify password
    if user and verify_password(user['password'], provided_password):
        client.close()
        return True, 'Authentication successful'
    else:
        client.close()
        return False, 'Invalid email or password'

# Function to verify password
def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)
