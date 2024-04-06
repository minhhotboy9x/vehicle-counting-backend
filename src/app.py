from flask import Flask, Response, render_template
from flask_cors import CORS
from routes.stream import streaming_bp
from routes.auth_route import auth_route
from pymongo import MongoClient
from config import STRING_CONNECTION

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins
app.register_blueprint(streaming_bp)
app.register_blueprint(auth_route)

# Function to test MongoDB connection
def test_mongodb_connection():
    try:
        client = MongoClient(STRING_CONNECTION)
        db = client.test_database
        collection = db.test_collection
        collection.insert_one({'test_key': 'test_value'})
        client.close()
        print("Connected to DB")
        return True
    except Exception as e:
        print("Error connecting to MongoDB:", e)
        return False

# Route to test MongoDB connection
@app.route('/test_mongodb')
def test_mongodb():
    if test_mongodb_connection():
        return "Successfully connected to MongoDB!"
    else:
        return "Failed to connect to MongoDB!"

# Route for testing other functionality
@app.route('/')
def index():
    return render_template('index.html')
    # return "hello"

if __name__ == '__main__':
    app.run(debug=True)
