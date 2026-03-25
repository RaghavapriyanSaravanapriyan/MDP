import base64
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from recognizer import FaceRecognizerWrapper

app = Flask(__name__, static_folder="../frontend")
CORS(app)

# Initialize Recognizer
recog = FaceRecognizerWrapper(data_dir=os.path.join(os.path.dirname(__file__), "..", "data"))

def base64_to_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/setup', methods=['POST'])
def setup():
    recog.clear_data()
    return jsonify({"status": "success", "message": "Environment cleared for new setup"})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get("name")
    image_b64 = data.get("image")
    
    if not name or not image_b64:
        return jsonify({"status": "error", "message": "Name or image missing"}), 400
        
    img = base64_to_image(image_b64)
    success = recog.save_face(name, img)
    
    if success:
        return jsonify({"status": "success", "message": f"Saved face for {name}"})
    else:
        return jsonify({"status": "error", "message": "No face detected in the image"}), 400

@app.route('/api/train', methods=['POST'])
def train():
    success = recog.train()
    if success:
        return jsonify({"status": "success", "message": "Model trained successfully"})
    else:
        return jsonify({"status": "error", "message": "Training failed, no faces found"}), 400

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    data = request.json
    image_b64 = data.get("image")
    
    if not image_b64:
        return jsonify({"status": "error", "message": "Image missing"}), 400
        
    img = base64_to_image(image_b64)
    name, confidence, box = recog.recognize(img)
    
    return jsonify({
        "status": "success",
        "name": name,
        "confidence": float(confidence),
        "locked": name == "Unknown"
    })

@app.route('/api/team', methods=['GET'])
def get_team():
    team_path = os.path.join(recog.data_dir, "team.txt")
    if os.path.exists(team_path):
        with open(team_path, 'r') as f:
            members = [line.strip() for line in f.readlines() if line.strip()]
        return jsonify({"status": "success", "team": members})
    return jsonify({"status": "success", "team": []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
