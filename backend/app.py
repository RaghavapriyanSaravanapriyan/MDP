import os
import warnings
import base64
import numpy as np
import cv2
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from recognizer import FaceRecognizerWrapper

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceApp")

app = Flask(__name__, static_folder="../frontend")
CORS(app)

# Configuration
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Initialize Recognizer
recog = FaceRecognizerWrapper(data_dir=str(DATA_DIR))

def base64_to_image(b64_string):
    """Convert base64 string to OpenCV BGR image."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        return img
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        return None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/setup', methods=['POST'])
def setup():
    recog.clear_data()
    return jsonify({"status": "success", "message": "System reset successfully"})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get("name")
    image_b64 = data.get("image")
    
    if not name or not image_b64:
        return jsonify({"status": "error", "message": "Name and image required"}), 400
        
    img = base64_to_image(image_b64)
    if img is None:
        return jsonify({"status": "error", "message": "Failed to decode image"}), 400
        
    success = recog.save_face(name, img)
    if success:
        return jsonify({"status": "success", "message": f"Collected samples for {name}"})
    else:
        return jsonify({"status": "error", "message": "Could not detect face in sample"}), 400

@app.route('/api/train', methods=['POST'])
def train():
    success = recog.train()
    if success:
        return jsonify({"status": "success", "message": "Database indexed successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to index faces"}), 400

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    data = request.json
    image_b64 = data.get("image")
    
    if not image_b64:
        return jsonify({"status": "error", "message": "Image required"}), 400
        
    img = base64_to_image(image_b64)
    if img is None:
        return jsonify({"status": "error", "message": "Failed to decode image"}), 400
        
    name, confidence, box = recog.recognize(img)
    
    return jsonify({
        "status": "success",
        "name": name,
        "confidence": float(confidence),
        "locked": name == "Unknown",
        "box": box
    })

@app.route('/api/team', methods=['GET'])
def get_team():
    team_path = DATA_DIR / "team.txt"
    if team_path.exists():
        try:
            with open(team_path, 'r') as f:
                members = [line.strip() for line in f.readlines() if line.strip()]
            return jsonify({"status": "success", "team": members})
        except Exception as e:
            logger.error(f"Error reading team.txt: {e}")
            return jsonify({"status": "error", "message": "Internal error"}), 500
    return jsonify({"status": "success", "team": []})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Vision Server active on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
