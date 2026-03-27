import os
import cv2
import numpy as np
import logging
import threading
import json
from pathlib import Path
from datetime import datetime

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognizer")

class FaceRecognizerWrapper:
    def __init__(self, data_dir="data"):
        self.base_dir = Path(data_dir).resolve()
        self.photos_dir = self.base_dir / "photos"
        self.model_path = self.base_dir / "trainer.yml"
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        
        self.lock = threading.Lock()
        
        # --- OPENCV LBPH CONFIG ---
        # No external AI weights needed. Extremely stable.
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            # Haar Cascade is built into OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.model_path.exists():
                self.recognizer.read(str(self.model_path))
                logger.info("LBPH Model loaded from disk.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV Recognizer: {e}")
            raise e

        self.label_map = self._load_label_map()

    def _load_label_map(self):
        map_path = self.base_dir / "labels.json"
        if map_path.exists():
            with open(map_path, 'r') as f:
                # Convert keys back to int
                return {int(k): v for k, v in json.load(f).items()}
        return {}

    def _save_label_map(self):
        map_path = self.base_dir / "labels.json"
        with open(map_path, 'w') as f:
            json.dump(self.label_map, f)

    def preprocess_image(self, img):
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Equalize histogram for better light consistency
        return cv2.equalizeHist(gray)

    def save_face(self, name, image_np):
        """Standard registration flow using Haar cascades and LBPH samples."""
        with self.lock:
            person_dir = self.photos_dir / name
            raw_dir = person_dir / "raw"
            person_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Save Raw Original
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_count = len(list(raw_dir.glob("*.jpg")))
            cv2.imwrite(str(raw_dir / f"raw_{raw_count}_{timestamp}.jpg"), image_np)
            
            # 2. Extract Face for Training
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                logger.warning(f"No face detected in capture for {name}")
                return False
                
            x, y, w, h = faces[0]
            face_sample = gray[y:y+h, x:x+w]
            
            sample_count = len(list(person_dir.glob("*.jpg")))
            img_path = person_dir / f"{name}_{sample_count}.jpg"
            cv2.imwrite(str(img_path), face_sample)
            
            # 3. Update Metadata
            info_path = person_dir / "info.json"
            info = {
                "name": name,
                "last_registered": str(datetime.now()),
                "sample_count": sample_count + 1
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            
            return True

    def train(self):
        """Train LBPH recognizer on all stored samples."""
        readiness = self.check_dataset_readiness()
        if not readiness["ready"]:
            return False, readiness["message"]
            
        with self.lock:
            faces = []
            labels = []
            new_label_map = {}
            current_id = 0
            
            for person_dir in self.photos_dir.iterdir():
                if person_dir.is_dir():
                    name = person_dir.name
                    new_label_map[current_id] = name
                    
                    for img_file in person_dir.glob("*.jpg"):
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                            labels.append(current_id)
                    current_id += 1
            
            if not faces:
                return False, "No samples found for training."
                
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(str(self.model_path))
            self.label_map = new_label_map
            self._save_label_map()
            
            logger.info(f"LBPH Model trained for {len(self.label_map)} people.")
            return True, "Model updated successfully."

    def check_dataset_readiness(self, min_samples=5):
        if not self.photos_dir.exists(): return {"ready": False, "message": "No registry."}
        people = [d for d in self.photos_dir.iterdir() if d.is_dir()]
        if not people: return {"ready": False, "message": "No people registered."}
        
        insufficient = []
        for person_dir in people:
            count = len(list(person_dir.glob("*.jpg")))
            if count < min_samples:
                insufficient.append(f"{person_dir.name} ({count}/{min_samples})")
        
        if insufficient:
            return {"ready": False, "message": f"Need {min_samples} samples for: {', '.join(insufficient)}"}
        return {"ready": True, "message": "Ready."}

    def recognize(self, image_np, fast_only=False):
        """Instant recognition using LBPH predict."""
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0: return "Unknown", 0.0, None
            
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            
            if not self.label_map:
                return "Unknown", 0.0, (int(x), int(y), int(w), int(h))

            # LBPH Predict
            label_id, confidence_dist = self.recognizer.predict(roi_gray)
            
            # LBPH confidence is DISTANCE (lower is better)
            # Typically < 100 is a good match depending on environment
            threshold = 85 
            if confidence_dist < threshold:
                name = self.label_map.get(label_id, "Unknown")
                # Normalize confidence for display (0-100 where higher is better)
                display_conf = round(max(0, 100 - confidence_dist / 1.5), 2)
                return name, display_conf, (int(x), int(y), int(w), int(h))
            
            return "Unknown", 0.0, (int(x), int(y), int(w), int(h))
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return "Unknown", 0.0, None

    def clear_data(self):
        import shutil
        if self.photos_dir.exists():
            shutil.rmtree(self.photos_dir)
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        if self.model_path.exists(): self.model_path.unlink()
        self.label_map = {}
        self._save_label_map()
        logger.info("Registry cleared.")
