import os
import cv2
import numpy as np
import logging
import threading
from pathlib import Path
from deepface import DeepFace

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognizer")

# Suppress excessively verbose TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FaceRecognizerWrapper:
    def __init__(self, data_dir="data"):
        self.base_dir = Path(data_dir).resolve()
        self.photos_dir = self.base_dir / "photos"
        self.db_path = str(self.photos_dir)
        
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        # --- VISION PIPELINE CONFIG ---
        # 1. Fast Path: OpenCV Haar Cascades for real-time box tracking
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 2. Identity Path: Facenet512 (99.65% Accuracy, stable download)
        self.model_name = "Facenet512" 
        self.detector_backend = "mediapipe" # Fast for real-time
        self.registration_detector = "retinaface" # Accurate for setup
        self.distance_metric = "cosine"
        self.threshold = 0.30 
        
        logger.info(f"Vision Engine: {self.model_name} initialized.")
        self._preload_models()

    def _preload_models(self):
        """Warm up deep learning models to eliminate first-request lag."""
        try:
            # Trigger lazy load of models
            DeepFace.represent(
                img_path=np.zeros((224, 224, 3), dtype=np.uint8),
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False
            )
            logger.info("Deep learning models warmed up.")
        except Exception as e:
            logger.error(f"Pre-load warning: {e}")

    def preprocess_image(self, img):
        """Enhance low-light and low-res images using CLAHE and sharpening."""
        if img is None: return None
        try:
            # 1. CLAHE for light balancing
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 2. Sharpening for clarity
            gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
            img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
        except: pass
        return img

    def save_face(self, name, image_np):
        """Robust registration with retinaface, raw backups, and metadata."""
        import json
        from datetime import datetime
        
        with self.lock:
            person_dir = self.photos_dir / name
            raw_dir = person_dir / "raw"
            person_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Save Raw Original
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_count = len(list(raw_dir.glob("*.jpg")))
            raw_path = raw_dir / f"raw_{raw_count}_{timestamp}.jpg"
            cv2.imwrite(str(raw_path), image_np)
            
            # 2. Extract and Save High-Accuracy Face Sample
            processed_img = self.preprocess_image(image_np)
            sample_count = len(list(person_dir.glob("*.jpg")))
            
            # Use retinaface for maximum accuracy during registration
            backends = [self.registration_detector, 'mediapipe', 'mtcnn', 'opencv']
            faces = None
            for backend in backends:
                try:
                    faces = DeepFace.extract_faces(
                        img_path=processed_img,
                        detector_backend=backend,
                        enforce_detection=True,
                        align=True
                    )
                    if faces: break
                except: continue
            
            if not faces:
                logger.warning(f"Registration failed: No face detected for {name}")
                return False
                
            for face_data in faces:
                face_img = face_data['face']
                if face_img.max() <= 1.0:
                    face_img = (face_img * 255).astype(np.uint8)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                
                img_path = person_dir / f"{name}_{sample_count}.jpg"
                cv2.imwrite(str(img_path), face_img)
                sample_count += 1
            
            # 3. Update Metadata
            info_path = person_dir / "info.json"
            info = {
                "name": name,
                "last_registered": str(datetime.now()),
                "sample_count": sample_count,
                "status": "authorized"
            }
            try:
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=4)
            except: pass
            
            self._clear_index()
            return True

    def _clear_index(self):
        """Flush old index files to force fresh rebuild. Safe for Windows file locks."""
        for pkl in self.photos_dir.glob("representations_*.pkl"):
            try:
                pkl.unlink()
            except Exception as e:
                logger.warning(f"Could not unlink {pkl.name}: {e}")

    def train(self):
        """Index the facial database. Hardened against empty or insufficient folders."""
        with self.lock:
            readiness_status = self.check_dataset_readiness()
            if not readiness_status["ready"]:
                logger.warning(f"Indexing failed: {readiness_status['message']}")
                return False, readiness_status["message"]
                
            try:
                self._clear_index()
                # Try primary detector, fall back to opencv if mediapipe is broken on client
                try:
                    DeepFace.find(
                        img_path=np.zeros((224, 224, 3), dtype=np.uint8),
                        db_path=self.db_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        distance_metric=self.distance_metric,
                        enforce_detection=False,
                        silent=True
                    )
                except Exception as e:
                    if "mediapipe" in str(e).lower():
                        logger.warning("Mediapipe failed on client, falling back to opencv for indexing...")
                        DeepFace.find(
                            img_path=np.zeros((224, 224, 3), dtype=np.uint8),
                            db_path=self.db_path,
                            model_name=self.model_name,
                            detector_backend="opencv",
                            distance_metric=self.distance_metric,
                            enforce_detection=False,
                            silent=True
                        )
                    else: raise e
                logger.info("Global facial index rebuilt.")
                return True, "Database indexed successfully"
            except Exception as e:
                logger.error(f"Indexing error: {e}")
                return False, f"Internal indexing error: {str(e)}"

    def check_dataset_readiness(self, min_samples=5):
        """Check registration status using metadata and file counts."""
        import json
        if not self.photos_dir.exists():
            return {"ready": False, "message": "No registry found."}
            
        people = [d for d in self.photos_dir.iterdir() if d.is_dir()]
        if not people:
            return {"ready": False, "message": "No people registered."}
            
        insufficient = []
        for person_dir in people:
            # Prefer metadata if exists
            info_path = person_dir / "info.json"
            count = 0
            if info_path.exists():
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        count = json.load(f).get("sample_count", 0)
                except: pass
            
            # Fallback to actual file count
            if count < min_samples:
                count = len(list(person_dir.glob("*.jpg")))
                
            if count < min_samples:
                insufficient.append(f"{person_dir.name} ({count}/{min_samples})")
        
        if insufficient:
            return {
                "ready": False, 
                "message": f"Insufficient data for: {', '.join(insufficient)}. Need {min_samples} samples."
            }
            
        return {"ready": True, "message": "Registry ready for indexing."}

    def recognize(self, image_np, fast_only=False):
        """Dual-Path logic: Optimized tracking vs. High-accuracy identity."""
        try:
            processed_img = self.preprocess_image(image_np)
            
            # --- Path A: Fast Tracking Box (Zero Latency) ---
            if fast_only:
                gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    return "Unknown", 0.0, (int(x), int(y), int(w), int(h))
                return "Unknown", 0.0, None

            # --- Path B: Secure Identity Check (Deep Learning) ---
            if not any(self.photos_dir.glob("*/*.jpg")):
                return "Unknown", 0.0, None

            results = DeepFace.find(
                img_path=processed_img,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=True,
                silent=True
            )
            
            if results and not results[0].empty:
                best = results[0].iloc[0]
                dist = best[f'{self.model_name}_{self.distance_metric}']
                
                if dist < self.threshold:
                    name = Path(best['identity']).parent.name
                    conf = round((1 - (dist / self.threshold)) * 100, 2)
                    box = (int(best['source_x']), int(best['source_y']), int(best['source_w']), int(best['source_h']))
                    return name, conf, box
            
            # Fallback to fast box if recognition is inconclusive
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                return "Unknown", 0.0, (int(x), int(y), int(w), int(h))

        except: pass 
        return "Unknown", 0.0, None

    def clear_data(self):
        import shutil
        if self.photos_dir.exists():
            shutil.rmtree(self.photos_dir)
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self._clear_index()
        logger.info("Face database cleared.")
