import cv2
import numpy as np
import os
import shutil
import logging
import threading
import warnings
from pathlib import Path
from deepface import DeepFace

# Suppress warnings for professional console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognizer")

class FaceRecognizerWrapper:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.photos_dir = self.data_dir / "photos"
        self.db_path = str(self.photos_dir)
        
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock() # For thread-safe file operations
        
        # Core Model: ArcFace (Highest Accuracy)
        self.model_name = "ArcFace" 
        # Detector: MediaPipe (Fastest on CPU)
        self.detector_backend = "mediapipe"
        self.distance_metric = "cosine"
        
        # Strict Threshold for security (0.68 is standard, 0.55 is very strict)
        self.threshold = 0.55 
        
        self._preload_models()

    def _preload_models(self):
        """Pre-load models to ensure instant first response."""
        try:
            logger.info(f"Pre-loading {self.model_name}...")
            DeepFace.build_model(self.model_name)
            # Dummy call to warm up the detector
            DeepFace.extract_faces(
                img_path=np.zeros((224, 224, 3), dtype=np.uint8),
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            logger.info("Models ready.")
        except Exception as e:
            logger.error(f"Pre-load failed: {e}")

    def preprocess_image(self, image_np):
        """Enhance image for low-light and low-res capture."""
        try:
            # Contrast Limited Adaptive Histogram Equalization
            lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Subtle Sharpening
            kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
            final_img = cv2.filter2D(final_img, -1, kernel)
            
            return final_img
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return image_np

    def clear_data(self):
        """Reset the system data."""
        with self.lock:
            try:
                if self.photos_dir.exists():
                    shutil.rmtree(self.photos_dir)
                self.photos_dir.mkdir(parents=True, exist_ok=True)
                
                # Remove DeepFace cache/index
                for pkl in self.data_dir.glob("representations_*.pkl"):
                    pkl.unlink()
                logger.info("Database cleared.")
            except Exception as e:
                logger.error(f"Clear failed: {e}")

    def save_face(self, name, image_np):
        """Robust registration with multiple detector fallbacks."""
        person_dir = self.photos_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        count = len(list(person_dir.glob("*.jpg")))
        processed_img = self.preprocess_image(image_np)
        
        # Fallback loop for difficult environments
        backends = [self.detector_backend, 'mtcnn', 'opencv', 'ssd']
        faces = None
        
        for backend in backends:
            try:
                faces = DeepFace.extract_faces(
                    img_path=processed_img,
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True
                )
                if faces:
                    logger.info(f"Registered {name} using {backend}")
                    break
            except:
                continue
        
        if not faces:
            return False
            
        saved_any = False
        for face_data in faces:
            # Color conversion for DeepFace compatibility
            face_img = (face_data["face"] * 255).astype(np.uint8)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            
            img_path = person_dir / f"{name}_{count}.jpg"
            cv2.imwrite(str(img_path), face_img)
            count += 1
            saved_any = True
            
        if saved_any:
            with self.lock:
                for pkl in self.data_dir.glob("representations_*.pkl"):
                    pkl.unlink()
                    
        return saved_any

    def train(self):
        """Trigger database indexing."""
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
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def recognize(self, image_np):
        """High-accuracy recognition with face tracking box."""
        if not self.photos_dir.exists() or not any(self.photos_dir.iterdir()):
            return "Unknown", 0.0, None

        try:
            processed_img = self.preprocess_image(image_np)
            
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
                
                # Security Check
                if dist < self.threshold:
                    name = Path(best['identity']).parent.name
                    # Map cosine distance to confidence percentage
                    conf = round((1 - (dist / self.threshold)) * 100, 2)
                    box = (int(best['source_x']), int(best['source_y']), int(best['source_w']), int(best['source_h']))
                    logger.info(f"Confidence HIGH: Identified {name}")
                    return name, conf, box
            
            # Fallback for box only if unidentified
            faces = DeepFace.extract_faces(processed_img, detector_backend=self.detector_backend, enforce_detection=False)
            if faces and faces[0]['facial_area']['w'] > 0:
                fa = faces[0]['facial_area']
                return "Unknown", 0.0, (int(fa['x']), int(fa['y']), int(fa['w']), int(fa['h']))

        except Exception as e:
            pass # Keep silent for real-time loop

        return "Unknown", 0.0, None
