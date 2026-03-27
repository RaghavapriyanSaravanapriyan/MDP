import os
import cv2
import numpy as np
import logging
import threading
import json
import requests
from pathlib import Path
from datetime import datetime

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognizer")

class FaceRecognizerWrapper:
    def __init__(self, data_dir="data"):
        self.base_dir = Path(data_dir).resolve()
        self.photos_dir = self.base_dir / "photos"
        self.models_dir = self.base_dir / "models"
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.lock = threading.Lock()
        
        # --- MODEL PATHS ---
        self.det_model_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        self.rec_model_path = self.models_dir / "face_recognition_sface_2021dec.onnx"
        
        # --- DOWNLOAD MODELS IF MISSING ---
        self._ensure_models()
        
        # --- INITIALIZE OPENCV DNN MODULES ---
        try:
            # 1. Initialize Detector (YuNet)
            self.detector = cv2.FaceDetectorYN.create(
                str(self.det_model_path), "", (320, 320), 0.9, 0.3, 5000
            ) 
            # 2. Initialize Recognizer (SFace)
            self.recognizer = cv2.FaceRecognizerSF.create(
                str(self.rec_model_path), ""
            )
            logger.info("High-Accuracy SFace Engine initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize SFace: {e}")
            raise e

        self.face_bank = {} # {name: [embedding1, embedding2, ...]}
        self.train() 

    def _ensure_models(self):
        """Download YuNet and SFace models from OpenCV Zoo if missing."""
        urls = {
            self.det_model_path: "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            self.rec_model_path: "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        }
        for path, url in urls.items():
            if not path.exists():
                logger.info(f"--- ATTENTION: DOWNLOADING MODEL '{path.name}' (~30MB) ---")
                logger.info(f"Source: {url}")
                try:
                    response = requests.get(url, stream=True, timeout=60)
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    with open(path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0 and downloaded % (1024 * 1024) == 0:
                                    logger.info(f"Progress: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
                    logger.info(f"Model successfully saved to {path}")
                except Exception as e:
                    logger.error(f"DOWNLOAD FAILED for {path.name}. Reason: {e}")
                    if path.exists(): path.unlink() # Remove partial file
                    raise Exception(f"Failed to download AI model {path.name}: {e}. Check your internet connection.")

    def save_face(self, name, image_np):
        """Capture raw frame, extract aligned face crop, and generate SFace embedding."""
        with self.lock:
            person_dir = self.photos_dir / name
            raw_dir = person_dir / "raw"
            person_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Save Raw Original
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_count = len(list(raw_dir.glob("*.jpg")))
            cv2.imwrite(str(raw_dir / f"raw_{raw_count}_{timestamp}.jpg"), image_np)
            
            # 2. Extract SFace Embedding
            h, w = image_np.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image_np)
            
            if faces is None:
                logger.warning(f"No face detected for {name}")
                return False
                
            # Align and crop the face
            aligned_face = self.recognizer.alignCrop(image_np, faces[0])
            embedding = self.recognizer.feature(aligned_face)
            
            # Save cropped face sample
            sample_count = len(list(person_dir.glob("*.jpg")))
            img_path = person_dir / f"{name}_{sample_count}.jpg"
            cv2.imwrite(str(img_path), aligned_face)
            
            # Save embedding for fast training
            np.save(str(person_dir / f"{name}_{sample_count}.npy"), embedding)
            
            # 3. Update Metadata
            info_path = person_dir / "info.json"
            info = {
                "name": name,
                "last_registered": str(datetime.now()),
                "sample_count": sample_count + 1
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            
            self.train() # Hot-reload
            return True

    def train(self):
        """Load all saved SFace embeddings into memory."""
        readiness = self.check_dataset_readiness()
        if not readiness["ready"]:
            return False, readiness["message"]
            
        with self.lock:
            self.face_bank = {}
            for person_dir in self.photos_dir.iterdir():
                if person_dir.is_dir():
                    name = person_dir.name
                    embeddings = []
                    for npy_file in person_dir.glob("*.npy"):
                        try:
                            embeddings.append(np.load(str(npy_file)))
                        except: continue
                    if embeddings:
                        self.face_bank[name] = embeddings
            
            logger.info(f"Loaded {len(self.face_bank)} people into SFace bank.")
            return True, "Identity database updated."

    def check_dataset_readiness(self, min_samples=5):
        if not self.photos_dir.exists(): return {"ready": False, "message": "No registry."}
        people = [d for d in self.photos_dir.iterdir() if d.is_dir()]
        if not people: return {"ready": False, "message": "Registry empty."}
        
        insufficient = []
        for person_dir in people:
            count = len(list(person_dir.glob("*.npy")))
            if count < min_samples:
                insufficient.append(f"{person_dir.name} ({count}/5)")
        
        if insufficient:
            return {"ready": False, "message": f"Need 5 samples for: {', '.join(insufficient)}"}
        return {"ready": True, "message": "Ready."}

    def recognize(self, image_np, fast_only=False):
        """Perform high-accuracy identity matching using Cosine Similarity on SFace embeddings."""
        try:
            h, w = image_np.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image_np)
            
            if faces is None: return "Unknown", 0.0, None
            
            face = faces[0]
            bbox = [int(v) for v in face[:4]]
            
            if fast_only or not self.face_bank:
                return "Unknown", 0.0, (bbox[0], bbox[1], bbox[2], bbox[3])

            # 1. Generate current embedding
            aligned_face = self.recognizer.alignCrop(image_np, face)
            current_emb = self.recognizer.feature(aligned_face)
            
            # 2. Match against bank
            best_name = "Unknown"
            max_score = 0
            
            for name, bank_embs in self.face_bank.items():
                for bank_emb in bank_embs:
                    score = self.recognizer.match(current_emb, bank_emb, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > max_score:
                        max_score = score
                        best_name = name
            
            # SFace Cosine threshold is typically ~0.36
            threshold = 0.36 
            if max_score > threshold:
                confidence = round(max_score * 100, 2)
                return best_name, confidence, (bbox[0], bbox[1], bbox[2], bbox[3])
            
            return "Unknown", 0.0, (bbox[0], bbox[1], bbox[2], bbox[3])
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return "Unknown", 0.0, None

    def clear_data(self):
        import shutil
        if self.photos_dir.exists():
            shutil.rmtree(self.photos_dir)
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.face_bank = {}
        logger.info("Registry cleared.")
