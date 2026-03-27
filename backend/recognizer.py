import os
import cv2
import numpy as np
import logging
import threading
import json
from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognizer")

class FaceRecognizerWrapper:
    def __init__(self, data_dir="data"):
        self.base_dir = Path(data_dir).resolve()
        self.photos_dir = self.base_dir / "photos"
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        
        self.lock = threading.Lock()
        
        # --- INSIGHTFACE CONFIG ---
        # buffalo_l is the high-accuracy model pack. 
        # We force CPUExecutionProvider to avoid CUDA/GPU version mismatches on client.
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace Engine: buffalo_l initialized on CPU.")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise e

        self.face_bank = {} # {name: [embedding1, embedding2, ...]}
        self.train() # Load existing data

    def preprocess_image(self, img):
        """Standard preprocessing for consistency."""
        if img is None: return None
        try:
            # Subtle enhancement for better detection
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        except: pass
        return img

    def save_face(self, name, image_np):
        """Capture raw frame, extract aligned face, and save embedding."""
        with self.lock:
            person_dir = self.photos_dir / name
            raw_dir = person_dir / "raw"
            person_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Save Raw Original
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_count = len(list(raw_dir.glob("*.jpg")))
            cv2.imwrite(str(raw_dir / f"raw_{raw_count}_{timestamp}.jpg"), image_np)
            
            # 2. InsightFace Extraction & Embedding
            faces = self.app.get(image_np)
            if not faces:
                logger.warning(f"No face detected for {name}")
                return False
                
            # Take the largest face found
            face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            
            # Save cropped aligned face
            sample_count = len(list(person_dir.glob("*.jpg")))
            # Note: insightface doesn't return aligned face directly in get(), 
            # we can crop using bbox or use its internal alignment if needed.
            # For DeepFace compatibility, we'll save a simple crop.
            x1, y1, x2, y2 = map(int, face.bbox)
            crop = image_np[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                cv2.imwrite(str(person_dir / f"{name}_{sample_count}.jpg"), crop)
            
            # 3. Save Embedding (.npy) for fast loading
            embedding = face.normed_embedding
            npy_path = person_dir / f"{name}_{sample_count}.npy"
            np.save(str(npy_path), embedding)
            
            # 4. Update Metadata
            info_path = person_dir / "info.json"
            info = {
                "name": name,
                "last_registered": str(datetime.now()),
                "sample_count": sample_count + 1,
                "status": "authorized"
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            
            self.train() # Hot-reload embeddings
            return True

    def train(self):
        """Load all saved embeddings into memory for fast matching."""
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
                            emb = np.load(str(npy_file))
                            embeddings.append(emb)
                        except: continue
                    if embeddings:
                        self.face_bank[name] = embeddings
            
            logger.info(f"Loaded {len(self.face_bank)} people into face bank.")
            return True, "Face bank updated."

    def check_dataset_readiness(self, min_samples=5):
        """Verify min samples requirements."""
        if not self.photos_dir.exists(): return {"ready": False, "message": "No registry."}
        people = [d for d in self.photos_dir.iterdir() if d.is_dir()]
        if not people: return {"ready": False, "message": "Registry empty."}
        
        insufficient = []
        for person_dir in people:
            count = len(list(person_dir.glob("*.npy")))
            if count < min_samples:
                insufficient.append(f"{person_dir.name} ({count}/{min_samples})")
        
        if insufficient:
            return {"ready": False, "message": f"Need more samples for: {', '.join(insufficient)}"}
        return {"ready": True, "message": "All set."}

    def recognize(self, image_np, fast_only=False):
        """Identify faces using InsightFace and Cosine Similarity."""
        try:
            # 1. Detect faces in current frame
            faces = self.app.get(image_np)
            if not faces: return "Unknown", 0.0, None
            
            # Consider the most prominent face
            face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            bbox = [int(v) for v in face.bbox]
            
            if fast_only or not self.face_bank:
                return "Unknown", 0.0, (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

            # 2. Compare against face bank
            current_emb = face.normed_embedding.reshape(1, -1)
            best_name = "Unknown"
            max_sim = 0
            
            for name, bank_embs in self.face_bank.items():
                # Compare against all samples for this person and take max
                sims = cosine_similarity(current_emb, np.array(bank_embs))
                avg_sim = np.max(sims) # Or np.mean
                if avg_sim > max_sim:
                    max_sim = avg_sim
                    best_name = name
            
            # Thresholding (InsightFace embeddings are very distinct)
            threshold = 0.45 
            if max_sim > threshold:
                confidence = round(max_sim * 100, 2)
                return best_name, confidence, (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
            
            return "Unknown", 0.0, (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
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
