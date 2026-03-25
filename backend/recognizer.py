import cv2
import numpy as np
import os
import shutil

class FaceRecognizerWrapper:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.photos_dir = os.path.join(data_dir, "photos")
        self.model_path = os.path.join(data_dir, "model.yml")
        self.names_path = os.path.join(data_dir, "names.npy")
        
        os.makedirs(self.photos_dir, exist_ok=True)
        
        # Initialize OpenCV components
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = {}  # Map int label to string name
        
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
        if os.path.exists(self.names_path):
            try:
                self.names = np.load(self.names_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading names: {e}")

    def clear_data(self):
        if os.path.exists(self.photos_dir):
            shutil.rmtree(self.photos_dir)
        os.makedirs(self.photos_dir, exist_ok=True)
        
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.names_path):
            os.remove(self.names_path)
            
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = {}
        
    def save_face(self, name, image_np):
        """
        Detect face and save it to the person's folder.
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        person_dir = os.path.join(self.photos_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        count = len(os.listdir(person_dir))
        
        # We save all faces found (or just the largest one). For simplicity, saving the first found face.
        # But maybe we should save all found to increase training data? Yes, let's process all.
        saved_any = False
        for (x, y, w, h) in faces:
            # Save the face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_roi)
            count += 1
            saved_any = True
            
        return saved_any
        
    def train(self):
        faces = []
        labels = []
        self.names = {}
        current_label = 0
        
        if not os.path.exists(self.photos_dir):
            return False
            
        for person_name in os.listdir(self.photos_dir):
            person_dir = os.path.join(self.photos_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            self.names[current_label] = person_name
            added_faces = False
            
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(current_label)
                    added_faces = True
                    
            if added_faces:
                current_label += 1
            
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(self.model_path)
            np.save(self.names_path, self.names)
            return True
        return False

    def recognize(self, image_np):
        if len(self.names) == 0:
            return "Unknown", 0.0, None
            
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        best_name = "Unknown"
        best_confidence = 0.0
        face_box = None
        min_distance = float('inf')
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            try:
                label, distance = self.recognizer.predict(face_roi)
            except Exception as e:
                continue
            
            # Distance usually 0 to 100+. Lower is better. 
            if distance < min_distance:
                min_distance = distance
                face_box = (x, y, w, h)
                if distance < 85: # Strictness threshold, can be tuned
                    best_name = self.names.get(label, "Unknown")
                    best_confidence = max(0, 100 - distance)
                else:
                    best_name = "Unknown"
                    best_confidence = max(0, 100 - distance)
                    
        return best_name, best_confidence, face_box
