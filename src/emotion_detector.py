"""
Emotion Detector Module
=======================
Pre-trained model untuk deteksi emosi dari ekspresi wajah
Menggunakan DeepFace library (tidak perlu training)

Kelas emosi: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
"""

import numpy as np
import cv2
import os

# Label emosi sesuai FER2013 dataset (sama untuk mapping)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Mapping emosi ke warna untuk visualisasi BGR
EMOTION_COLORS_BGR = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (144, 0, 255),   # Purple
    'Fear': (255, 0, 255),      # Magenta
    'Happy': (0, 255, 0),       # Green
    'Sad': (255, 0, 0),         # Blue
    'Surprise': (0, 255, 255),  # Yellow
    'Neutral': (128, 128, 128)  # Gray
}

# Mapping emosi ke warna HEX untuk Tkinter
EMOTION_COLORS = {
    'Angry': '#FF0000', # Red
    'Disgust': '#9000FF',      # Purple
    'Fear': '#FF00FF',         # Magenta
    'Happy': '#00FF00',        # Green
    'Sad': '#0000FF',          # Blue
    'Surprise': '#FFFF00',     # Yellow
    'Neutral': '#808080'        # Gray
}

# Mapping emosi ke tingkat stres (1=Rendah, 7=Kritis)
EMOTION_TO_STRESS = {
    'Happy': 1,
    'Surprise': 2,
    'Neutral': 3,
    'Sad': 4,
    'Fear': 5,
    'Angry': 6,
    'Disgust': 7
}

# Mapping dari DeepFace emotion labels ke format standar
DEEPFACE_EMOTION_MAP = {
    'happy': 'Happy',
    'sad': 'Sad',
    'angry': 'Angry',
    'surprise': 'Surprise',
    'fear': 'Fear',
    'disgust': 'Disgust',
    'neutral': 'Neutral'
}


class EmotionDetector:
    """
    Kelas untuk mendeteksi emosi dari gambar wajah
    Menggunakan DeepFace (pre-trained model - tanpa training)
    """

    def __init__(self, model_path=None):
        """
        Inisialisasi EmotionDetector dengan DeepFace

        Args:
            model_path: Path ke file model .h5 (optional, tidak dipakai di mode DeepFace)
        """
        self.model = None  # Tidak dipakai dengan DeepFace
        self.use_deepface = True
        self.deepface_available = False
        self.face_cascade = None

        # Cek apakah DeepFace tersedia
        self._check_deepface()

        # Load face detector untuk fallback/manual detection
        self._load_face_detector()

    def _check_deepface(self):
        """Cek dan inisialisasi DeepFace"""
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.deepface_available = True
            print("[OK] DeepFace loaded - Pre-trained model siap digunakan!")
            print("      Tidak perlu training!")
        except ImportError:
            print("[!] DeepFace belum terinstall")
            print(" Install dengan: pip install deepface")
            self.deepface_available = False

    def _load_face_detector(self):
        """Load OpenCV face cascade classifier"""
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            'data/haarcascade_frontalface_default.xml'
        ]

        for path in cascade_paths:
            try:
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    print(f"[OK] Face detector loaded: {path}")
                    break
            except:
                continue

        if self.face_cascade is None or self.face_cascade.empty():
            print("[!] Warning: Face cascade tidak ditemukan - akan coba download")

    def load_model(self, model_path):
        """Load model dari file .h5 (tidak diperlukan dengan DeepFace)"""
        print("DeepFace mode - tidak perlu load model manual")
        return True

    def create_model(self):
        """Buat model baru (tidak diperlukan dengan DeepFace)"""
        print("DeepFace mode - model sudah pre-trained!")
        return None

    def preprocess_image(self, face_image):
        """
        Preprocessing gambar wajah untuk input model

        Args:
            face_image: Gambar wajah (numpy array)

        Returns:
            Gambar yang sudah dipreprocess
        """
        if face_image is None or face_image.size == 0:
            return None

        # Resize ke 48x48
        face = cv2.resize(face_image, (48, 48))

        # Convert ke grayscale jika belum
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Normalisasi
        face = face.astype('float32') / 255.0

        # Reshape untuk input model (batch, height, width, channels)
        face = face.reshape(1, 48, 48, 1)

        return face

    def detect_faces(self, frame):
        """
        Deteksi wajah dalam frame

        Args:
            frame: Gambar frame dari webcam

        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return faces

    def predict_emotion(self, face_image):
        """
        Prediksi emosi dari gambar wajah menggunakan DeepFace

        Args:
            face_image: Gambar wajah (numpy array)

        Returns:
            Tuple (emotion_label, confidence, probabilities)
        """
        if not self.deepface_available:
            return None, 0, [0] * 7

        try:
            # DeepFace analyze
            result = self.deepface.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'  # lebih cepat
            )

            # Parse hasil
            emotions = result[0]['emotion']

            # Convert ke format standar
            probs = []
            for label in EMOTION_LABELS:
                deep_label = label.lower()
                probs.append(emotions.get(deep_label, 0))

            # Normalize probabilities
            probs = np.array(probs)
            probs = probs / probs.sum() if probs.sum() > 0 else probs

            # Get dominant emotion
            emotion_idx = np.argmax(probs)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = float(probs[emotion_idx])

            return emotion, confidence, probs.tolist()

        except Exception as e:
            # Suppress error print to avoid encoding issues on Windows
            return None, 0, [0] * 7

    def process_frame(self, frame, draw_box=True):
        """
        Process satu frame: detect face dan predict emotion

        Args:
            frame: Frame dari webcam
            draw_box: Apakah menggambar bounding box

        Returns:
            List of dict dengan informasi deteksi
        """
        results = []
        faces = self.detect_faces(frame)

        for (x, y, w, h) in faces:
            # Extract face region
            face_region = frame[y:y+h, x:x+w]

            # Predict emotion
            emotion, confidence, probs = self.predict_emotion(face_region)

            if emotion:
                result = {
                    'bbox': (x, y, w, h),
                    'emotion': emotion,
                    'confidence': confidence,
                    'probabilities': probs
                }
                results.append(result)

                # Draw bounding box
                if draw_box:
                    color = EMOTION_COLORS_BGR.get(emotion, (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                    # Draw label
                    label = f"{emotion}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return results

    def get_model_summary(self):
        """Print ringkasan model"""
        if self.deepface_available:
            print("Using DeepFace pre-trained model")
            print("Models available: VGG-Face, FaceNet, OpenFace, DeepFace, ArcFace")
        else:
            print("DeepFace not available")


if __name__ == "__main__":
    # Test sederhana
    print("EmotionDetector dengan DeepFace (tanpa training)")
    detector = EmotionDetector()
    detector.get_model_summary()