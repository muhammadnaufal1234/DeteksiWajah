"""
Emotion Detector Module
=======================
Pre-trained model for emotion detection from facial expressions.
Uses DeepFace library (no training required).

Emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

Author: AI Assistant
"""

import numpy as np
import cv2
import os

# Import configuration
from .config import (
    EMOTION_LABELS,
    EMOTION_COLORS_BGR,
    IMAGE_SIZE
)


class EmotionDetector:
    """
    Class for detecting emotions from facial images.
    Uses DeepFace (pre-trained model - no training required).
    """

    def __init__(self):
        """
        Initialize EmotionDetector with DeepFace.
        """
        self.model = None  # Not used with DeepFace
        self.deepface_available = False
        self.deepface = None
        self.face_cascade = None

        # Initialize components
        self._check_deepface()
        self._load_face_detector()

    def _check_deepface(self) -> None:
        """Check and initialize DeepFace."""
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.deepface_available = True
            print("[OK] DeepFace loaded - Pre-trained model ready!")
            print("     No training needed!")
        except ImportError:
            print("[!] DeepFace not installed")
            print("    Install with: pip install deepface")
            self.deepface_available = False

    def _load_face_detector(self) -> None:
        """Load OpenCV Haar Cascade face detector."""
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
            except Exception:
                continue

        if self.face_cascade is None or self.face_cascade.empty():
            print("[!] Warning: Face cascade not found")

    def load_model(self, model_path: str) -> bool:
        """
        Load model from .h5 file (not required with DeepFace).

        Args:
            model_path: Path to model file (unused, kept for compatibility)

        Returns:
            True if successful
        """
        print("DeepFace mode - no manual model loading needed")
        return True

    def create_model(self):
        """Create new model (not required with DeepFace)."""
        print("DeepFace mode - using pre-trained model!")
        return None

    def preprocess_image(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model input.

        Args:
            face_image: Face image as numpy array

        Returns:
            Preprocessed image
        """
        if face_image is None or face_image.size == 0:
            return None

        # Resize to 48x48
        face = cv2.resize(face_image, IMAGE_SIZE)

        # Convert to grayscale if needed
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Normalize
        face = face.astype('float32') / 255.0

        # Reshape for model input (batch, height, width, channels)
        face = face.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

        return face

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in a frame.

        Args:
            frame: Image frame from webcam

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

    def predict_emotion(self, face_image: np.ndarray) -> tuple:
        """
        Predict emotion from face image using DeepFace.

        Args:
            face_image: Face image as numpy array

        Returns:
            Tuple of (emotion_label, confidence, probabilities)
        """
        if not self.deepface_available:
            return None, 0, [0] * len(EMOTION_LABELS)

        try:
            # DeepFace analyze
            result = self.deepface.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'  # Faster option
            )

            # Parse results
            emotions = result[0]['emotion']

            # Convert to standard format
            probs = []
            for label in EMOTION_LABELS:
                deep_label = label.lower()
                probs.append(emotions.get(deep_label, 0))

            # Normalize probabilities
            probs = np.array(probs)
            total = probs.sum()
            probs = probs / total if total > 0 else probs

            # Get dominant emotion
            emotion_idx = np.argmax(probs)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = float(probs[emotion_idx])

            return emotion, confidence, probs.tolist()

        except Exception:
            return None, 0, [0] * len(EMOTION_LABELS)

    def process_frame(self, frame: np.ndarray, draw_box: bool = True) -> list:
        """
        Process single frame: detect face and predict emotion.

        Args:
            frame: Frame from webcam
            draw_box: Whether to draw bounding box

        Returns:
            List of detection result dictionaries
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
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x, y - label_size[1] - 10),
                        (x + label_size[0], y),
                        color,
                        -1
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        return results

    def get_model_summary(self) -> None:
        """Print model summary."""
        if self.deepface_available:
            print("Using DeepFace pre-trained model")
            print("Available models: VGG-Face, FaceNet, OpenFace, DeepFace, ArcFace")
        else:
            print("DeepFace not available")


def create_cnn_model(input_shape: tuple = (48, 48, 1), num_classes: int = 7):
    """
    Helper function to create CNN model for emotion recognition.
    Delegates to src.train_model.create_model to avoid duplication.

    Args:
        input_shape: Input image shape
        num_classes: Number of emotion classes

    Returns:
        Keras model
    """
    from .train_model import create_model
    return create_model(input_shape, num_classes)


if __name__ == "__main__":
    # Simple test
    print("EmotionDetector with DeepFace (no training required)")
    detector = EmotionDetector()
    detector.get_model_summary()
