"""
Evaluation Script untuk DeepFace Emotion Detection
===================================================
Tidak perlu training - hanya evaluasi pre-trained model
"""

import cv2
import numpy as np
from src.emotion_detector import EmotionDetector, EMOTION_LABELS


def evaluate_on_camera():
    """Evaluasi model dengan webcam"""
    print("=" * 50)
    print("Evaluasi DeepFace Emotion Detector")
    print("=" * 50)

    # Initialize detector
    detector = EmotionDetector()

    if not detector.deepface_available:
        print("ERROR: DeepFace tidak tersedia. Install dengan:")
        print("pip install deepface")
        return

    # Buka webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Tidak dapat membuka webcam")
        return

    print("\nTekan 'q' untuk quit")
    print("Tekan 's' untuk screenshot")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        results = detector.process_frame(frame, draw_box=True)

        # Tampilkan info
        if results:
            for r in results:
                print(f"Emotion: {r['emotion']} ({r['confidence']:.2f})")

        # Tampilkan frame
        cv2.imshow('Emotion Detection - DeepFace (Pre-trained)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{np.random.randint(1000)}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nEvaluasi selesai!")


if __name__ == "__main__":
    evaluate_on_camera()