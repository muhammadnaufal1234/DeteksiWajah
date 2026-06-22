# 📚 DOKUMENTASI PROJECT
# Deteksi Tingkat Stres dari Ekspresi Wajah

## 1. IDENTITAS PROJECT

| Item | Detail |
|------|--------|
| **Judul** | Deteksi Tingkat Stres dari Ekspresi Wajah Menggunakan Deep Learning |
| **Nama Program** | Facial Expression Stress Detection |
| **Teknologi** | Python, DeepFace, OpenCV, Tkinter |
| **Jenis** | Computer Vision / Deep Learning |
| **Penggunaan** | Real-time emotion & stress detection via webcam |

---

## 2. SOURCE CODE PROGRAM

### 2.1 Struktur Direktori

```
deteksi-wajah-stres/
│
├── main.py                    # Entry point utama
├── SPEC.md                    # Spesifikasi teknis
├── README.md                  # Dokumentasi pengguna
├── requirements.txt           # Dependencies
│
├── src/                       # Kode sumber utama
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Konfigurasi terpusat
│   ├── emotion_detector.py     # Modul deteksi emosi
│   ├── stress_analyzer.py      # Modul analisis stres
│   ├── gui_app.py             # Aplikasi GUI
│   ├── train_model.py          # Script training model
│   └── evaluate.py            # Script evaluasi
│
├── tests/                     # Unit tests
│   └── test_emotion_detection.py
│
├── models/                    # Model terlatih
│   └── emotion_model.h5
│
├── logs/                      # Training logs
│   └── training_log_*.csv
│
├── data/                      # Dataset
│   └── fer2013.csv
│
└── fer2013.csv               # Dataset (root)
```

### 2.2 Kode Utama - Konfigurasi (src/config.py)

```python
"""
Configuration Module
====================
Konfigurasi terpusat untuk aplikasi deteksi stres.

Author: AI Assistant
"""

# =============================================================================
# Emotion Configuration
# =============================================================================

# Label emosi sesuai standar FER2013 dataset
EMOTION_LABELS = [
    'Angry',    # 0 - Marah
    'Disgust',  # 1 - Jijik
    'Fear',     # 2 - Takut
    'Happy',    # 3 - Senang
    'Sad',      # 4 - Sedih
    'Surprise', # 5 - Terkejut
    'Neutral'   # 6 - Netral
]

# Mapping emosi ke skor stres (1=Rendah, 7=Kritis)
EMOTION_TO_STRESS_SCORE = {
    'Happy':     1,   #happy,     lowest stress
    'Surprise':  2,
    'Neutral':   3,
    'Sad':       4,
    'Fear':      5,
    'Angry':     6,
    'Disgust':   7    # disgust,   highest stress
}

# Warna emosi untuk visualisasi (BGR - OpenCV)
EMOTION_COLORS_BGR = {
    'Angry':     (0, 0, 255),       # Merah
    'Disgust':   (144, 0, 255),     # Ungu
    'Fear':      (255, 0, 255),     # Magenta
    'Happy':     (0, 255, 0),       # Hijau
    'Sad':       (255, 0, 0),       # Biru
    'Surprise':  (0, 255, 255),     # Kuning
    'Neutral':   (128, 128, 128)    # Abu-abu
}

# Warna emosi untuk Tkinter (HEX)
EMOTION_COLORS_HEX = {
    'Angry':     '#FF0000',
    'Disgust':   '#9000FF',
    'Fear':      '#FF00FF',
    'Happy':     '#00FF00',
    'Sad':       '#0000FF',
    'Surprise':  '#FFFF00',
    'Neutral':   '#808080'
}

# =============================================================================
# Stress Level Configuration
# =============================================================================

STRESS_LEVELS = {
    1: {'name': 'Rendah',       'color': '#4CAF50', 'description': 'Kondisi prima'},
    2: {'name': 'Rendah-Sedang','color': '#8BC34A', 'description': 'Kondisi baik'},
    3: {'name': 'Sedang',       'color': '#FFC107', 'description': 'Perlu perhatian'},
    4: {'name': 'Sedang-Tinggi','color': '#FF9800', 'description': 'Relaksasi diperlukan'},
    5: {'name': 'Tinggi',       'color': '#F44336', 'description': 'Stres meningkat'},
    6: {'name': 'Sangat Tinggi','color': '#D32F2F', 'description': 'Konsultasi disarankan'},
    7: {'name': 'Kritis',       'color': '#B71C1C', 'description': 'Bantuan segera'}
}

# Rekomendasi berdasarkan tingkat stres
STRESS_RECOMMENDATIONS = {
    1: "🎉 Kondisi prima! Tetap jaga pola hidup sehat.",
    2: "😊 Kondisi baik. Lanjutkan aktivitas seperti biasa.",
    3: "📋 Perhatikan tanda-tanda stres. Istirahat jika perlu.",
    4: "💆 Lakukan relaksasi, coba tarik napas dalam.",
    5: "⚠️ Tingkat stres meningkat. Luangkan waktu untuk relaksasi.",
    6: "🚨 Disarankan untuk berkonsultasi dengan profesional.",
    7: "🏥 Perlu perhatian segera. Segera cari bantuan profesional."
}
```

### 2.3 Kode Utama - Emotion Detector (src/emotion_detector.py)

```python
"""
Emotion Detector Module
======================
Modul untuk mendeteksi emosi dari ekspresi wajah.
Menggunakan DeepFace (pre-trained model - tanpa training).

Author: AI Assistant
"""

import numpy as np
import cv2
from .config import EMOTION_LABELS, EMOTION_COLORS_BGR, IMAGE_SIZE


class EmotionDetector:
    """
    Kelas untuk mendeteksi emosi dari gambar wajah.
    Menggunakan DeepFace (pre-trained model).
    """

    def __init__(self):
        """Inisialisasi EmotionDetector dengan DeepFace."""
        self.deepface_available = False
        self.deepface = None
        self.face_cascade = None

        # Cek DeepFace
        self._check_deepface()
        # Load face detector
        self._load_face_detector()

    def _check_deepface(self):
        """Cek dan inisialisasi DeepFace."""
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.deepface_available = True
            print("[OK] DeepFace loaded!")
        except ImportError:
            print("[!] DeepFace not installed")
            self.deepface_available = False

    def _load_face_detector(self):
        """Load Haar Cascade untuk deteksi wajah."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"[OK] Face detector loaded")

    def detect_faces(self, frame):
        """
        Deteksi wajah dalam frame.

        Args:
            frame: Gambar dari webcam

        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
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
        Prediksi emosi dari gambar wajah.

        Args:
            face_image: Crop wajah (numpy array)

        Returns:
            Tuple (emotion, confidence, probabilities)
        """
        if not self.deepface_available:
            return None, 0, [0] * 7

        try:
            result = self.deepface.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            emotions = result[0]['emotion']
            probs = [emotions.get(label.lower(), 0) for label in EMOTION_LABELS]
            probs = np.array(probs)
            probs = probs / probs.sum() if probs.sum() > 0 else probs

            emotion_idx = np.argmax(probs)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = float(probs[emotion_idx])

            return emotion, confidence, probs.tolist()

        except Exception:
            return None, 0, [0] * 7

    def process_frame(self, frame, draw_box=True):
        """
        Proses satu frame: deteksi wajah + prediksi emosi.

        Args:
            frame: Frame dari webcam
            draw_box: Gambar bounding box atau tidak

        Returns:
            List of detection results
        """
        results = []
        faces = self.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            emotion, confidence, probs = self.predict_emotion(face_region)

            if emotion:
                result = {
                    'bbox': (x, y, w, h),
                    'emotion': emotion,
                    'confidence': confidence,
                    'probabilities': probs
                }
                results.append(result)

                # Gambar bounding box
                if draw_box:
                    color = EMOTION_COLORS_BGR.get(emotion, (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return results
```

### 2.4 Kode Utama - Stress Analyzer (src/stress_analyzer.py)

```python
"""
Stress Analyzer Module
=====================
Modul untuk menganalisis tingkat stres dari ekspresi wajah.

Author: AI Assistant
"""

import numpy as np
from collections import deque, Counter
from datetime import datetime
from .config import EMOTION_TO_STRESS_SCORE, STRESS_LEVELS


class StressAnalyzer:
    """Kelas untuk menganalisis tingkat stres."""

    def __init__(self, history_size=30):
        """
        Inisialisasi StressAnalyzer.

        Args:
            history_size: Jumlah frame untuk analisis temporal
        """
        self.history_size = history_size
        self.emotion_history = deque(maxlen=history_size)
        self.stress_history = deque(maxlen=history_size)
        self.session_start = datetime.now()
        self.detection_count = 0

    def add_detection(self, emotion, confidence, probabilities=None):
        """Tambahkan hasil deteksi emosi."""
        if emotion is None:
            return

        stress_score = EMOTION_TO_STRESS_SCORE.get(emotion, 3)
        self.emotion_history.append(emotion)
        self.stress_history.append(stress_score)
        self.detection_count += 1

    def get_current_stress_level(self):
        """Dapatkan tingkat stres saat ini (rata-rata bergerak)."""
        if not self.stress_history:
            return {
                'level': 0,
                'name': 'Tidak Ada Data',
                'color': '#9E9E9E'
            }

        avg_score = np.mean(self.stress_history)
        level = self._score_to_level(avg_score)

        return {
            'level': level,
            'name': STRESS_LEVELS[level]['name'],
            'color': STRESS_LEVELS[level]['color'],
            'description': STRESS_LEVELS[level]['description'],
            'average_score': round(avg_score, 2)
        }

    def _score_to_level(self, score):
        """Konversi skor ke tingkat (1-7)."""
        if score <= 1.5:   return 1
        elif score <= 2.5:  return 2
        elif score <= 3.5:  return 3
        elif score <= 4.5:  return 4
        elif score <= 5.5:  return 5
        elif score <= 6.5:  return 6
        else:               return 7

    def get_recommendation(self):
        """Dapatkan rekomendasi berdasarkan tingkat stres."""
        from .config import STRESS_RECOMMENDATIONS
        level = self.get_current_stress_level()['level']
        return STRESS_RECOMMENDATIONS.get(level, "Tidak ada rekomendasi.")
```

### 2.5 Kode Utama - GUI Application (src/gui_app.py)

```python
"""
GUI Application untuk Deteksi Stres
===================================
Aplikasi desktop dengan Tkinter untuk deteksi real-time.

Author: AI Assistant
"""

import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
from src.emotion_detector import EmotionDetector
from src.stress_analyzer import StressAnalyzer
from src.config import EMOTION_LABELS, EMOTION_COLORS_HEX


class StressDetectionGUI:
    """GUI untuk deteksi stres dari ekspresi wajah."""

    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Tingkat Stres")
        self.root.geometry("1200x800")

        # Inisialisasi komponen
        self.emotion_detector = EmotionDetector()
        self.stress_analyzer = StressAnalyzer(history_size=30)
        self.is_running = False

        # Build UI
        self._create_widgets()
        self._start_video()

    def _create_widgets(self):
        """Buat komponen GUI."""
        # Panel kiri: Video feed
        # Panel kanan: Hasil deteksi (stress level, emosi, rekomendasi)

    def _video_loop(self):
        """Loop pemrosesan video."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Mirror frame
            frame = cv2.flip(frame, 1)

            # Proses deteksi
            results = self.emotion_detector.process_frame(frame, draw_box=True)

            # Update analyzer
            for result in results:
                self.stress_analyzer.add_detection(
                    result['emotion'],
                    result['confidence'],
                    result['probabilities']
                )

            # Update display
            self.current_frame = frame.copy()
            self.root.after(1, self._update_display, results)

    def _update_display(self, results):
        """Update tampilan GUI."""
        # Convert frame untuk display
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (480, 360))
        image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.video_label.config(image=image)
        self.video_label.image = image

        # Update stress level
        stress_info = self.stress_analyzer.get_current_stress_level()
        self.stress_label.config(
            text=stress_info['name'],
            fg=stress_info['color']
        )

        # Update emotion bars
        if results:
            probs = results[0]['probabilities']
            for i, emotion in enumerate(EMOTION_LABELS):
                # Update progress bar
                ...
```

---

## 3. DATASET YANG DIGUNAKAN

### 3.1 FER2013 Dataset

| Property | Detail |
|----------|--------|
| **Nama** | FER2013 (Facial Expression Recognition 2013) |
| **Sumber** | Kaggle: https://www.kaggle.com/datasets/deadskull7/fer2013 |
| **Jumlah Data** | 35,887 gambar |
| **Format** | CSV (pixels 48x48 grayscale) |
| **Kelas Emosi** | 7 kelas |

### 3.2 Struktur Dataset FER2013

```
fer2013.csv
├── Usage     # Pembagian data (Training/PublicTest/PrivateTest)
├── emotion   # Label emosi (0-6)
└── pixels    # 2304 nilai pixel (48x48 grayscale)
```

### 3.3 Distribusi Emosi

| Label | Emosi   | Jumlah (approx) | Persentase |
|-------|---------|-----------------|------------|
| 0     | Angry   | 4,953           | ~13.8%     |
| 1     | Disgust | 547             | ~1.5%      |
| 2     | Fear    | 5,121           | ~14.3%     |
| 3     | Happy   | 8,989           | ~25.0%     |
| 4     | Sad     | 6,074           | ~16.9%     |
| 5     | Surprise| 4,002           | ~11.1%     |
| 6     | Neutral | 6,198           | ~17.3%     |

### 3.4 Contoh Format Data

```csv
Usage,emotion,pixels
Training,3,70 80 82 72 58 58 100 100 ...
Training,0,126 126 126 127 127 125 122 ...
PublicTest,6,255 255 255 254 254 253 ...
```

### 3.5 Pre-trained Model (DeepFace)

| Model | Deskripsi | Akurasi |
|-------|-----------|---------|
| VGG-Face | CNN-based face recognition | ~98% |
| FaceNet | Google's face embedding | ~99.6% |
| ArcFace | State-of-the-art | ~99.8% |

> **Catatan:** Project ini menggunakan **DeepFace** dengan model pre-trained, sehingga tidak memerlukan training tambahan. Model otomatis didownload saat pertama kali dijalankan.

---

## 4. DOKUMENTASI PENGOLAHAN CITRA

### 4.1 Diagram Alur Pengolahan Citra

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE PENGOLAHAN CITRA                            │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐      ┌──────────────┐      ┌─────────────────┐
    │  INPUT   │ ───► │  PREPROCESS  │ ───► │   FACE DETECT   │
    │  Frame   │      │  Conversion  │      │  Haar Cascade   │
    │  Webcam  │      │  Grayscale   │      │   OpenCV        │
    └──────────┘      └──────────────┘      └────────┬────────┘
                                                     │
                                                     ▼
                    ┌───────────────────────────────────────────────┐
                    │              CROP FACE REGION                 │
                    │         Extract (x,y,w,h) from frame          │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
    ┌──────────────┐      ┌─────────────────┐      ┌──────────────┐
    │   OUTPUT     │ ◄─── │   PREDICT       │ ◄─── │   RESIZE     │
    │  Display &   │      │   EMOTION       │      │   48x48      │
    │  Analysis     │      │   (DeepFace)    │      │   Grayscale  │
    └──────────────┘      └─────────────────┘      └──────────────┘
```

### 4.2 Tahapan Pengolahan Citra

#### TAHAP 1: INPUT (Pemasukan Citra)

```
┌─────────────────────────────────────────┐
│           INPUT: Frame Webcam           │
├─────────────────────────────────────────┤
│                                         │
│   📷 Capture Frame (640x480)            │
│   Format: BGR (OpenCV default)          │
│   FPS: ~30                              │
│                                         │
│   ┌───────────────────────────────┐    │
│   │                               │    │
│   │      Raw Camera Feed         │    │
│   │                               │    │
│   │   ┌─────────┐               │    │
│   │   │  Wajah  │  ← Target     │    │
│   │   │  Person │               │    │
│   │   └─────────┘               │    │
│   │                               │    │
│   └───────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

**Proses:**
1. Capture frame dari webcam menggunakan OpenCV (`cv2.VideoCapture`)
2. Frame berformat BGR dengan dimensi default 640x480
3. Frame di-mirror (flip horizontal) untuk pengalaman yang lebih natural

#### TAHAP 2: PREPROCESSING

```
┌─────────────────────────────────────────┐
│         PREPROCESSING: Grayscale         │
├─────────────────────────────────────────┤
│                                         │
│   Input: BGR Color Image (640x480)      │
│                 │                        │
│                 ▼                        │
│   ┌─────────────────────────────┐       │
│   │   cv2.cvtColor(frame,      │       │
│   │    cv2.COLOR_BGR2GRAY)     │       │
│   └─────────────────────────────┘       │
│                 │                        │
│                 ▼                        │
│   Output: Grayscale Image (640x480)      │
│           Single channel (0-255)        │
│                                         │
│   ┌───────────────────────────────┐    │
│   │                               │    │
│   │      Grayscale Frame         │    │
│   │      (Wajah lebih jelas       │    │
│   │       untuk deteksi fitur)   │    │
│   │                               │    │
│   └───────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

**Proses:**
1. Konversi BGR → Grayscale
2. Mengapa grayscale? Haar Cascade bekerja lebih baik dengan single channel
3. Mengurangi noise dan fokus pada struktur wajah

#### TAHAP 3: FACE DETECTION

```
┌─────────────────────────────────────────┐
│      FACE DETECTION: Haar Cascade        │
├─────────────────────────────────────────┤
│                                         │
│   Grayscale Image ──► detectMultiScale  │
│                              │           │
│                              ▼           │
│   ┌─────────────────────────────────┐  │
│   │  Haar Cascade Classifier         │  │
│   │  XML: haarcascade_frontalface_  │  │
│   │         default.xml              │  │
│   └─────────────────────────────────┘  │
│                              │           │
│                              ▼           │
│   Output: List of Bounding Boxes        │
│   [(x, y, w, h), (x, y, w, h), ...]    │
│                                         │
│   ┌───────────────────────────────┐    │
│   │                               │    │
│   │    ┌───────────────────┐     │    │
│   │    │                   │     │    │
│   │    │   DETECTED FACE   │     │    │
│   │    │   (x, y, w, h)   │     │    │
│   │    │                   │     │    │
│   │    └───────────────────┘     │    │
│   │                               │    │
│   └───────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

**Parameter Haar Cascade:**
```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,    # Scale image pyramid
    minNeighbors=5,     # Min neighbors for detection
    minSize=(30, 30)    # Minimum face size
)
```

#### TAHAP 4: FACE CROP & RESIZE

```
┌─────────────────────────────────────────┐
│           FACE CROP & RESIZE            │
├─────────────────────────────────────────┤
│                                         │
│   Full Frame ──► [x:y+h, x:x+w] ──► Crop│
│                  (Face Region)           │
│                                         │
│   ┌─────────┐    ┌─────────┐           │
│   │ Original│    │  Crop   │           │
│   │ Frame   │───►│  Face   │           │
│   └─────────┘    │ Region  │           │
│   (640x480)      └────┬────┘           │
│                       │                 │
│                       ▼                 │
│              ┌────────────────┐        │
│              │ cv2.resize     │        │
│              │ (48, 48)       │        │
│              └────────┬───────┘        │
│                       │                 │
│                       ▼                 │
│              ┌────────────────┐        │
│              │ Resized Face  │        │
│              │    48x48      │        │
│              │  Grayscale    │        │
│              └────────────────┘        │
│                                         │
└─────────────────────────────────────────┘
```

**Proses:**
```python
# Crop face region
face_region = frame[y:y+h, x:x+w]

# Resize for model input
face_resized = cv2.resize(face_region, (48, 48))
```

#### TAHAP 5: EMOTION PREDICTION (DeepFace)

```
┌─────────────────────────────────────────┐
│        EMOTION PREDICTION: DeepFace      │
├─────────────────────────────────────────┤
│                                         │
│   Resized Face (48x48)                   │
│          │                               │
│          ▼                               │
│   ┌─────────────────────────────────┐   │
│   │     DeepFace.analyze()          │   │
│   │  - Model: VGG-Face / etc.      │   │
│   │  - Actions: ['emotion']        │   │
│   │  - Detector: opencv            │   │
│   └─────────────────────────────────┘   │
│          │                               │
│          ▼                               │
│   ┌─────────────────────────────────┐   │
│   │     Emotion Probabilities       │   │
│   │  ┌───────────────────────────┐  │   │
│   │  │ Angry:     0.05 (5%)      │  │   │
│   │  │ Disgust:   0.02 (2%)      │  │   │
│   │  │ Fear:      0.08 (8%)      │  │   │
│   │  │ Happy:     0.75 (75%)  ◄──│  │   │
│   │  │ Sad:       0.03 (3%)      │  │   │
│   │  │ Surprise:  0.05 (5%)      │  │   │
│   │  │ Neutral:   0.02 (2%)      │  │   │
│   │  └───────────────────────────┘  │   │
│   └─────────────────────────────────┘   │
│          │                               │
│          ▼                               │
│   Output: ("Happy", 0.75, [0.05, ...])  │
│                                         │
└─────────────────────────────────────────┘
```

**Proses DeepFace:**
```python
result = DeepFace.analyze(
    face_image,
    actions=['emotion'],
    enforce_detection=False,
    detector_backend='opencv'
)

# Parse results
emotions = result[0]['emotion']
dominant_emotion = result[0]['dominant_emotion']
```

#### TAHAP 6: STRESS ANALYSIS

```
┌─────────────────────────────────────────┐
│          STRESS ANALYSIS                 │
├─────────────────────────────────────────┤
│                                         │
│   Emotion Detection Result               │
│         │                                │
│         ▼                                │
│   ┌─────────────────────────────────┐   │
│   │   EMOTION → STRESS MAPPING     │   │
│   │  ┌────────────────────────────┐ │   │
│   │  │ Happy     → Score: 1      │ │   │
│   │  │ Surprise  → Score: 2      │ │   │
│   │  │ Neutral   → Score: 3      │ │   │
│   │  │ Sad       → Score: 4      │ │   │
│   │  │ Fear      → Score: 5      │ │   │
│   │  │ Angry     → Score: 6      │ │   │
│   │  │ Disgust   → Score: 7      │ │   │
│   │  └────────────────────────────┘ │   │
│   └─────────────────────────────────┘   │
│         │                                │
│         ▼                                │
│   ┌─────────────────────────────────┐   │
│   │   TEMPORAL ANALYSIS            │   │
│   │   (Moving Average - 30 frames) │   │
│   │                                │   │
│   │   Stress Scores: [1, 1, 3, 4] │   │
│   │   Average: 2.25                │   │
│   │   Level: Rendah-Sedang (2)    │   │
│   └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

#### TAHAP 7: OUTPUT (Tampilan Hasil)

```
┌─────────────────────────────────────────┐
│                OUTPUT                    │
├─────────────────────────────────────────┤
│                                         │
│   ┌─────────────────┬─────────────────┐│
│   │                 │  TINGKAT STRES   ││
│   │   CAMERA FEED   │     Rendah       ││
│   │   + Face Box    │   (Skor: 2.1)    ││
│   │   + Emotion     │                  ││
│   │                 │  DETEKSI EMOSI    ││
│   │   ┌─────────┐   │  Happy     75%  ││
│   │   │ Happy 😊│   │  Sad        5%   ││
│   │   │ : 0.85  │   │  Angry      3%   ││
│   │   └─────────┘   │  Fear       8%   ││
│   │                 │  ...             ││
│   │                 │                  ││
│   │                 │  REKOMENDASI     ││
│   │                 │  "Kondisi baik!  ││
│   │                 │   Lanjutkan..."  ││
│   └─────────────────┴─────────────────┘│
│                                         │
└─────────────────────────────────────────┘
```

### 4.3 Ringkasan Input-Process-Output

```
╔═══════════════════════════════════════════════════════════════════╗
║                         INPUT                                     ║
╠═══════════════════════════════════════════════════════════════════╣
║  Sumber: Webcam / Kamera                                          ║
║  Format: Frame BGR (640x480)                                      ║
║  Tipe: Real-time video stream (~30 FPS)                           ║
║  Contoh: Citra wajah manusia dari webcam                          ║
╚═══════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════╗
║                         PROCESS                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  Step 1: Preprocessing                                           ║
║          - Konversi BGR → Grayscale                              ║
║          - Mirror/Flip horizontal                                ║
║                                                                   ║
║  Step 2: Face Detection                                           ║
║          - Haar Cascade Classifier (OpenCV)                       ║
║          - Output: Bounding box (x, y, w, h)                     ║
║                                                                   ║
║  Step 3: Face Preprocessing                                       ║
║          - Crop face region                                      ║
║          - Resize ke 48x48                                      ║
║                                                                   ║
║  Step 4: Emotion Recognition                                      ║
║          - DeepFace Pre-trained Model                            ║
║          - Output: 7 probability scores                          ║
║          - Dominant emotion selection                            ║
║                                                                   ║
║  Step 5: Stress Analysis                                          ║
║          - Map emotion → stress score (1-7)                      ║
║          - Moving average (30 frames)                            ║
║          - Determine stress level                                ║
╚═══════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════╗
║                         OUTPUT                                    ║
╠═══════════════════════════════════════════════════════════════════╣
║  Visual:                                                        ║
║  - Video feed dengan bounding box wajah                          ║
║  - Label emosi pada setiap wajah terdeteksi                      ║
║  - Progress bar probabilitas setiap emosi                        ║
║  - Indikator tingkat stres (warna + teks)                        ║
║  - Rekomendasi berdasarkan stres                                 ║
║                                                                   ║
║  Data:                                                           ║
║  - Emotion label: "Happy", "Sad", dll                           ║
║  - Confidence: 0.0 - 1.0                                        ║
║  - Probabilities: [p1, p2, ..., p7]                             ║
║  - Stress level: 1-7                                            ║
║  - Recommendation: string                                       ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 5. FLOWCHART SISTEM

```
                    ┌──────────────────┐
                    │   START          │
                    │   Buka Webcam    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Capture Frame │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
              ┌─────│  Wajah Terdeteksi?│
              │     └────────┬─────────┘
              │ NO          │ YES
              │             ▼
              │    ┌──────────────────┐
              │    │  Crop Face Region │
              │    └────────┬─────────┘
              │             ▼
              │    ┌──────────────────┐
              │    │ DeepFace Analyze │
              │    │ Emotion Detection │
              │    └────────┬─────────┘
              │             ▼
              │    ┌──────────────────┐
              │    │  Update Stress   │
              │    │    Analyzer     │
              │    └────────┬─────────┘
              │             ▼
              │    ┌──────────────────┐
              │    │  Update GUI      │
              │    │  Display Results │
              │    └────────┬─────────┘
              │             │
              │             └────────────┐
              │                          │
              │             ┌────────────┘
              │             │
              ▼             ▼
    ┌──────────────────┐    │
    │  Tampilkan:      │    │
    │  "Tidak ada      │    │
    │   wajah"         │    │
    └────────┬─────────┘    │
             │              │
             └──────┬───────┘
                    │
                    ▼
           ┌──────────────────┐
           │  Tekan 'q'       │───YES──► END
           │  untuk quit?     │
           └────────┬─────────┘
                    │ NO
                    │
                    └──────────► (kembali ke Capture Frame)
```

---

## 6. TEKNOLOGI YANG DIGUNAKAN

| Teknologi | Fungsi | Versi |
|-----------|--------|-------|
| Python | Bahasa pemrograman | 3.8+ |
| OpenCV | Face detection, image processing | 4.5+ |
| DeepFace | Pre-trained emotion recognition | 0.0.89+ |
| TensorFlow | Deep learning framework | 2.x |
| Tkinter | GUI (built-in Python) | - |
| NumPy | Numerical operations | 1.21+ |

---

## 7. PETUNJUK PENGGUNAAN

### Menjalankan Aplikasi

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan aplikasi
python main.py

# 3. Atau dengan command explicit
python main.py --gui
```

### Menjalankan Unit Tests

```bash
python main.py --test
```

### Training Model (Optional)

```bash
python main.py --train --epochs 50
```

---

*Document Generated: 2026*
*Author: AI Assistant*
