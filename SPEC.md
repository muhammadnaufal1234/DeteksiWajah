# SPEC.md - Deteksi Tingkat Stres Berdasarkan Ekspresi Wajah

## 1. Project Overview

**Project Name:** Stress Detection from Facial Expressions
**Type:** Computer Vision / Deep Learning Application
**Core Functionality:** Real-time stress level detection from facial expressions using webcam feed
**Target Users:** Healthcare professionals, researchers, individuals interested in mental health monitoring

## 2. Problem Statement

Stres adalah masalah kesehatan mental yang signifikan. Deteksi dini melalui ekspresi wajah dapat membantu mengidentifikasi individu yang mengalami stres tanpa memerlukan peralatan medis khusus. Project ini menggunakan deep learning untuk menganalisis ekspresi wajah dan mengklasifikasikan tingkat stres.

## 3. Features

### Core Features

- [x] **Real-time Face Detection** - Deteksi wajah secara real-time menggunakan webcam
- [x] **Emotion Classification** - Klasifikasi 7 emosi: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- [x] **Stress Level Detection** - Konversi emosi ke tingkat stres (Rendah - Kritis)
- [x] **GUI Interface** - Antarmuka pengguna dengan visualisasi emosi
- [x] **Webcam Integration** - Akses kamera untuk feed real-time
- [x] **History Tracking** - Riwayat deteksi selama sesi
- [x] **Recommendation System** - Saran berdasarkan tingkat stres

### Technical Features

- [x] **DeepFace Integration** - Pre-trained model untuk emotion detection
- [x] **CNN Model** - Convolutional Neural Network untuk klasifikasi emosi (optional training)
- [x] **FER2013 Dataset** - Pre-trained model dengan dataset standar FER
- [x] **OpenCV Integration** - Computer vision untuk deteksi wajah
- [x] **TensorFlow/Keras** - Deep learning framework
- [x] **Tkinter GUI** - Antarmuka desktop Python native

## 4. Stress Level Mapping

Emosi dikonversi ke tingkat stres berdasarkan学术界 research:

| Emotion   | Stress Level     | Score |
|-----------|------------------|-------|
| Happy     | Rendah (Low)     | 1     |
| Surprise  | Rendah-Sedang    | 2     |
| Neutral   | Sedang (Medium)  | 3     |
| Sad       | Sedang-Tinggi    | 4     |
| Fear      | Tinggi (High)    | 5     |
| Angry     | Sangat Tinggi    | 6     |
| Disgust   | Kritis (Critical)| 7     |

## 5. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GUI Layer (Tkinter)                       │
├─────────────────────────────────────────────────────────────┤
│              Stress Analyzer Engine                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Face      │→ │  Emotion    │→ │  Stress Level       │ │
│  │  Detector   │  │ Classifier  │  │    Calculator       │ │
│  │  (OpenCV)   │  │ (DeepFace)  │  │    (Mapping)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│              Configuration Layer (config.py)                 │
│  - EMOTION_LABELS, STRESS_LEVELS                             │
│  - EMOTION_TO_STRESS_SCORE                                   │
│  - APP_THEME, VIDEO_SETTINGS                                 │
└─────────────────────────────────────────────────────────────┘
```

## 6. File Structure

```
deteksi-wajah-stres/
├── main.py                    # Entry point utama
├── SPEC.md                    # Spesifikasi project
├── README.md                  # Dokumentasi
├── requirements.txt           # Dependencies
│
├── src/                       # Source code
│   ├── __init__.py            # Package init dengan exports
│   ├── config.py              # Konfigurasi terpusat (BARU)
│   ├── emotion_detector.py    # DeepFace pre-trained model
│   ├── stress_analyzer.py     # Stress level analysis
│   ├── train_model.py         # Training script
│   ├── gui_app.py             # GUI application
│   └── evaluate.py            # Evaluation script
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_emotion_detection.py
│
├── models/                    # Trained models
│   └── emotion_model.h5
│
├── logs/                      # Training logs
└── data/                      # Dataset ( FER2013)
    └── fer2013.csv
```

## 7. Technology Stack

- **Python 3.8+**
- **TensorFlow 2.x** - Deep learning (optional for custom training)
- **Keras** - Neural network API
- **DeepFace** - Pre-trained emotion detection
- **OpenCV** - Face detection
- **NumPy** - Numerical operations
- **Pandas** - Data handling
- **Matplotlib** - Visualization
- **Tkinter** - GUI (built-in)

## 8. Configuration

Semua konfigurasi terpusat di `src/config.py`:

```python
# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Stress levels
STRESS_LEVELS = {
    1: {'name': 'Rendah', 'color': '#4CAF50'},
    2: {'name': 'Rendah-Sedang', 'color': '#8BC34A'},
    3: {'name': 'Sedang', 'color': '#FFC107'},
    4: {'name': 'Sedang-Tinggi', 'color': '#FF9800'},
    5: {'name': 'Tinggi', 'color': '#F44336'},
    6: {'name': 'Sangat Tinggi', 'color': '#D32F2F'},
    7: {'name': 'Kritis', 'color': '#B71C1C'}
}

# GUI Theme
APP_THEME = {
    'bg_primary': '#1a1a2e',
    'bg_secondary': '#16213e',
    'accent_green': '#4CAF50',
    ...
}
```

## 9. Acceptance Criteria

1. [x] Aplikasi dapat membuka webcam dan menampilkan feed video
2. [x] Wajah terdeteksi dalam frame dengan bounding box
3. [x] Emosi terklasifikasi dengan akurasi reasonable (>60%)
4. [x] Tingkat stres ditampilkan dengan visualisasi yang jelas
5. [x] GUI responsif dan tidak crash saat penggunaan normal
6. [x] Model dapat di-train dari FER2013 dataset (optional)
7. [x] Konfigurasi terpusat di config.py
8. [x] Unit tests tersedia dan berjalan

## 10. Performance Targets

- Face detection: Real-time (>24 FPS)
- Emotion classification: <100ms per frame
- GUI responsiveness: No freezing during detection
- Model accuracy: >60% on validation set (pre-trained DeepFace)

## 11. CLI Commands

| Command                  | Fungsi                          |
|--------------------------|---------------------------------|
| `python main.py`         | Jalankan GUI (default)          |
| `python main.py --gui`   | Jalankan GUI (explicit)         |
| `python main.py --train` | Training model baru             |
| `python main.py --test`  | Jalankan unit tests             |
| `python main.py --evaluate` | Evaluasi dengan kamera       |