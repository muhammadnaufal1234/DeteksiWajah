# SPEC.md - Deteksi Tingkat Stres Berdasarkan Ekspresi Wajah

## 1. Project Overview

**Project Name:** Stress Detection from Facial Expressions  
**Type:** Computer Vision / Deep Learning Application  
**Core Functionality:** Real-time stress level detection from facial expressions using webcam feed with CNN model  
**Target Users:** Healthcare professionals, researchers, individuals interested in mental health monitoring

## 2. Problem Statement

Stres adalah masalah kesehatan mental yang signifikan. Deteksi dini melalui ekspresi wajah dapat membantu mengidentifikasi individu yang mengalami stres tanpa memerlukan peralatan medis khusus. Project ini menggunakan deep learning untuk menganalisis ekspresi wajah dan mengklasifikasikan tingkat stres.

## 3. Features

### Core Features
- [x] **Real-time Face Detection** - Deteksi wajah secara real-time menggunakan webcam
- [x] **Emotion Classification** - Klasifikasi 7 emosi: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- [x] **Stress Level Detection** - Konversi emosi ke tingkat stres (Rendah, Sedang, Tinggi)
- [x] **GUI Interface** - Antarmuka pengguna dengan visualisasi emosi
- [x] **Webcam Integration** - Akses kamera untuk feed real-time
- [x] **History Tracking** - Riwayat deteksi selama sesi

### Technical Features
- [x] **CNN Model** - Convolutional Neural Network untuk klasifikasi emosi
- [x] **FER2013 Dataset** - Pre-trained model dengan dataset standar FER
- [x] **OpenCV Integration** - Computer vision untuk deteksi wajah
- [x] **TensorFlow/Keras** - Deep learning framework
- [x] **Tkinter GUI** - Antarmuka desktop Python native

## 4. Stress Level Mapping

Emosi dikonversi ke tingkat stres berdasarkan学术界 research:

| Emotion | Stress Level | Score |
|---------|--------------|-------|
| Happy | Rendah (Low) | 1 |
| Surprise | Rendah (Low) | 2 |
| Neutral | Sedang (Medium) | 3 |
| Sad | Sedang (Medium) | 4 |
| Fear | Tinggi (High) | 5 |
| Angry | Tinggi (High) | 6 |
| Disgust | Tinggi (High) | 7 |

##5. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GUI Layer (Tkinter)                   │
├─────────────────────────────────────────────────────────┤
│              Stress Analyzer Engine                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Face │→ │ Emotion │→ │ Stress Level    │ │
│  │ Detector    │  │ Classifier  │  │ Calculator      │ │
│  │ (OpenCV)    │  │ (CNN)       │  │ (Mapping)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│              Data Layer (Training Pipeline)             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ FER2013     │  │ Data │  │ Model │ │
│  │ Dataset     │  │ Augment.    │  │ Training │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 6. File Structure

```
deteksi-wajah-stres/
├── SPEC.md
├── README.md
├── requirements.txt
├── models/
│   └── emotion_model.h5
├── src/
│   ├── __init__.py
│   ├── train_model.py
│   ├── emotion_detector.py
│   ├── stress_analyzer.py
│   └── gui_app.py
├── data/
│   └── (FER2013 dataset - download separately)
└── tests/
    └── test_emotion_detection.py
```

## 7. Technology Stack

- **Python3.8+**
- **TensorFlow 2.x** - Deep learning
- **Keras** - Neural network API
- **OpenCV** - Face detection
- **NumPy** - Numerical operations
- **Pandas** - Data handling
- **Matplotlib** - Visualization
- **Tkinter** - GUI (built-in)

## 8. Acceptance Criteria

1. Aplikasi dapat membuka webcam dan menampilkan feed video
2. Wajah terdeteksi dalam frame dengan bounding box
3. Emosi terklasifikasi dengan akurasi reasonable (>60%)
4. Tingkat stres ditampilkan dengan visualisasi yang jelas
5. GUI responsif dan tidak crash saat penggunaan normal
6. Model dapat di-train dari FER2013 dataset

## 9. Performance Targets

- Face detection: Real-time (>24 FPS)
- Emotion classification: <100ms per frame
- GUI responsiveness: No freezing during detection
- Model accuracy: >60% on validation set
