# 🧠 Deteksi Tingkat Stres dari Ekspresi Wajah
Project deep learning untuk mendeteksi tingkat stres berdasarkan ekspresi wajah secara real-time menggunakan webcam.

## 📋 Deskripsi
Project ini menggunakan **DeepFace** (pre-trained model) untuk menganalisis ekspresi wajah dan mengklasifikasikan tingkat stres. Aplikasi dapat berjalan secara real-time dengan webcam dan menampilkan hasil deteksi emosi serta tingkat stres.

> ✨ **Tanpa Training!** - Menggunakan model yang sudah dilatih sebelumnya.

## 🎯 Fitur

- **Deteksi Wajah Real-time** - Deteksi wajah menggunakan Haar Cascade
- **Klasifikasi Emosi** - 7 kelas emosi: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Analisis Stres** - Konversi emosi ke tingkat stres (Rendah - Kritis)
- **GUI Interaktif** - Antarmuka desktop dengan Tkinter
- **Visualisasi** - Bar chart untuk probabilitas emosi
- **Rekomendasi** - Saran berdasarkan tingkat stres

## 📊 Mapping Emosi ke Stres

| Emosi | Tingkat Stres | Skor |
|-------|---------------|------|
| Happy | Rendah | 1 |
| Surprise | Rendah-Sedang | 2 |
| Neutral | Sedang | 3 |
| Sad | Sedang-Tinggi | 4 |
| Fear | Tinggi | 5 |
| Angry | Sangat Tinggi | 6 |
| Disgust | Kritis | 7 |

## 🛠️ Instalasi

### 1. Clone/Download Project
```bash
git clone <repository-url>
cd deteksi-wajah-stres
```

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Pre-trained (Otomatis)
DeepFace akan otomatis mengunduh model yang diperlukan saat pertama kali dijalankan.

## 🚀 Penggunaan

### Menjalankan Aplikasi GUI
```bash
python src/gui_app.py
```

### Evaluasi Model (Opsional)
```bash
python src/evaluate.py
```

## 📁 Struktur Project

```
deteksi-wajah-stres/
├── SPEC.md                 # Spesifikasi project
├── README.md               # Dokumentasi
├── requirements.txt       # Dependencies
├── src/
│   ├── __init__.py
│   ├── emotion_detector.py # DeepFace pre-trained model
│   ├── stress_analyzer.py  # Stress level analysis
│   ├── gui_app.py          # GUI application
│   └── evaluate.py         # Evaluation script
└── logs/                   # Logs
```

## 🧠 Teknologi yang Digunakan

- **Python 3.8+**
- **DeepFace** - Pre-trained model untuk emotion detection
- **OpenCV** - Computer vision untuk deteksi wajah
- **Tkinter** - GUI (built-in dengan Python)
- **NumPy** - Numerical operations

## ⚠️ Catatan Penting

1. **Webcam Required** - Aplikasi memerlukan webcam untuk deteksi real-time
2. **Tanpa Training** - DeepFace sudah include pre-trained models
3. **Auto-download** - Model akan didownload otomatis saat pertama kali jalan

## 📝 Lisensi
Project ini bebas digunakan untuk pembelajaran dan penelitian.

## 🙏 Acknowledgments
- Model: [DeepFace](https://github.com/serengil/deepface)
- Face Detection: OpenCV Haar Cascades