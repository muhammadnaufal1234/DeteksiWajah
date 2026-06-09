"""
Stress Analyzer Module
=====================
Modul untuk menganalisis dan mengklasifikasikan tingkat stres
berdasarkan ekspresi wajah yang terdeteksi

Author: AI Assistant
"""

import numpy as np
from collections import deque
from datetime import datetime


class StressAnalyzer:
    """
    Kelas untuk menganalisis tingkat stres dari ekspresi wajah
    """

    # Mapping emosi ke tingkat stres
    STRESS_LEVELS = {
        1: {'name': 'Rendah', 'label': 'Low', 'color': '#4CAF50', 'description': 'Tingkat stres rendah'},
        2: {'name': 'Rendah-Sedang', 'label': 'Low-Medium', 'color': '#8BC34A', 'description': 'Tingkat stres rendah-sedang'},
        3: {'name': 'Sedang', 'label': 'Medium', 'color': '#FFC107', 'description': 'Tingkat stres sedang'},
        4: {'name': 'Sedang-Tinggi', 'label': 'Medium-High', 'color': '#FF9800', 'description': 'Tingkat stres sedang-tinggi'},
        5: {'name': 'Tinggi', 'label': 'High', 'color': '#F44336', 'description': 'Tingkat stres tinggi'},
        6: {'name': 'Sangat Tinggi', 'label': 'Very High', 'color': '#D32F2F', 'description': 'Tingkat stres sangat tinggi'},
        7: {'name': 'Kritis', 'label': 'Critical', 'color': '#B71C1C', 'description': 'Tingkat stres kritis'}
    }

    # Mapping emosi ke skor stres
    EMOTION_STRESS_MAP = {
        'Happy': 1,
        'Surprise': 2,
        'Neutral': 3,
        'Sad': 4,
        'Fear': 5,
        'Angry': 6,
        'Disgust': 7
    }

    def __init__(self, history_size=30):
        """
        Inisialisasi StressAnalyzer

        Args:
            history_size: Jumlah frame yang disimpan untuk analisis temporal
        """
        self.history_size = history_size
        self.emotion_history = deque(maxlen=history_size)
        self.stress_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        self.session_start = datetime.now()
        self.detection_count = 0

    def add_detection(self, emotion, confidence, probabilities=None):
        """
        Tambahkan hasil deteksi emosi

        Args:
            emotion: Label emosi yang terdeteksi
            confidence: Confidence score (0-1)
            probabilities: Array probabilitas semua emosi (optional)
        """
        if emotion is None:
            return

        # Calculate stress score
        stress_score = self.EMOTION_STRESS_MAP.get(emotion, 3)

        # Store in history
        self.emotion_history.append(emotion)
        self.stress_history.append(stress_score)
        self.timestamp_history.append(datetime.now())
        self.detection_count += 1

    def get_current_stress_level(self):
        """
        Dapatkan tingkat stres saat ini berdasarkan rata-rata bergerak

        Returns:
            Dict dengan informasi tingkat stres
        """
        if not self.stress_history:
            return {
                'level': 0,
                'name': 'Tidak Ada Data',
                'label': 'No Data',
                'color': '#9E9E9E',
                'description': 'Belum ada data deteksi',
                'average_score': 0,
                'confidence': 0
            }

        # Calculate average stress score
        avg_score = np.mean(self.stress_history)
        current_score = self.stress_history[-1]

        # Determine stress level
        level = self._score_to_level(avg_score)

        return {
            'level': level,
            'name': self.STRESS_LEVELS[level]['name'],
            'label': self.STRESS_LEVELS[level]['label'],
            'color': self.STRESS_LEVELS[level]['color'],
            'description': self.STRESS_LEVELS[level]['description'],
            'average_score': round(avg_score, 2),
            'current_score': current_score,
            'confidence': confidence if hasattr(self, '_last_confidence') else 0,
            'history_count': len(self.stress_history)
        }

    def get_average_stress(self, last_n=None):
        """
        Dapatkan rata-rata stres dalam N frame terakhir

        Args:
            last_n: Jumlah frame terakhir (default: semua)

        Returns:
            Float rata-rata skor stres
        """
        if not self.stress_history:
            return 0

        if last_n is None:
            return np.mean(self.stress_history)

        history = list(self.stress_history)[-last_n:]
        return np.mean(history) if history else 0

    def get_dominant_emotion(self):
        """
        Dapatkan emosi dominan dalam sesi

        Returns:
            Tuple (emotion, count, percentage)
        """
        if not self.emotion_history:
            return None, 0, 0

        from collections import Counter
        emotion_counts = Counter(self.emotion_history)
        total = len(self.emotion_history)

        dominant = emotion_counts.most_common(1)[0]
        return dominant[0], dominant[1], (dominant[1] / total) * 100

    def get_stress_trend(self):
        """
        Dapatkan tren stres (meningkat, menurun, stabil)

        Returns:
            String tren dan slope
        """
        if len(self.stress_history) < 5:
            return 'insufficient_data', 0

        history = list(self.stress_history)

        # Simple linear regression
        x = np.arange(len(history))
        y = np.array(history)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.1:
            trend = 'increasing'
        elif slope < -0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'

        return trend, slope

    def get_session_summary(self):
        """
        Dapatkan ringkasan sesi deteksi

        Returns:
            Dict dengan ringkasan sesi
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()

        return {
            'session_start': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'session_duration': round(session_duration, 1),
            'total_detections': self.detection_count,
            'average_stress': round(self.get_average_stress(), 2),
            'current_stress': self.get_current_stress_level(),
            'dominant_emotion': self.get_dominant_emotion(),
            'stress_trend': self.get_stress_trend()[0],
            'emotion_distribution': self._get_emotion_distribution()
        }

    def _score_to_level(self, score):
        """
        Konversi skor ke tingkat stres

        Args:
            score: Skor stres (1-7)

        Returns:
            Level (1-7)
        """
        if score <= 1.5:
            return 1
        elif score <= 2.5:
            return 2
        elif score <= 3.5:
            return 3
        elif score <= 4.5:
            return 4
        elif score <= 5.5:
            return 5
        elif score <= 6.5:
            return 6
        else:
            return 7

    def _get_emotion_distribution(self):
        """
        Dapatkan distribusi emosi dalam sesi

        Returns:
            Dict dengan distribusi emosi
        """
        if not self.emotion_history:
            return {}

        from collections import Counter
        total = len(self.emotion_history)
        counts = Counter(self.emotion_history)

        return {
            emotion: {
                'count': count,
                'percentage': round((count / total) * 100, 1)
            }
            for emotion, count in counts.items()
        }

    def reset(self):
        """Reset semua history"""
        self.emotion_history.clear()
        self.stress_history.clear()
        self.timestamp_history.clear()
        self.session_start = datetime.now()
        self.detection_count = 0

    def get_stress_percentage(self):
        """
        Dapatkan persentase tingkat stres (untuk visualisasi)

        Returns:
            Dict dengan persentase untuk setiap level
        """
        if not self.stress_history:
            return {i: 0 for i in range(1, 8)}

        from collections import Counter
        counts = Counter(self.stress_history)
        total = len(self.stress_history)

        return {
            level: round((counts.get(level, 0) / total) * 100, 1)
            for level in range(1, 8)
        }

    def get_recommendation(self):
        """
        Dapatkan rekomendasi berdasarkan tingkat stres

        Returns:
            String rekomendasi
        """
        current = self.get_current_stress_level()
        level = current['level']

        recommendations = {
            1: "🎉 Kondisi prima! Tetap jaga pola hidup sehat.",
            2: "😊 Kondisi baik. Lanjutkan aktivitas seperti biasa.",
            3: "📋 Perhatikan tanda-tanda stres. Istirahat jika perlu.",
            4: "💆 Lakukan relaksasi, coba tarik napas dalam.",
            5: "⚠️ Tingkat stres meningkat. Luangkan waktu untuk relaksasi.",
            6: "🚨 Disarankan untuk berkonsultasi dengan profesional.",
            7: "🏥 Perlu perhatian segera. Segera cari bantuan profesional."
        }

        return recommendations.get(level, "Tidak ada rekomendasi tersedia.")


if __name__ == "__main__":
    # Test sederhana
    analyzer = StressAnalyzer()

    # Simulasi deteksi
    test_emotions = ['Happy', 'Happy', 'Neutral', 'Sad', 'Fear', 'Angry', 'Angry']

    for emotion in test_emotions:
        analyzer.add_detection(emotion, 0.8)

    print("=== Stress Analysis Results ===")
    print(f"Current Stress Level: {analyzer.get_current_stress_level()}")
    print(f"Average Stress: {analyzer.get_average_stress()}")
    print(f"Dominant Emotion: {analyzer.get_dominant_emotion()}")
    print(f"Stress Trend: {analyzer.get_stress_trend()}")
    print(f"Recommendation: {analyzer.get_recommendation()}")
    print(f"Session Summary: {analyzer.get_session_summary()}")