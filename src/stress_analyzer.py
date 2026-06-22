"""
Stress Analyzer Module
======================
Module for analyzing and classifying stress levels
based on detected facial expressions.

Author: AI Assistant
"""

import numpy as np
from collections import deque, Counter
from datetime import datetime

# Import configuration
from .config import (
    EMOTION_TO_STRESS_SCORE,
    STRESS_LEVELS,
    DEFAULT_HISTORY_SIZE,
    MIN_DETECTIONS_FOR_TREND,
    get_stress_level,
    get_recommendation
)


class StressAnalyzer:
    """
    Class for analyzing stress levels from facial expressions.
    Uses temporal analysis with a moving average approach.
    """

    def __init__(self, history_size: int = DEFAULT_HISTORY_SIZE):
        """
        Initialize StressAnalyzer.

        Args:
            history_size: Number of frames to store for temporal analysis
        """
        self.history_size = history_size
        self.emotion_history = deque(maxlen=history_size)
        self.stress_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        self.session_start = datetime.now()
        self.detection_count = 0
        self._last_confidence = 0

    def add_detection(self, emotion: str, confidence: float, probabilities: list = None) -> None:
        """
        Add emotion detection result.

        Args:
            emotion: Detected emotion label
            confidence: Confidence score (0-1)
            probabilities: Array of all emotion probabilities (optional, reserved for future use)
        """
        if emotion is None:
            return

        # Calculate stress score from emotion
        stress_score = EMOTION_TO_STRESS_SCORE.get(emotion, 3)

        # Store in history
        self.emotion_history.append(emotion)
        self.stress_history.append(stress_score)
        self.timestamp_history.append(datetime.now())
        self.detection_count += 1
        self._last_confidence = confidence

    def get_current_stress_level(self) -> dict:
        """
        Get current stress level based on moving average.

        Returns:
            Dictionary with stress level information
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
        level = get_stress_level(avg_score)

        return {
            'level': level,
            'name': STRESS_LEVELS[level]['name'],
            'label': STRESS_LEVELS[level]['label'],
            'color': STRESS_LEVELS[level]['color'],
            'description': STRESS_LEVELS[level]['description'],
            'average_score': round(avg_score, 2),
            'current_score': current_score,
            'confidence': self._last_confidence,
            'history_count': len(self.stress_history)
        }

    def get_average_stress(self, last_n: int = None) -> float:
        """
        Get average stress over N frames.

        Args:
            last_n: Number of last frames (default: all)

        Returns:
            Average stress score
        """
        if not self.stress_history:
            return 0

        if last_n is None:
            return np.mean(self.stress_history)

        history = list(self.stress_history)[-last_n:]
        return np.mean(history) if history else 0

    def get_dominant_emotion(self) -> tuple:
        """
        Get dominant emotion in the session.

        Returns:
            Tuple of (emotion, count, percentage)
        """
        if not self.emotion_history:
            return None, 0, 0

        emotion_counts = Counter(self.emotion_history)
        total = len(self.emotion_history)

        dominant = emotion_counts.most_common(1)[0]
        return dominant[0], dominant[1], (dominant[1] / total) * 100

    def get_stress_trend(self) -> tuple:
        """
        Get stress trend (increasing, decreasing, stable).

        Returns:
            Tuple of (trend_string, slope)
        """
        if len(self.stress_history) < MIN_DETECTIONS_FOR_TREND:
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

    def get_session_summary(self) -> dict:
        """
        Get session summary.

        Returns:
            Dictionary with session summary
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

    def _get_emotion_distribution(self) -> dict:
        """
        Get emotion distribution in the session.

        Returns:
            Dictionary with emotion distribution
        """
        if not self.emotion_history:
            return {}

        total = len(self.emotion_history)
        counts = Counter(self.emotion_history)

        return {
            emotion: {
                'count': count,
                'percentage': round((count / total) * 100, 1)
            }
            for emotion, count in counts.items()
        }

    def get_stress_percentage(self) -> dict:
        """
        Get stress level percentages for visualization.

        Returns:
            Dictionary with percentage for each level
        """
        if not self.stress_history:
            return {i: 0 for i in range(1, 8)}

        counts = Counter(self.stress_history)
        total = len(self.stress_history)

        return {
            level: round((counts.get(level, 0) / total) * 100, 1)
            for level in range(1, 8)
        }

    def get_recommendation(self) -> str:
        """
        Get recommendation based on current stress level.

        Returns:
            Recommendation string
        """
        current = self.get_current_stress_level()
        level = current['level']

        return get_recommendation(level)

    def reset(self) -> None:
        """Reset all history and start new session."""
        self.emotion_history.clear()
        self.stress_history.clear()
        self.timestamp_history.clear()
        self.session_start = datetime.now()
        self.detection_count = 0
        self._last_confidence = 0


if __name__ == "__main__":
    # Simple test
    analyzer = StressAnalyzer()

    # Simulate detections
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