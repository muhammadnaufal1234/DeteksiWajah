"""
Unit Tests for Emotion Detection and Stress Analysis
======================================================

Author: AI Assistant
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion_detector import EmotionDetector, create_cnn_model
from src.stress_analyzer import StressAnalyzer
from src.config import (
    EMOTION_TO_STRESS_SCORE,
    get_stress_level,
    get_recommendation
)


def test_cnn_model_creation():
    """Test CNN model creation."""
    print("Testing CNN model creation...")
    model = create_cnn_model()
    assert model is not None
    assert len(model.layers) > 0
    print("[OK] CNN model creation passed")


def test_emotion_detector_init():
    """Test EmotionDetector initialization."""
    print("Testing EmotionDetector initialization...")
    detector = EmotionDetector()
    assert detector is not None
    assert detector.face_cascade is not None
    print("[OK] EmotionDetector initialization passed")


def test_image_preprocessing():
    """Test image preprocessing."""
    print("Testing image preprocessing...")
    detector = EmotionDetector()

    # Create dummy image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    processed = detector.preprocess_image(test_image)
    assert processed is not None
    assert processed.shape == (1, 48, 48, 1)
    print("[OK] Image preprocessing passed")


def test_stress_analyzer_init():
    """Test StressAnalyzer initialization."""
    print("Testing StressAnalyzer initialization...")
    analyzer = StressAnalyzer()
    assert analyzer is not None
    assert analyzer.history_size == 30
    print("[OK] StressAnalyzer initialization passed")


def test_add_detection():
    """Test adding detection."""
    print("Testing add_detection...")
    analyzer = StressAnalyzer()

    analyzer.add_detection('Happy', 0.9)
    assert len(analyzer.emotion_history) == 1
    assert len(analyzer.stress_history) == 1

    analyzer.add_detection('Angry', 0.8)
    assert len(analyzer.emotion_history) == 2
    print("[OK] Add detection passed")


def test_get_current_stress_level():
    """Test getting current stress level."""
    print("Testing get_current_stress_level...")
    analyzer = StressAnalyzer()

    # Empty state
    result = analyzer.get_current_stress_level()
    assert result['level'] == 0
    assert result['name'] == 'Tidak Ada Data'

    # With detections
    analyzer.add_detection('Happy', 0.9)
    analyzer.add_detection('Happy', 0.9)
    analyzer.add_detection('Neutral', 0.8)

    result = analyzer.get_current_stress_level()
    assert result['level'] > 0
    assert 'name' in result
    print("[OK] Get current stress level passed")


def test_emotion_stress_mapping():
    """Test emotion to stress mapping."""
    print("Testing emotion-stress mapping...")

    # Happy should map to low stress
    happy_analyzer = StressAnalyzer()
    happy_analyzer.add_detection('Happy', 0.9)
    happy_result = happy_analyzer.get_current_stress_level()
    assert happy_result['level'] <= 2

    # Angry should map to high stress
    angry_analyzer = StressAnalyzer()
    angry_analyzer.add_detection('Angry', 0.9)
    angry_result = angry_analyzer.get_current_stress_level()
    assert angry_result['level'] >= 5

    print("[OK] Emotion-stress mapping passed")


def test_get_average_stress():
    """Test average stress calculation."""
    print("Testing get_average_stress...")
    analyzer = StressAnalyzer()

    emotions = ['Happy', 'Happy', 'Neutral', 'Sad', 'Fear']
    for emotion in emotions:
        analyzer.add_detection(emotion, 0.8)

    avg = analyzer.get_average_stress()
    assert 1 <= avg <= 7
    print(f"[OK] Average stress: {avg:.2f} passed")


def test_get_dominant_emotion():
    """Test dominant emotion detection."""
    print("Testing get_dominant_emotion...")
    analyzer = StressAnalyzer()

    emotions = ['Happy', 'Happy', 'Happy', 'Neutral', 'Sad']
    for emotion in emotions:
        analyzer.add_detection(emotion, 0.8)

    dominant, count, percentage = analyzer.get_dominant_emotion()
    assert dominant == 'Happy'
    assert count == 3
    assert percentage == 60.0
    print("[OK] Get dominant emotion passed")


def test_get_recommendation():
    """Test recommendation generation."""
    print("Testing get_recommendation...")
    analyzer = StressAnalyzer()

    # Low stress
    analyzer.add_detection('Happy', 0.9)
    rec = analyzer.get_recommendation()
    assert 'prima' in rec or 'sehat' in rec

    # High stress
    analyzer2 = StressAnalyzer()
    analyzer2.add_detection('Angry', 0.9)
    rec2 = analyzer2.get_recommendation()
    assert 'profesional' in rec2 or 'bantuan' in rec2

    print("[OK] Get recommendation passed")


def test_reset():
    """Test analyzer reset."""
    print("Testing reset...")
    analyzer = StressAnalyzer()

    analyzer.add_detection('Happy', 0.9)
    analyzer.add_detection('Angry', 0.8)
    assert len(analyzer.emotion_history) == 2

    analyzer.reset()
    assert len(analyzer.emotion_history) == 0
    assert len(analyzer.stress_history) == 0
    print("[OK] Reset passed")


def test_config_functions():
    """Test config helper functions."""
    print("Testing config functions...")

    # Test get_stress_level
    assert get_stress_level(1.0) == 1
    assert get_stress_level(2.0) == 2
    assert get_stress_level(3.5) == 3
    assert get_stress_level(7.0) == 7

    # Test get_recommendation
    rec1 = get_recommendation(1)
    assert len(rec1) > 0
    rec7 = get_recommendation(7)
    assert len(rec7) > 0

    # Test EMOTION_TO_STRESS_SCORE
    assert EMOTION_TO_STRESS_SCORE['Happy'] == 1
    assert EMOTION_TO_STRESS_SCORE['Disgust'] == 7

    print("[OK] Config functions passed")


def run_all_tests() -> bool:
    """Run all tests."""
    print("\n" + "="*50)
    print("Running Unit Tests")
    print("="*50 + "\n")

    tests = [
        test_cnn_model_creation,
        test_emotion_detector_init,
        test_image_preprocessing,
        test_stress_analyzer_init,
        test_add_detection,
        test_get_current_stress_level,
        test_emotion_stress_mapping,
        test_get_average_stress,
        test_get_dominant_emotion,
        test_get_recommendation,
        test_reset,
        test_config_functions
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "="*50)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)