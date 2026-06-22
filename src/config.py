"""
Configuration Module
====================
Centralized constants and settings for the stress detection application.

Author: AI Assistant
"""

# =============================================================================
# Emotion Configuration
# =============================================================================

# Emotion labels following FER2013 dataset standard
EMOTION_LABELS = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# Emotion colors for OpenCV visualization (BGR format)
EMOTION_COLORS_BGR = {
    'Angry':     (0, 0, 255),       # Red
    'Disgust':   (144, 0, 255),     # Purple
    'Fear':      (255, 0, 255),     # Magenta
    'Happy':     (0, 255, 0),       # Green
    'Sad':       (255, 0, 0),       # Blue
    'Surprise':  (0, 255, 255),     # Yellow
    'Neutral':   (128, 128, 128)    # Gray
}

# Emotion colors for Tkinter visualization (HEX format)
EMOTION_COLORS_HEX = {
    'Angry':     '#FF0000',         # Red
    'Disgust':   '#9000FF',         # Purple
    'Fear':      '#FF00FF',         # Magenta
    'Happy':     '#00FF00',         # Green
    'Sad':       '#0000FF',         # Blue
    'Surprise':  '#FFFF00',         # Yellow
    'Neutral':   '#808080'          # Gray
}

# Mapping emotion to stress score (1=Low, 7=Critical)
EMOTION_TO_STRESS_SCORE = {
    'Happy':     1,
    'Surprise':  2,
    'Neutral':   3,
    'Sad':       4,
    'Fear':      5,
    'Angry':     6,
    'Disgust':   7
}

# Mapping from DeepFace emotion labels to standard format
DEEPFACE_EMOTION_MAP = {
    'happy':     'Happy',
    'sad':       'Sad',
    'angry':     'Angry',
    'surprise':  'Surprise',
    'fear':      'Fear',
    'disgust':   'Disgust',
    'neutral':   'Neutral'
}

# =============================================================================
# Stress Level Configuration
# =============================================================================

STRESS_LEVELS = {
    1: {
        'name': 'Rendah',
        'label': 'Low',
        'color': '#4CAF50',
        'description': 'Tingkat stres rendah'
    },
    2: {
        'name': 'Rendah-Sedang',
        'label': 'Low-Medium',
        'color': '#8BC34A',
        'description': 'Tingkat stres rendah-sedang'
    },
    3: {
        'name': 'Sedang',
        'label': 'Medium',
        'color': '#FFC107',
        'description': 'Tingkat stres sedang'
    },
    4: {
        'name': 'Sedang-Tinggi',
        'label': 'Medium-High',
        'color': '#FF9800',
        'description': 'Tingkat stres sedang-tinggi'
    },
    5: {
        'name': 'Tinggi',
        'label': 'High',
        'color': '#F44336',
        'description': 'Tingkat stres tinggi'
    },
    6: {
        'name': 'Sangat Tinggi',
        'label': 'Very High',
        'color': '#D32F2F',
        'description': 'Tingkat stres sangat tinggi'
    },
    7: {
        'name': 'Kritis',
        'label': 'Critical',
        'color': '#B71C1C',
        'description': 'Tingkat stres kritis'
    }
}

# Stress score thresholds for level calculation
STRESS_SCORE_THRESHOLDS = {
    'low':       1.5,
    'low_med':   2.5,
    'medium':    3.5,
    'med_high':  4.5,
    'high':      5.5,
    'very_high': 6.5
}

# =============================================================================
# Model Configuration
# =============================================================================

# Image dimensions
IMAGE_SIZE = (48, 48)
NUM_CLASSES = 7

# Training parameters
BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
LEARNING_RATE = 0.001

# =============================================================================
# Application Configuration
# =============================================================================

# GUI Settings
APP_TITLE = "Deteksi Tingkat Stres - Facial Expression Recognition"
APP_GEOMETRY = "1200x800"
APP_THEME = {
    'bg_primary':    '#1a1a2e',
    'bg_secondary':  '#16213e',
    'bg_dark':       '#0f0f23',
    'text_primary':  '#ffffff',
    'text_secondary':'#aaaaaa',
    'accent_green':  '#4CAF50',
    'accent_red':    '#F44336',
    'accent_yellow': '#FFC107'
}

# Video settings
VIDEO_WIDTH = 480
VIDEO_HEIGHT = 360
CAMERA_INDEX = 0

# Analysis settings
DEFAULT_HISTORY_SIZE = 30
MIN_DETECTIONS_FOR_TREND = 5

# =============================================================================
# Recommendation Messages
# =============================================================================

STRESS_RECOMMENDATIONS = {
    1: "🎉 Kondisi prima! Tetap jaga pola hidup sehat.",
    2: "😊 Kondisi baik. Lanjutkan aktivitas seperti biasa.",
    3: "📋 Perhatikan tanda-tanda stres. Istirahat jika perlu.",
    4: "💆 Lakukan relaksasi, coba tarik napas dalam.",
    5: "⚠️ Tingkat stres meningkat. Luangkan waktu untuk relaksasi.",
    6: "🚨 Disarankan untuk berkonsultasi dengan profesional.",
    7: "🏥 Perlu perhatian segera. Segera cari bantuan profesional."
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_stress_level(score: float) -> int:
    """
    Convert stress score to stress level (1-7).

    Args:
        score: Stress score (1-7)

    Returns:
        Stress level (1-7)
    """
    if score <= STRESS_SCORE_THRESHOLDS['low']:
        return 1
    elif score <= STRESS_SCORE_THRESHOLDS['low_med']:
        return 2
    elif score <= STRESS_SCORE_THRESHOLDS['medium']:
        return 3
    elif score <= STRESS_SCORE_THRESHOLDS['med_high']:
        return 4
    elif score <= STRESS_SCORE_THRESHOLDS['high']:
        return 5
    elif score <= STRESS_SCORE_THRESHOLDS['very_high']:
        return 6
    else:
        return 7


def get_stress_info(score: float) -> dict:
    """
    Get complete stress information from score.

    Args:
        score: Stress score (1-7)

    Returns:
        Dictionary with stress level information
    """
    level = get_stress_level(score)
    level_info = STRESS_LEVELS[level]

    return {
        'level': level,
        'name': level_info['name'],
        'label': level_info['label'],
        'color': level_info['color'],
        'description': level_info['description'],
        'average_score': round(score, 2)
    }


def get_recommendation(level: int) -> str:
    """
    Get recommendation for stress level.

    Args:
        level: Stress level (1-7)

    Returns:
        Recommendation string
    """
    return STRESS_RECOMMENDATIONS.get(level, "Tidak ada rekomendasi tersedia.")