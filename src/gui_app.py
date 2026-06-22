"""
GUI Application for Stress Detection from Facial Expressions
=============================================================
Desktop application with Tkinter interface for real-time detection.

Author: AI Assistant
"""

import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local modules
from src.emotion_detector import EmotionDetector
from src.stress_analyzer import StressAnalyzer
from src.config import (
    EMOTION_LABELS,
    EMOTION_COLORS_HEX,
    APP_TITLE,
    APP_GEOMETRY,
    APP_THEME,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    CAMERA_INDEX,
    DEFAULT_HISTORY_SIZE
)


class StressDetectionGUI:
    """
    GUI Application for stress detection from facial expressions.
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize GUI.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(APP_GEOMETRY)
        self.root.configure(bg=APP_THEME['bg_primary'])

        # Initialize components
        self.emotion_detector = None
        self.stress_analyzer = StressAnalyzer(history_size=DEFAULT_HISTORY_SIZE)

        # Video capture state
        self.cap = None
        self.video_thread = None
        self.is_running = False
        self.current_frame = None

        # Setup model and build UI
        self._setup_model()
        self._create_widgets()

        # Start video when app opens
        self.root.after(100, self.start_video)

    def _setup_model(self) -> None:
        """Setup emotion detection model with DeepFace."""
        self.emotion_detector = EmotionDetector()

        if not self.emotion_detector.deepface_available:
            print("[!] DeepFace not installed. Install with:")
            print("    pip install deepface")
            self.status_label.config(
                text="[!] DeepFace not installed - please install first",
                fg="#FF9800"
            )

    def _create_widgets(self) -> None:
        """Create all GUI widgets."""
        # Title
        title_frame = tk.Frame(self.root, bg=APP_THEME['bg_primary'])
        title_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            title_frame,
            text="Deteksi Tingkat Stres dari Ekspresi Wajah",
            font=('Helvetica', 20, 'bold'),
            fg=APP_THEME['text_primary'],
            bg=APP_THEME['bg_primary']
        ).pack()

        # Main content frame
        main_frame = tk.Frame(self.root, bg=APP_THEME['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Video feed
        left_frame = tk.Frame(
            main_frame,
            bg=APP_THEME['bg_secondary'],
            bd=2,
            relief=tk.RIDGE
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            left_frame,
            text="Kamera",
            font=('Helvetica', 14, 'bold'),
            fg=APP_THEME['text_primary'],
            bg=APP_THEME['bg_secondary']
        ).pack(pady=5)

        self.video_label = tk.Label(left_frame, bg=APP_THEME['bg_dark'])
        self.video_label.pack(padx=10, pady=10)

        # Video controls
        control_frame = tk.Frame(left_frame, bg=APP_THEME['bg_secondary'])
        control_frame.pack(pady=10)

        self.start_btn = tk.Button(
            control_frame,
            text="Mulai",
            command=self.start_video,
            width=10,
            bg=APP_THEME['accent_green'],
            fg='white',
            font=('Helvetica', 10, 'bold')
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_video,
            width=10,
            bg=APP_THEME['accent_red'],
            fg='white',
            font=('Helvetica', 10, 'bold'),
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Right panel - Results
        right_frame = tk.Frame(
            main_frame,
            bg=APP_THEME['bg_secondary'],
            bd=2,
            relief=tk.RIDGE
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))

        # Stress Level Display
        tk.Label(
            right_frame,
            text="Tingkat Stres",
            font=('Helvetica', 14, 'bold'),
            fg=APP_THEME['text_primary'],
            bg=APP_THEME['bg_secondary']
        ).pack(pady=5)

        # Stress level indicator
        stress_frame = tk.Frame(right_frame, bg=APP_THEME['bg_secondary'])
        stress_frame.pack(pady=10, padx=10, fill=tk.X)

        self.stress_label = tk.Label(
            stress_frame,
            text="---",
            font=('Helvetica', 36, 'bold'),
            fg=APP_THEME['accent_yellow'],
            bg=APP_THEME['bg_secondary']
        )
        self.stress_label.pack()

        self.stress_desc = tk.Label(
            stress_frame,
            text="Tunggu deteksi...",
            font=('Helvetica', 12),
            fg=APP_THEME['text_secondary'],
            bg=APP_THEME['bg_secondary']
        )
        self.stress_desc.pack()

        # Emotion probabilities with Canvas
        tk.Label(
            right_frame,
            text="Deteksi Emosi",
            font=('Helvetica', 12, 'bold'),
            fg=APP_THEME['text_primary'],
            bg=APP_THEME['bg_secondary']
        ).pack(pady=(20, 5))

        self.emotion_labels = {}
        self.emotion_canvases = {}
        self.emotion_rects = {}

        for emotion in EMOTION_LABELS:
            row = tk.Frame(right_frame, bg=APP_THEME['bg_secondary'])
            row.pack(fill=tk.X, padx=20, pady=2)

            label = tk.Label(
                row,
                text=f"{emotion:10s}",
                font=('Courier', 10),
                fg=APP_THEME['text_primary'],
                bg=APP_THEME['bg_secondary'],
                width=12,
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            # Use Canvas for progress bar
            canvas = tk.Canvas(
                row,
                width=150,
                height=15,
                bg='#333333',
                highlightthickness=0
            )
            canvas.pack(side=tk.LEFT, padx=5)
            rect = canvas.create_rectangle(0, 0, 0, 15, fill=APP_THEME['accent_green'])

            self.emotion_labels[emotion] = label
            self.emotion_canvases[emotion] = canvas
            self.emotion_rects[emotion] = rect

        # Recommendation
        rec_frame = tk.Frame(
            right_frame,
            bg=APP_THEME['bg_primary'],
            bd=1,
            relief=tk.SUNKEN
        )
        rec_frame.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(
            rec_frame,
            text="Rekomendasi",
            font=('Helvetica', 11, 'bold'),
            fg=APP_THEME['text_primary'],
            bg=APP_THEME['bg_primary']
        ).pack(pady=5)

        self.recommendation_label = tk.Label(
            rec_frame,
            text="Mulai deteksi untuk melihat rekomendasi",
            font=('Helvetica', 10),
            fg=APP_THEME['text_secondary'],
            bg=APP_THEME['bg_primary'],
            wraplength=300,
            justify=tk.LEFT
        )
        self.recommendation_label.pack(pady=5, padx=10)

        # Statistics
        stats_frame = tk.Frame(right_frame, bg=APP_THEME['bg_secondary'])
        stats_frame.pack(pady=10, padx=20, fill=tk.X)

        self.stats_label = tk.Label(
            stats_frame,
            text="Deteksi: 0 | Rata-rata: 0.0",
            font=('Helvetica', 9),
            fg='#888888',
            bg=APP_THEME['bg_secondary']
        )
        self.stats_label.pack()

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Siap",
            font=('Helvetica', 9),
            fg='#888888',
            bg=APP_THEME['bg_dark'],
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, padx=20, pady=5)

    def start_video(self) -> None:
        """Start video capture and processing."""
        if self.is_running:
            return

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat mengakses kamera!")
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Kamera aktif - Mendeteksi...")

        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_video(self) -> None:
        """Stop video capture."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Kamera dihentikan")

        if self.cap:
            self.cap.release()
            self.cap = None

    def _video_loop(self) -> None:
        """Main video processing loop."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Process frame
            results = self.emotion_detector.process_frame(frame, draw_box=True)

            # Update stress analyzer
            for result in results:
                self.stress_analyzer.add_detection(
                    result['emotion'],
                    result['confidence'],
                    result['probabilities']
                )

            # Update display
            self.current_frame = frame.copy()
            self.root.after(1, self._update_display, results)

    def _update_display(self, results: list) -> None:
        """Update GUI display with detection results."""
        if self.current_frame is None:
            return

        # Convert frame for display
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (VIDEO_WIDTH, VIDEO_HEIGHT))
        image = Image.fromarray(frame_resized)
        image_tk = ImageTk.PhotoImage(image)
        self.video_label.config(image=image_tk)
        self.video_label.image = image_tk

        # Update stress level
        stress_info = self.stress_analyzer.get_current_stress_level()
        self.stress_label.config(
            text=stress_info['name'],
            fg=stress_info['color']
        )
        self.stress_desc.config(text=stress_info['description'])

        # Update emotion bars using Canvas
        if results:
            probs = results[0]['probabilities']
            for i, emotion in enumerate(EMOTION_LABELS):
                prob = probs[i]
                color = EMOTION_COLORS_HEX.get(emotion, '#4CAF50')

                # Update canvas bar
                canvas = self.emotion_canvases[emotion]
                rect = self.emotion_rects[emotion]

                bar_width = int(prob * 150)
                canvas.coords(rect, 0, 0, bar_width, 15)
                canvas.itemconfig(rect, fill=color)

                # Update percentage text
                self.emotion_labels[emotion].config(
                    text=f"{emotion:10s} {prob*100:5.1f}%"
                )

        # Update recommendation
        recommendation = self.stress_analyzer.get_recommendation()
        self.recommendation_label.config(text=recommendation)

        # Update stats
        avg = self.stress_analyzer.get_average_stress()
        count = self.stress_analyzer.detection_count
        self.stats_label.config(text=f"Deteksi: {count} | Rata-rata: {avg:.1f}")

        # Update status with DeepFace info
        if self.emotion_detector and self.emotion_detector.deepface_available:
            self.status_label.config(
                text="[OK] DeepFace aktif - Pre-trained model (tanpa training)",
                fg=APP_THEME['accent_green']
            )

    def on_closing(self) -> None:
        """Handle window closing."""
        self.stop_video()
        self.root.destroy()


def main() -> None:
    """Main entry point."""
    root = tk.Tk()
    app = StressDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()