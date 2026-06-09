"""
GUI Application untuk Deteksi Stres dari Ekspresi Wajah
=======================================================
Aplikasi desktop dengan antarmuka Tkinter untuk real-time detection

Author: AI Assistant
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion_detector import EmotionDetector, EMOTION_LABELS, EMOTION_COLORS
from src.stress_analyzer import StressAnalyzer


class StressDetectionGUI:
    """
    GUI Application untuk deteksi stres dari ekspresi wajah
    """

    def __init__(self, root):
        """
        Inisialisasi GUI

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Deteksi Tingkat Stres - Facial Expression Recognition")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')

        # Initialize components
        self.emotion_detector = None
        self.stress_analyzer = StressAnalyzer(history_size=50)

        # Video capture
        self.cap = None
        self.video_thread = None
        self.is_running = False
        self.current_frame = None

        # Create model
        self._setup_model()

        # Build UI
        self._create_widgets()

        # Start video when app opens
        self.root.after(100, self.start_video)

    def _setup_model(self):
        """Setup emotion detection model dengan DeepFace"""
        self.emotion_detector = EmotionDetector()

        if not self.emotion_detector.deepface_available:
            print("[!] DeepFace belum terinstall. Install dengan:")
            print("    pip install deepface")
            self.status_label.config(text="[!] DeepFace belum terinstall - perlu install dulu", fg="#FF9800")

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a2e')
        title_frame.pack(fill=tk.X, pady=10)

        title_label = tk.Label(
            title_frame,
            text="Deteksi Tingkat Stres dari Ekspresi Wajah",
            font=('Helvetica', 20, 'bold'),
            fg='#ffffff',
            bg='#1a1a2e'
        )
        title_label.pack()

        # Main content frame
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Video feed
        left_frame = tk.Frame(main_frame, bg='#16213e', bd=2, relief=tk.RIDGE)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        video_label = tk.Label(
            left_frame,
            text="Kamera",
            font=('Helvetica', 14, 'bold'),
            fg='#ffffff',
            bg='#16213e'
        )
        video_label.pack(pady=5)

        self.video_label = tk.Label(left_frame, bg='#0f0f23')
        self.video_label.pack(padx=10, pady=10)

        # Video controls
        control_frame = tk.Frame(left_frame, bg='#16213e')
        control_frame.pack(pady=10)

        self.start_btn = tk.Button(
            control_frame,
            text="Mulai",
            command=self.start_video,
            width=10,
            bg='#4CAF50',
            fg='white',
            font=('Helvetica', 10, 'bold')
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_video,
            width=10,
            bg='#F44336',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg='#16213e', bd=2, relief=tk.RIDGE)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))

        # Stress Level Display
        stress_title = tk.Label(
            right_frame,
            text="Tingkat Stres",
            font=('Helvetica', 14, 'bold'),
            fg='#ffffff',
            bg='#16213e'
        )
        stress_title.pack(pady=5)

        # Stress level indicator
        self.stress_frame = tk.Frame(right_frame, bg='#16213e')
        self.stress_frame.pack(pady=10, padx=10, fill=tk.X)

        self.stress_label = tk.Label(
            self.stress_frame,
            text="---",
            font=('Helvetica', 36, 'bold'),
            fg='#FFC107',
            bg='#16213e'
        )
        self.stress_label.pack()

        self.stress_desc = tk.Label(
            self.stress_frame,
            text="Tunggu deteksi...",
            font=('Helvetica', 12),
            fg='#aaaaaa',
            bg='#16213e'
        )
        self.stress_desc.pack()

        # Emotion probabilities with Canvas
        emotion_title = tk.Label(
            right_frame,
            text="Deteksi Emosi",
            font=('Helvetica', 12, 'bold'),
            fg='#ffffff',
            bg='#16213e'
        )
        emotion_title.pack(pady=(20, 5))

        self.emotion_labels = {}
        self.emotion_canvases = {}
        self.emotion_rects = {}

        for emotion in EMOTION_LABELS:
            row = tk.Frame(right_frame, bg='#16213e')
            row.pack(fill=tk.X, padx=20, pady=2)

            label = tk.Label(
                row,
                text=f"{emotion:10s}",
                font=('Courier', 10),
                fg='#ffffff',
                bg='#16213e',
                width=12,
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            # Use Canvas for progress bar
            canvas = tk.Canvas(row, width=150, height=15, bg='#333333', highlightthickness=0)
            canvas.pack(side=tk.LEFT, padx=5)
            rect = canvas.create_rectangle(0, 0, 0, 15, fill='#4CAF50')

            percent_label = tk.Label(
                row,
                text="0%",
                font=('Courier', 10),
                fg='#ffffff',
                bg='#16213e',
                width=6
            )
            percent_label.pack(side=tk.LEFT)

            self.emotion_labels[emotion] = label
            self.emotion_canvases[emotion] = canvas
            self.emotion_rects[emotion] = rect

        # Recommendation
        rec_frame = tk.Frame(right_frame, bg='#1a1a2e', bd=1, relief=tk.SUNKEN)
        rec_frame.pack(fill=tk.X, padx=20, pady=20)

        rec_title = tk.Label(
            rec_frame,
            text="Rekomendasi",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#1a1a2e'
        )
        rec_title.pack(pady=5)

        self.recommendation_label = tk.Label(
            rec_frame,
            text="Mulai deteksi untuk melihat rekomendasi",
            font=('Helvetica', 10),
            fg='#aaaaaa',
            bg='#1a1a2e',
            wraplength=300,
            justify=tk.LEFT
        )
        self.recommendation_label.pack(pady=5, padx=10)

        # Statistics
        stats_frame = tk.Frame(right_frame, bg='#16213e')
        stats_frame.pack(pady=10, padx=20, fill=tk.X)

        self.stats_label = tk.Label(
            stats_frame,
            text="Deteksi: 0 | Rata-rata:0.0",
            font=('Helvetica', 9),
            fg='#888888',
            bg='#16213e'
        )
        self.stats_label.pack()

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Siap",
            font=('Helvetica', 9),
            fg='#888888',
            bg='#0f0f23',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, padx=20, pady=5)

    def start_video(self):
        """Start video capture and processing"""
        if self.is_running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat mengakses kamera!")
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Kamera aktif - Mendeteksi...")

        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_video(self):
        """Stop video capture"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Kamera dihentikan")

        if self.cap:
            self.cap.release()
            self.cap = None

    def _video_loop(self):
        """Main video processing loop"""
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

    def _update_display(self, results):
        """Update GUI display with detection results"""
        if self.current_frame is None:
            return

        # Convert frame for display
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (480, 360))
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
                color = EMOTION_COLORS.get(emotion, '#4CAF50')

                # Update canvas bar
                canvas = self.emotion_canvases[emotion]
                rect = self.emotion_rects[emotion]

                bar_width = int(prob * 150)
                canvas.coords(rect, 0, 0, bar_width, 15)
                canvas.itemconfig(rect, fill=color)

                # Update percentage
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

        # Update status dengan info DeepFace
        if self.emotion_detector and self.emotion_detector.deepface_available:
            self.status_label.config(
                text="[OK] DeepFace aktif - Pre-trained model (tanpa training)",
                fg="#4CAF50"
            )

    def on_closing(self):
        """Handle window closing"""
        self.stop_video()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = StressDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()