"""
Main Entry Point untuk Deteksi Stres dari Ekspresi Wajah
=======================================================
Jalankan aplikasi GUI atau training model

Usage:
    python main.py              # Jalankan GUI
    python main.py --train     # Training model
    python main.py --test      # Jalankan unit tests
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == '--train':
            print("Starting model training...")
            from src.train_model import train_model
            train_model()

        elif command == '--test':
            print("Running unit tests...")
            from tests.test_emotion_detection import run_all_tests
            run_all_tests()

        elif command == '--gui':
            print("Starting GUI application...")
            from src.gui_app import main
            main()

        else:
            print("Unknown command. Use --gui, --train, or --test")
            print_usage()

    else:
        print("Starting GUI application...")
        from src.gui_app import main
        main()


def print_usage():
    """Print usage information"""
    print("\nUsage:")
    print("  python main.py              # Jalankan GUI")
    print("  python main.py --train      # Training model")
    print("  python main.py --test      # Jalankan unit tests")
    print("  python main.py --gui       # Jalankan GUI (explicit)")


if __name__ == "__main__":
    main()
