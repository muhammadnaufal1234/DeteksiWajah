"""
Main Entry Point for Stress Detection from Facial Expressions
============================================================
Run GUI application, train model, or execute tests.

Usage:
    python main.py              # Run GUI (default)
    python main.py --gui       # Run GUI (explicit)
    python main.py --train     # Train model
    python main.py --test      # Run unit tests

Author: AI Assistant
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    """Main entry point."""
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
            from src.gui_app import main as gui_main
            gui_main()

        elif command == '--evaluate':
            print("Starting evaluation...")
            from src.evaluate import evaluate_on_camera
            evaluate_on_camera()

        else:
            print_usage()

    else:
        print("Starting GUI application...")
        from src.gui_app import main as gui_main
        gui_main()


def print_usage() -> None:
    """Print usage information."""
    print("\nUsage:")
    print("  python main.py              # Run GUI (default)")
    print("  python main.py --gui       # Run GUI (explicit)")
    print("  python main.py --train      # Train model")
    print("  python main.py --test      # Run unit tests")
    print("  python main.py --evaluate  # Evaluate on camera")


if __name__ == "__main__":
    main()
