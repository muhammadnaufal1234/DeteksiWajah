"""
Training Script untuk Emotion Recognition Model
================================================
Script untuk training model CNN dengan FER2013 dataset

Usage:
    python train_model.py --data data/fer2013.csv --epochs 50

Author: AI Assistant
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from datetime import datetime

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion_detector import EMOTION_LABELS


# Constants
IMAGE_SIZE = (48, 48)
NUM_CLASSES = 7
BATCH_SIZE = 64
DEFAULT_EPOCHS = 50


def load_fer2013(csv_path):
    """
    Load FER2013 dataset dari CSV file

    Args:
        csv_path: Path ke file fer2013.csv

    Returns:
        X_train, X_val, y_train, y_val
    """
    print("Loading FER2013 dataset...")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Extract pixels and labels
    X = []
    y = []

    for idx, row in df.iterrows():
        pixels = np.array(row['pixels'].split(), dtype=np.uint8)
        pixels = pixels.reshape(48, 48)
        X.append(pixels)
        y.append(row['emotion'])

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset loaded: {len(X)} samples")
    print(f"Emotion distribution:\n{pd.Series(y).value_counts().sort_index()}")

    # Reshape untuk CNN
    X = X.reshape(-1, 48, 48, 1)
    X = X.astype('float32') / 255.0

    # One-hot encode labels
    y = to_categorical(y, num_classes=NUM_CLASSES)

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    return X_train, X_val, y_train, y_val


def create_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create CNN model untuk emotion recognition

    Args:
        input_shape: Shape input gambar
        num_classes: Jumlah kelas emosi

    Returns:
        Keras model
    """
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 4
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


def create_callbacks(checkpoint_path, log_path):
    """
    Create training callbacks

    Args:
        checkpoint_path: Path untuk menyimpan model
        log_path: Path untuk training log

    Returns:
        List of callbacks
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),

        # CSV Logger
        CSVLogger(log_path)
    ]

    return callbacks


def train_model(data_path=None, epochs=DEFAULT_EPOCHS, batch_size=BATCH_SIZE):
    """
    Train emotion recognition model

    Args:
        data_path: Path ke FER2013 CSV
        epochs: Number of training epochs
        batch_size: Batch size
    """
    # Project directories
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')
    logs_dir = os.path.join(project_dir, 'logs')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Model paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(models_dir, f'emotion_model_{timestamp}.h5')
    best_model_path = os.path.join(models_dir, 'emotion_model.h5')
    log_path = os.path.join(logs_dir, f'training_log_{timestamp}.csv')

    # Load dataset
    if data_path is None:
        root_data_path = os.path.join(project_dir, 'fer2013.csv')
        data_folder_path = os.path.join(project_dir, 'data', 'fer2013.csv')
        
        if os.path.exists(root_data_path):
            data_path = root_data_path
            print(f"[OK] Dataset ditemukan di root folder: {data_path}")
        else:
            data_path = data_folder_path

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("\nSilakan download FER2013 dataset:")
        print("1. Kunjungi: https://www.kaggle.com/datasets/deadskull7/fer2013")
        print("2. Download fer2013.csv")
        print("3. Tempatkan di folder data/")
        return None

    X_train, X_val, y_train, y_val = load_fer2013(data_path)

    # Create model
    print("\nCreating model...")
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Create callbacks
    callbacks = create_callbacks(checkpoint_path, log_path)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(checkpoint_path)
    model.save(best_model_path)
    print(f"\nModel saved to: {best_model_path}")

    # Print results
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Model saved: {best_model_path}")
    print(f"Log saved: {log_path}")

    return model, history


def evaluate_model(model_path, data_path=None):
    """
    Evaluate trained model

    Args:
        model_path: Path ke model
        data_path: Path ke dataset
    """
    from tensorflow.keras.models import load_model

    print(f"\nEvaluating model: {model_path}")

    # Load model
    model = load_model(model_path)

    # Load data
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(project_dir, 'data', 'fer2013.csv')

    if not os.path.exists(data_path):
        print("Dataset not found!")
        return

    X_train, X_val, y_train, y_val = load_fer2013(data_path)

    # Evaluate
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print(f"\nValidation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Classification report
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=EMOTION_LABELS))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Model')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to FER2013 CSV file')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--evaluate', type=str, default=None,
                        help='Evaluate trained model')

    args = parser.parse_args()

    if args.evaluate:
        evaluate_model(args.evaluate, args.data)
    else:
        train_model(args.data, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()