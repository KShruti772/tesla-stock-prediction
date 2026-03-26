"""
train_model.py

Script to train LSTM model for stock price prediction.
Builds, trains, and saves the model weights for use in the Streamlit app.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import pickle
import os

from data_utils import load_and_preprocess_data, save_scaler


def build_lstm_model(input_shape=(60, 5)):
    """
    Build and compile LSTM model architecture.

    Args:
        input_shape: tuple specifying input shape (default: (60, 5) for 5 features)

    Returns:
        Compiled Sequential model
    """
    print(f"🏗️ Building LSTM model with input shape {input_shape}...")

    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        print("✅ LSTM model built successfully")
        print(f"📊 Model summary:")
        model.summary()

        return model

    except Exception as e:
        raise RuntimeError(f"❌ Failed to build LSTM model: {e}")


def train_lstm_model(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1):
    """
    Train the LSTM model.

    Args:
        X_train: Training sequences (n_samples, 60, n_features)
        y_train: Target values (n_samples,)
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation

    Returns:
        Trained model
    """
    print(f"🚀 Starting LSTM model training...")
    print(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} time steps, {X_train.shape[2]} features")
    print(f"⚙️ Training parameters: epochs={epochs}, batch_size={batch_size}, validation_split={validation_split}")

    # Build model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Train model
    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=True
        )

        print("✅ Model training completed successfully")
        print(f"📈 Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"📈 Final validation loss: {history.history['val_loss'][-1]:.4f}")
        return model

    except Exception as e:
        raise RuntimeError(f"❌ Model training failed: {e}")


def save_model_weights(model, filepath='lstm_model.weights.h5'):
    """
    Save model weights to disk.

    Args:
        model: Trained Keras model
        filepath: Path to save the weights
    """
    try:
        model.save_weights(filepath)
        print(f"✅ Model weights saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving model weights: {e}")


def main(csv_path, epochs=25, batch_size=32):
    """
    Main training pipeline.

    Args:
        csv_path: Path to CSV file containing stock data
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("🎯 Starting LSTM Stock Prediction Training Pipeline")
    print("=" * 50)

    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    try:
        # Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        X_train, y_train, scaler = load_and_preprocess_data(csv_path)

        # Train model
        print("\n2. Training LSTM model...")
        trained_model = train_lstm_model(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Save model and scaler
        print("\n3. Saving model and scaler...")
        save_model_weights(trained_model, 'lstm_model.weights.h5')
        save_scaler(scaler, 'scaler.pkl')

        print("\n🎉 Training pipeline completed successfully!")
        print("📁 Files saved:")
        print("   - lstm_model.weights.h5 (model weights)")
        print("   - scaler.pkl (data scaler)")
        print("\n🚀 You can now use these files in the Streamlit app!")

    except Exception as e:
        print(f"\n❌ Training pipeline failed: {e}")
        return


if __name__ == "__main__":
    # Default parameters
    csv_path = "tesla_stock_data.csv"  # Replace with your CSV path
    epochs = 25
    batch_size = 32

    # You can modify these parameters
    main(csv_path, epochs=epochs, batch_size=batch_size)