"""
data_utils.py

Utility module for loading and preprocessing stock data for LSTM training.
Handles CSV loading, data cleaning, scaling, and sequence creation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


def load_and_preprocess_data(csv_path):
    """
    Load CSV data and preprocess for LSTM training.

    Args:
        csv_path: Path to the CSV file containing stock data

    Returns:
        tuple: (X_train, y_train, scaler) where
               X_train: Training sequences (n_samples, 60, n_features)
               y_train: Target values (n_samples,)
               scaler: Fitted MinMaxScaler object
    """
    print("🔄 Loading data from CSV...")

    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows of data")
    except Exception as e:
        raise ValueError(f"❌ Error loading CSV file: {e}")

    # Basic validation
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"❌ CSV must contain columns: {required_cols}")

    print("🔄 Preprocessing data...")

    # Preprocess data
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')  # Ensure chronological order
    df.set_index('Date', inplace=True)

    # Handle missing values
    df = df.ffill()  # Forward fill
    df = df.dropna()  # Drop any remaining NaN

    if len(df) < 100:
        raise ValueError("❌ Insufficient data: Need at least 100 data points for training")

    print(f"✅ Preprocessed data: {len(df)} rows, {len(df.columns)} features")

    # Feature selection
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_col = 'Close'

    print("🔄 Scaling data...")

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols].values)

    print("🔄 Creating training sequences...")

    # Create sequences
    X, y = create_dataset(scaled_data, time_step=60, target_index=feature_cols.index(target_col))

    if len(X) == 0:
        raise ValueError("❌ Not enough data to create sequences. Need at least 61 data points.")

    print(f"✅ Created {len(X)} training sequences with shape {X.shape}")

    return X, y, scaler


def create_dataset(dataset, time_step=60, target_index=3):
    """
    Create sequences for LSTM training.

    Args:
        dataset: numpy array of scaled data (n_samples, n_features)
        time_step: number of time steps for sequence (default: 60)
        target_index: column index to predict (default: 3 for Close)

    Returns:
        tuple: (X, y) where X is sequences and y is targets
    """
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step)])
        y.append(dataset[i + time_step, target_index])
    return np.array(X), np.array(y)


def save_scaler(scaler, filepath='scaler.pkl'):
    """
    Save the fitted scaler to disk.

    Args:
        scaler: Fitted MinMaxScaler object
        filepath: Path to save the scaler
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving scaler: {e}")


def load_scaler(filepath='scaler.pkl'):
    """
    Load scaler from disk.

    Args:
        filepath: Path to the saved scaler

    Returns:
        MinMaxScaler object
    """
    try:
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {filepath}")
        return scaler
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    csv_path = "tesla_stock_data.csv"  # Replace with your CSV path
    if os.path.exists(csv_path):
        X_train, y_train, scaler = load_and_preprocess_data(csv_path)
        save_scaler(scaler)
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    else:
        print(f"❌ CSV file not found: {csv_path}")