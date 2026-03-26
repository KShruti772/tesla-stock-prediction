"""
model_utils.py

Utility module for safe model loading and management.
Handles model initialization, error handling, and caching.
"""

import streamlit as st
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape=(60, 5)):
    """
    Build and compile LSTM model architecture.
    
    Args:
        input_shape: tuple specifying input shape (default: (60, 5) for 5 features)
    
    Returns:
        Compiled Sequential model
    """
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
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to build LSTM model: {e}")


@st.cache_resource
def load_trained_lstm_model():
    """
    Load pre-trained LSTM model.
    Cached at session level for performance.
    
    Returns:
        Loaded model or None if loading fails
    """
    try:
        if not os.path.exists("lstm_model.weights.h5"):
            return None
        
        model = build_lstm_model()
        model.load_weights("lstm_model.weights.h5")
        return model
    
    except FileNotFoundError:
        return None
    except Exception as e:
        return None


def validate_model(model, model_name="model"):
    """
    Validate that a model loaded successfully.
    
    Args:
        model: Model instance or None
        model_name: Name of model for error messages
    
    Returns:
        bool: True if model is valid, False otherwise
    """
    return model is not None


def get_model_status(lstm_model):
    """
    Get status of loaded LSTM model for display.
    
    Args:
        lstm_model: Loaded LSTM model or None
    
    Returns:
        dict with model status information
    """
    return {
        "lstm_available": validate_model(lstm_model),
        "has_models": validate_model(lstm_model)
    }
