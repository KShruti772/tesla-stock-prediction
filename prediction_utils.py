"""
prediction_utils.py

Utility module for handling predictions and preprocessing.
Manages prediction generation, future forecasting, and data rescaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_dataset(dataset, time_step=60, target_index=3):
    """
    Create sequences for LSTM training.
    
    Args:
        dataset: numpy array of scaled data
        time_step: number of time steps for sequence (default: 60)
        target_index: column index to predict (default: 3 for Close)
    
    Returns:
        tuple of (X, y) where X is sequences and y is targets
    """
    try:
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step)])
            y.append(dataset[i + time_step, target_index])
        return np.array(X), np.array(y)
    except Exception as e:
        raise ValueError(f"Error creating dataset: {e}")


def inverse_close_values(scaled_closes, scaler, close_index):
    """
    Inverse transform scaled close prices back to original scale.
    
    Args:
        scaled_closes: array of scaled close prices (n_samples, 1)
        scaler: fitted MinMaxScaler object
        close_index: index of close column in original features
    
    Returns:
        numpy array of unscaled close prices
    """
    try:
        n = len(scaled_closes)
        batch = np.zeros((n, scaler.n_features_in_))
        batch[:, close_index] = scaled_closes.flatten()
        inv = scaler.inverse_transform(batch)
        return inv[:, close_index].reshape(-1, 1)
    except Exception as e:
        raise ValueError(f"Error inverse transforming values: {e}")


def predict_future(model, data, days, scaler, close_index):
    """
    Generate future price predictions using the trained model.
    
    Args:
        model: Trained neural network model
        data: Last 60 days of scaled data (numpy array)
        days: Number of days to predict (integer)
        scaler: Fitted MinMaxScaler object
        close_index: Index of close column in features
    
    Returns:
        numpy array of predicted prices in original scale (n_days, 1)
    
    Raises:
        ValueError: If prediction generation fails
    """
    try:
        if model is None:
            raise ValueError("Model is None. Cannot generate predictions.")
        
        n_features = data.shape[1]
        temp_input = data.copy()
        output_scaled = []

        for _ in range(days):
            x_input = temp_input[-60:].reshape(1, 60, n_features)
            yhat = model.predict(x_input, verbose=0)[0][0]

            next_row = temp_input[-1].copy()
            next_row[close_index] = yhat
            temp_input = np.vstack((temp_input, next_row))

            output_scaled.append(yhat)

        output_scaled = np.array(output_scaled).reshape(-1, 1)
        return inverse_close_values(output_scaled, scaler, close_index)
    
    except Exception as e:
        raise ValueError(f"Error generating predictions: {e}")


def generate_future_dates(last_date, days):
    """
    Generate business day dates for predictions.
    
    Args:
        last_date: Last date in historical data (pandas Timestamp or string)
        days: Number of business days to generate
    
    Returns:
        pandas DatetimeIndex of future business days
    """
    try:
        last_date = pd.to_datetime(last_date)
        future_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
        return future_dates
    except Exception as e:
        raise ValueError(f"Error generating future dates: {e}")


def create_prediction_dataframe(predicted_prices, future_dates):
    """
    Create a clean DataFrame for predicted prices.
    
    Args:
        predicted_prices: numpy array of predicted prices
        future_dates: pandas DatetimeIndex of dates
    
    Returns:
        pandas DataFrame with dates as index and predictions as values
    """
    try:
        if len(predicted_prices) != len(future_dates):
            raise ValueError(f"Predictions length ({len(predicted_prices)}) doesn't match dates ({len(future_dates)})")
        
        future_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Price'])
        return future_df
    except Exception as e:
        raise ValueError(f"Error creating prediction dataframe: {e}")


def validate_prediction_data(scaled_data, days, scaler, close_index):
    """
    Validate input data before prediction.
    
    Args:
        scaled_data: Scaled data array
        days: Number of days to predict
        scaler: MinMaxScaler fitted on original data
        close_index: Index of close column
    
    Returns:
        tuple of (is_valid, error_message)
    """
    errors = []
    
    if scaled_data is None:
        errors.append("Scaled data is None")
    elif len(scaled_data) < 60:
        errors.append("Insufficient data: need at least 60 days")
    
    if days <= 0 or days > 365:
        errors.append("Days must be between 1 and 365")
    
    if scaler is None:
        errors.append("Scaler is not initialized")
    
    if close_index < 0:
        errors.append("Invalid close index")
    
    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    
    return is_valid, error_message
