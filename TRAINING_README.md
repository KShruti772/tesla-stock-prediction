# LSTM Stock Prediction Training Pipeline

This directory contains a complete local training pipeline for LSTM stock price prediction, separate from the Streamlit app.

## Files Created:

### `data_utils.py`
- **Purpose**: Data loading and preprocessing utilities
- **Functions**:
  - `load_and_preprocess_data(csv_path)`: Load CSV, preprocess, and create training sequences
  - `create_dataset()`: Create time series sequences for LSTM training
  - `save_scaler()` / `load_scaler()`: Save/load MinMaxScaler

### `train_model.py`
- **Purpose**: Complete training pipeline script
- **Functions**:
  - `build_lstm_model()`: Build LSTM architecture (same as app)
  - `train_lstm_model()`: Train the model with progress output
  - `save_model_weights()`: Save trained weights

## Usage:

### 1. Prepare your data:
Place a CSV file with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

### 2. Run training:
```bash
# Default training (25 epochs, batch size 32)
python train_model.py

# Custom parameters
python -c "from train_model import main; main('your_data.csv', epochs=50, batch_size=64)"
```

### 3. Output files:
- `lstm_model.weights.h5`: Trained model weights (compatible with app.py)
- `scaler.pkl`: Fitted MinMaxScaler for data preprocessing

## Features:

✅ **Complete pipeline**: Data loading → preprocessing → training → saving  
✅ **Progress output**: Detailed print statements for monitoring  
✅ **Error handling**: Comprehensive error checking and messages  
✅ **App compatibility**: Same architecture and file formats as Streamlit app  
✅ **No Streamlit dependency**: Pure Python training script  

## Model Architecture:

- **Input shape**: (60, 5) - 60 time steps, 5 features
- **Layers**: LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1)
- **Optimizer**: Adam
- **Loss**: Mean Squared Error

## Example Output:

```
🎯 Starting LSTM Stock Prediction Training Pipeline
==================================================

1. Loading and preprocessing data...
🔄 Loading data from CSV...
✅ Loaded 1000 rows of data
🔄 Preprocessing data...
✅ Preprocessed data: 1000 rows, 5 features
🔄 Scaling data...
🔄 Creating training sequences...
✅ Created 939 training sequences with shape (939, 60, 5)

2. Training LSTM model...
🏗️ Building LSTM model with input shape (60, 5)...
✅ LSTM model built successfully
🚀 Starting LSTM model training...
Epoch 1/25
...
✅ Model training completed successfully
Final loss: 0.0012, Val loss: 0.0021

3. Saving model and scaler...
✅ Model weights saved to lstm_model.weights.h5
✅ Scaler saved to scaler.pkl

🎉 Training pipeline completed successfully!
```

The trained model can be used directly in the Streamlit app without any modifications!