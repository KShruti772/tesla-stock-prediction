# 🚗 Tesla Stock Price Prediction using LSTM

## 📌 Overview
This project predicts Tesla stock prices using Deep Learning (RNN & LSTM) and visualizes results through a Streamlit dashboard.

## 🎯 Features
- Historical stock visualization
- LSTM-based prediction
- Future price forecasting (1, 5, 10 days)
- Model comparison (RNN vs LSTM)
- Error metrics (MSE, MAE, RMSE)

## 🧠 Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- Pandas, NumPy
- Matplotlib, Plotly

## ⚙️ How to Run

1. Install dependencies
```bash
pip install -r requirements.txt

2. Train model
python train_model.py

3. Run dashboard
streamlit run app.py


Run train_model.py to generate model


📊 Results
LSTM performs better than RNN
Model captures trends effectively
Predictions are smooth (limitation)

⚠️ Limitations
Cannot predict sudden market changes
Uses only historical data

💼 Use Cases
Investment decision support
Stock trend analysis
Risk management