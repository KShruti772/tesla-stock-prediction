import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import os

# Import utility modules
from model_utils import (
    build_lstm_model,
    load_trained_lstm_model,
    validate_model,
    get_model_status
)
from prediction_utils import (
    create_dataset,
    inverse_close_values,
    predict_future,
    generate_future_dates,
    create_prediction_dataframe,
    validate_prediction_data
)

# Page configuration
st.set_page_config(
    page_title="Tesla Stock Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
        margin: 0.5rem 0;
        text-color: black;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚗 Tesla Stock Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Prediction", "LSTM Model Analysis", "Train Model", "Insights"])

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("Professional Tesla stock price prediction dashboard using LSTM neural networks.")

# Functions
def preprocess_data(raw_df):
    data = raw_df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    data = data.dropna()
    return data


def train_model(model, X_data, y_data, name, epochs=25, batch_size=32):
    with st.spinner(f"Training {name} model... This may take a few minutes."):
        model.fit(X_data, y_data, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def calculate_metrics(actual, predicted):
    return mean_squared_error(actual, predicted)


def build_lstm_model(input_shape=(60, 5)):
    """
    Build and compile LSTM model architecture.

    Args:
        input_shape: tuple specifying input shape (default: (60, 5) for 5 features)

    Returns:
        Compiled Sequential model
    """
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


def train_lstm(X, y):
    model = build_lstm_model((60, X.shape[2]))
    return train_model(model, X, y, "LSTM", epochs=25, batch_size=32)


# Load model once (cached) - Wrapper for backward compatibility
@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists("lstm_model.weights.h5"):
            st.error("❌ Model file 'lstm_model.weights.h5' not found. Run train_model.py first.")
            return None

        model = build_lstm_model()
        model.load_weights("lstm_model.weights.h5")

        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None



# Main content based on navigation
if page == "Home":
    st.title("🏠 Home")
    st.markdown('<h1 class="main-header">🚗 Tesla Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)

    st.write("This professional dashboard uses LSTM neural networks to predict Tesla stock prices based on historical data.")
    st.write("**Features:** Historical data visualization, future price predictions, model analysis, and key metrics.")
    st.info("📁 Navigate to Prediction or LSTM Model Analysis pages to upload data and start analysis.")

    uploaded_file_home = st.file_uploader("📁 Upload Tesla CSV for historical display", type=["csv"], key="home_uploader")
    if uploaded_file_home:
        raw_df = pd.read_csv(uploaded_file_home)

        if 'Date' not in raw_df.columns:
            st.error("CSV must contain a 'Date' column.")
        else:
            raw_df['Date'] = pd.to_datetime(raw_df['Date'])
            raw_df = raw_df.sort_values('Date')
            close_col = 'Close' if 'Close' in raw_df.columns else 'Adj Close' if 'Adj Close' in raw_df.columns else None

            if close_col is None:
                st.error("CSV must contain a 'Close' or 'Adj Close' column.")
            else:
                raw_df = raw_df.dropna(subset=[close_col])
                raw_df.set_index('Date', inplace=True)

                st.subheader("📈 Historical Stock Data")
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(x=raw_df.index, y=raw_df[close_col], mode='lines', name='Close', line=dict(color='#1f77b4', width=2)))
                fig_history.update_layout(title='Tesla Stock Price History', xaxis_title='Date', yaxis_title=f'{close_col} ($)', template='plotly_dark', hovermode='x unified')
                st.plotly_chart(fig_history, use_container_width=True)

                st.subheader("📊 Key Statistics")
                latest_price = raw_df[close_col].iloc[-1]
                highest_price = raw_df[close_col].max()
                lowest_price = raw_df[close_col].min()
                average_price = raw_df[close_col].mean()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Latest Price", f"${latest_price:.2f}")
                col2.metric("Highest Price", f"${highest_price:.2f}")
                col3.metric("Lowest Price", f"${lowest_price:.2f}")
                col4.metric("Average Price", f"${average_price:.2f}")

                st.markdown("---")

    st.subheader("⚙️ Feature Engineering")
    st.write("""
    - Moving Average (10 days)
    - Moving Average (50 days)
    - Helps in identifying trends
    """)

elif page == "Prediction":
    st.title("📈 Prediction")

    uploaded_file_pred = st.file_uploader("📁 Upload Tesla CSV", type=["csv"], key="pred_uploader")

    if uploaded_file_pred:
        df = preprocess_data(pd.read_csv(uploaded_file_pred))

        st.subheader("⚙️ Feature Engineering")
        if 'Close' in df.columns:
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            st.line_chart(df[['Close', 'MA_10', 'MA_50']].dropna())

        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in feature_cols):
            st.error("CSV must contain Open, High, Low, Close, Volume columns for advanced modeling.")
        else:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[feature_cols].values)

            close_index = feature_cols.index('Close')

            # Load pre-trained model
            try:
                trained_model_lstm = load_trained_model()

                if trained_model_lstm is None:
                    st.error("❌ Pre-trained LSTM model not found. Please run train_model.py first to generate 'lstm_model.weights.h5'.")
                else:
                    st.success("✅ Pre-trained LSTM model loaded successfully.")

                st.subheader("🔮 Future Price Predictions")
                days = st.selectbox("Select prediction days:", [1, 5, 10], index=1)

                if st.button("Generate Prediction"):
                    try:
                        last_60_days = scaled_data[-60:]
                        pred_lstm_future = predict_future(trained_model_lstm, last_60_days, days, scaler, close_index)

                        last_date = pd.to_datetime(df.index[-1])
                        future_dates = generate_future_dates(last_date, days)

                        future_df = create_prediction_dataframe(pred_lstm_future, future_dates)

                        st.info("Predictions are generated for business days only (stock market trading days).")
                        st.subheader("Predicted Prices")
                        st.dataframe(future_df.style.format("{:.2f}"))

                        st.subheader("Prediction Chart")
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted Price'], mode='lines+markers', name='Predicted Price', line=dict(color='#1f77b4', width=2)))
                        fig_pred.update_layout(title='Future Predicted Prices (Business Days)', xaxis_title='Date', yaxis_title='Predicted Price ($)', template='plotly_dark')
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Prediction generation failed: {str(e)}")
                        st.info("Please ensure data has at least 60 days of history.")

            except Exception as e:
                st.error(f"❌ Model loading failed: {str(e)}")
                st.info("Please check if the model file exists and is properly saved.")


elif page == "LSTM Model Analysis":
    st.title("📊 LSTM Model Analysis")

    uploaded_file_comp = st.file_uploader("📁 Upload Tesla CSV", type=["csv"], key="comp_uploader")

    if uploaded_file_comp:
        df = preprocess_data(pd.read_csv(uploaded_file_comp))

        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in feature_cols):
            st.error("CSV must contain Open, High, Low, Close, Volume columns for advanced modeling.")
        else:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[feature_cols].values)

            close_index = feature_cols.index('Close')
            X, y = create_dataset(scaled_data, 60, target_index=close_index)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

            # Load pre-trained model
            trained_model_lstm = load_trained_model()

            if trained_model_lstm is None:
                st.error("❌ Pre-trained LSTM model not found. Please run train_model.py first to generate 'lstm_model.weights.h5'.")
            else:
                try:
                    st.success("✅ Pre-trained LSTM model loaded successfully.")

                    train_pred_lstm = trained_model_lstm.predict(X, verbose=0)

                    y_unscaled = inverse_close_values(y.reshape(-1, 1), scaler, close_index).flatten()
                    train_pred_lstm_unscaled = inverse_close_values(train_pred_lstm, scaler, close_index).flatten()

                    mse = calculate_metrics(y_unscaled, train_pred_lstm_unscaled)
                    mae = mean_absolute_error(y_unscaled, train_pred_lstm_unscaled)
                    rmse = np.sqrt(mse)

                    st.subheader("Model Evaluation")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("MSE", f"{mse:.5f}")
                    col2.metric("MAE", f"{mae:.5f}")
                    col3.metric("RMSE", f"{rmse:.5f}")

                    st.subheader("Actual vs Predicted (Training)")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(y=y_unscaled, mode='lines', name='Actual', line=dict(color='black', width=2)))
                    fig2.add_trace(go.Scatter(y=train_pred_lstm_unscaled, mode='lines', name='LSTM Predicted', line=dict(color='blue', width=2)))
                    fig2.update_layout(title='Actual vs Predicted: LSTM Model', xaxis_title='Time Steps', yaxis_title='Price ($)', legend=dict(title='Series', x=0.01, y=0.99), template='plotly_dark')
                    st.plotly_chart(fig2, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Model analysis failed: {str(e)}")
                    st.info("Please ensure data has sufficient records and model is properly trained.")

elif page == "Train Model":
    st.title("🧠 Train Model")

    uploaded_file_train = st.file_uploader("📁 Upload Tesla CSV for Training", type=["csv"], key="train_uploader")

    if uploaded_file_train:
        df = preprocess_data(pd.read_csv(uploaded_file_train))

        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in feature_cols):
            st.error("CSV must contain Open, High, Low, Close, Volume columns.")
        else:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[feature_cols].values)

            close_index = feature_cols.index('Close')
            X, y = create_dataset(scaled_data, 60, target_index=close_index)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

            st.subheader("🧠 Train LSTM Model")
            epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=5)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

            if st.button("Train LSTM Model"):
                try:
                    model_lstm = build_lstm_model()
                    trained_lstm = train_model(model_lstm, X, y, "LSTM", epochs=epochs, batch_size=batch_size)

                    # Save the model
                    trained_lstm.save("final_model.h5")
                    trained_lstm.save_weights("lstm_model.weights.h5")
                    st.success("✅ LSTM model trained and saved successfully! (Weights saved as 'lstm_model.weights.h5')")

                    # Evaluate on training data
                    train_pred = trained_lstm.predict(X, verbose=0)
                    y_unscaled = inverse_close_values(y.reshape(-1, 1), scaler, close_index).flatten()
                    train_pred_unscaled = inverse_close_values(train_pred, scaler, close_index).flatten()

                    mse = mean_squared_error(y_unscaled, train_pred_unscaled)
                    mae = mean_absolute_error(y_unscaled, train_pred_unscaled)
                    rmse = np.sqrt(mse)

                    st.subheader("Training Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MSE", f"{mse:.5f}")
                    col2.metric("MAE", f"{mae:.5f}")
                    col3.metric("RMSE", f"{rmse:.5f}")

                    st.subheader("Actual vs Predicted (Training Data)")
                    fig_train = go.Figure()
                    fig_train.add_trace(go.Scatter(y=y_unscaled, mode='lines', name='Actual', line=dict(color='black', width=2)))
                    fig_train.add_trace(go.Scatter(y=train_pred_unscaled, mode='lines', name='Predicted', line=dict(color='#1f77b4', width=2)))
                    fig_train.update_layout(title='Actual vs Predicted Prices', xaxis_title='Time Steps', yaxis_title='Price ($)', template='plotly_dark')
                    st.plotly_chart(fig_train, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Model training failed: {str(e)}")
                    st.info("Please ensure data is properly formatted and has sufficient records.")

elif page == "Insights":
    st.title("📊 Insights & Conclusion")
    st.write("""
    - LSTM model captures long-term dependencies in stock price data
    - Performs well on sequential time series prediction
    - Captures trends but may not predict sudden market changes
    - Model performance depends on quality and quantity of historical data
    """)

    st.title("💼 Business Use Cases")
    st.write("""
    - Automated stock trading systems
    - Investment decision support tools
    - Risk management and portfolio optimization
    - Financial forecasting and market analysis
    """)

    st.subheader("⚙️ Model Hyperparameters")
    st.write("""
    - **LSTM Units**: 50, 100
    - **Dropout**: 0.2, 0.3
    - **Optimizer**: Adam (default learning rate)
    - **Loss Function**: Mean Squared Error (MSE)
    - **Training Epochs**: 3 (demo setting, increase for production)
    - **Batch Size**: 32
    - **Input Time Steps**: 60
    """)

# Footer
st.markdown("""
<div class="footer" style="color: black;">
    <h4 style="color: #1f77b4;">🚗 Tesla Stock Price Prediction Dashboard</h4>
    <p style="color: black;">This professional dashboard uses LSTM neural networks to predict Tesla stock prices based on historical data. Upload your CSV file to get started with predictions and analysis.</p>
    <p style="color: black;"><strong>Features:</strong> Historical data visualization, future price predictions, model analysis, and key metrics.</p>
</div>
""", unsafe_allow_html=True)