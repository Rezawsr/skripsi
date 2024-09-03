import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from tensorflow.keras.callbacks import EarlyStopping

# Title of the app
st.title('Stock Price Prediction using LSTM and GRU')

# Sidebar for stock tickers
st.sidebar.header('Select Stock Ticker')
stocks = ("BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "MEGA.JK")
selected_stock = st.sidebar.selectbox("Pilih Dataset untuk di prediksi", stocks)

# Sidebar for prediction years
st.sidebar.header('Prediction Settings')
n_years = st.sidebar.slider("Years of Prediction", 1, 4)
period = n_years * 365

# Sidebar for model selection
st.sidebar.header('Select Prediction Model')
model_choice = st.sidebar.radio("Choose Model", ["LSTM", "GRU"])

# Define the start and end dates for data fetching
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Function to load data
import yfinance as yf
import pandas as pd
import time

@st.cache_data
def load_data(ticker, start, end, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                st.error("Data is empty!")
                return pd.DataFrame()
            return data
        except Exception as e:
            attempt += 1
            st.warning(f"Attempt {attempt} failed: {e}")
            time.sleep(delay)  # Wait before retrying
    st.error("Failed to download data after several attempts.")
    return pd.DataFrame()


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close', line=dict(color='red')))
    fig.update_layout(title="Historical Stock Prices", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig)

plot_raw_data()

# Financial Metrics for different stocks
financial_metrics = {
    "BBCA.JK": {
        "Current Ratio": 0.31,
        "LDR": "65.06",
        "CAR": "25.94",
        "ROE": "19.35",
        "ROA": "1.76",
        "EPS": "Rp.388"
    },
    "BBRI.JK": {
        "Current Ratio": 1.12,
        "LDR": "65.50",
        "CAR": "21.94",
        "ROE": "19.10",
        "ROA": "1.41",
        "EPS": "Rp.400"
    },
    "BMRI.JK": {
        "Current Ratio": 0.29,
        "LDR": "64.00",
        "CAR": "23.20",
        "ROE": "19.35",
        "ROA": "1.16",
        "EPS": "Rp.350"
    },
    "BBNI.JK": {
        "Current Ratio": 0.92,
        "LDR": "62.00",
        "CAR": "18.00",
        "ROE": "12.85",
        "ROA": "0.87",
        "EPS": "Rp.280"
    },
    "BRIS.JK": {
        "Current Ratio": 2.86,
        "LDR": "85.00",
        "CAR": "25.00",
        "ROE": "15.73",
        "ROA": "0.76",
        "EPS": "Rp.220"
    },
    "MEGA.JK": {
        "Current Ratio": 2.92,
        "LDR": "70.00",
        "CAR": "24.00",
        "ROE": "18.48",
        "ROA": "2.14",
        "EPS": "Rp.410"
    }
}

# Display KPI metrics based on selected stock
st.subheader("Financial Metrics")
st.markdown("---")
col1, col2, col3 = st.columns(3)

metrics = financial_metrics.get(selected_stock, {})
with col1:
    st.metric(label="**Current Ratio**", value=metrics.get("Current Ratio", "N/A"))

with col2:
    st.metric(label="**LDR**", value=metrics.get("LDR", "N/A"))

with col3:
    st.metric(label="**CAR**", value=metrics.get("CAR", "N/A"))

col4, col5, col6 = st.columns(3)
with col4:
    st.metric(label="**ROE**", value=metrics.get("ROE", "N/A"))

with col5:
    st.metric(label="**ROA**", value=metrics.get("ROA", "N/A"))

with col6:
    st.metric(label="**EPS**", value=metrics.get("EPS", "N/A"))

# Rekomendasi Berdasarkan Metrik Keuangan
st.subheader("Rekomendasi dan KPI")
st.markdown("---")

def give_recommendation(metrics):
    roe = float(metrics.get("ROE", "N/A").replace(",", ".")) if metrics.get("ROE") != "N/A" else None
    roa = float(metrics.get("ROA", "N/A").replace(",", ".")) if metrics.get("ROA") != "N/A" else None
    car = float(metrics.get("CAR", "N/A").replace(",", ".")) if metrics.get("CAR") != "N/A" else None

    recommendations = []
    potential = True  # Default is that the stock has potential

    if roe and roe > 10:
        recommendations.append("ROE yang tinggi (>10%)")
    else:
        recommendations.append("ROE yang rendah")
        potential = False

    if roa and roa > 1.5:
        recommendations.append("ROA yang baik (>1.5%)")
    else:
        recommendations.append("ROA yang rendah")
        potential = False

    if car and car > 20:
        recommendations.append("CAR yang tinggi (>20%)")
    else:
        recommendations.append("CAR yang rendah")
        potential = False

    return recommendations, potential

recommendations, is_potential = give_recommendation(metrics)

# Displaying recommendations in boxes
st.markdown("### Financial Health Indicators")
st.markdown("#### Potensi Saham Berdasarkan Metrik Keuangan:")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div style='padding:10px; border: 2px solid black; border-radius:10px; background-color: {'#90EE90' if 'ROE yang tinggi' in recommendations[0] else '#FF6F61'};'>{recommendations[0]}</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div style='padding:10px; border: 2px solid black; border-radius:10px; background-color: {'#90EE90' if 'ROA yang baik' in recommendations[1] else '#FF6F61'};'>{recommendations[1]}</div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div style='padding:10px; border: 2px solid black; border-radius:10px; background-color: {'#90EE90' if 'CAR yang tinggi' in recommendations[2] else '#FF6F61'};'>{recommendations[2]}</div>", unsafe_allow_html=True)

# KPI based on potential
st.markdown("---")
st.markdown("### KPI Analysis")

kpi_color = "green" if is_potential else "red"
kpi_message = "Saham ini memiliki potensi yang baik." if is_potential else "Saham ini memerlukan perhatian lebih."

st.markdown(f"<div style='text-align: center; padding:10px; border: 2px solid black; border-radius:10px; background-color: {kpi_color}; color:white;'>{kpi_message}</div>", unsafe_allow_html=True)

# Prepare data for LSTM and GRU
if 'Close' not in data.columns or data.empty:
    st.error("Data not available or 'Close' column is missing!")
else:
    data = data[['Date', 'Close']]
    data.set_index('Date', inplace=True)
    data = data[['Close']]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Convert the data to a supervised learning problem
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    # Define time step
    time_step = 60

    X, y = create_dataset(scaled_data, time_step)

    # Split the data into training and validation sets
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_valid, y_valid = X[split:], y[split:]

    # Reshape data for LSTM and GRU [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

    # Function to build and train LSTM model
    def build_lstm_model(X_train, y_train, X_valid, y_valid):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32, verbose=1, callbacks=[early_stop])
        return model

    # Function to build and train GRU model
    def build_gru_model(X_train, y_train, X_valid, y_valid):
        model = Sequential()
        model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32, verbose=1, callbacks=[early_stop])
        return model

    # Function to make predictions and forecast future prices
    def predict_future(model, data, scaler, time_step, period):
        last_sequence = data[-time_step:].values
        last_sequence = scaler.transform(last_sequence)
        predicted_prices = []

        for _ in range(period):
            y_pred = model.predict(last_sequence.reshape(1, time_step, 1))
            predicted_prices.append(y_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], [[y_pred[0, 0]]], axis=0)

        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        future_df = pd.DataFrame({
            'Date': pd.date_range(start=TODAY, periods=period),
            'Predicted Close': predicted_prices.flatten()
        })
        return future_df

    # Select and train model based on user choice
    if model_choice == "LSTM":
        st.subheader("Training LSTM Model")
        lstm_model = build_lstm_model(X_train, y_train, X_valid, y_valid)
        future_df = predict_future(lstm_model, data, scaler, time_step, period)
        model_name = "LSTM"
        line_color = 'orange'
    else:
        st.subheader("Training GRU Model")
        gru_model = build_gru_model(X_train, y_train, X_valid, y_valid)
        future_df = predict_future(gru_model, data, scaler, time_step, period)
        model_name = "GRU"
        line_color = 'green'

    # Plot future predictions
    st.subheader(f"Future Predictions ({model_name})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Close'], mode='lines', name=f'{model_name} Predicted Data', line=dict(color=line_color)))
    fig.update_layout(title=f"{model_name} Stock Price Predictions", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig)
