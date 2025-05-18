import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf

# Load historical stock data
def load_data(symbol='AAPL', period='5y'):
    df = yf.download(symbol, period=period)
    df = df[['Close']]
    return df

# Prepare data for LSTM
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(scaled)):
        X.append(scaled[i-time_step:i, 0])
        y.append(scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Train model
def train_model(symbol='AAPL'):
    df = load_data(symbol)
    X, y, scaler = prepare_data(df)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=10, batch_size=32)
    model.save(f'model/{symbol}_lstm.h5')
    joblib.dump(scaler, f'model/{symbol}_scaler.save')
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()
