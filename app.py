from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    symbol = 'AAPL'
    model = load_model(f'model/{symbol}_lstm.h5')
    scaler = joblib.load(f'model/{symbol}_scaler.save')

    df = yf.download(symbol, period='1y')[['Close']]
    scaled_data = scaler.transform(df)
    
    last_60 = scaled_data[-60:]
    X_input = np.reshape(last_60, (1, 60, 1))
    prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction)

    return jsonify({
        'symbol': symbol,
        'predicted_price': round(float(prediction[0][0]), 2),
        'last_price': round(float(df['Close'].iloc[-1]), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
