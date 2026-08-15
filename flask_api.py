from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path("models")

LINEAR_MODEL_PATH = MODEL_DIR / "linear_regression.pkl"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "lstm_scaler.pkl"

SEQ_LENGTH = 60


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading trained models...")

linear_model = joblib.load(LINEAR_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)
lstm_scaler = joblib.load(SCALER_PATH)

print("All models loaded successfully.")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "service": "Finance Forecaster API",
        "models": [
            "Linear Regression",
            "LSTM"
        ]
    })


# ============================================================
# DOWNLOAD STOCK DATA
# ============================================================

def get_stock_data(ticker):

    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(
            f"No market data found for ticker: {ticker}"
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


# ============================================================
# LINEAR REGRESSION PREDICTION
# ============================================================

def linear_prediction(data):

    df = data.copy()

    df["Previous_Close"] = df["Close"].shift(1)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Daily_Return"] = df["Close"].pct_change()

    df = df.dropna()

    features = [
        "Previous_Close",
        "MA_5",
        "MA_10",
        "MA_20",
        "Daily_Return",
        "Volume"
    ]

    latest_features = df[features].iloc[-1:]

    prediction = linear_model.predict(
        latest_features
    )[0]

    return float(prediction)


# ============================================================
# LSTM PREDICTION
# ============================================================

def lstm_prediction(data, forecast_days):

    close_prices = data["Close"].values.reshape(-1, 1)

    scaled_prices = lstm_scaler.transform(
        close_prices
    )

    sequence = scaled_prices[-SEQ_LENGTH:]

    predictions = []

    for _ in range(forecast_days):

        model_input = sequence.reshape(
            1,
            SEQ_LENGTH,
            1
        )

        next_prediction = lstm_model.predict(
            model_input,
            verbose=0
        )

        predicted_price = lstm_scaler.inverse_transform(
            next_prediction
        )[0][0]

        predictions.append(
            float(predicted_price)
        )

        sequence = np.append(
            sequence[1:],
            next_prediction
        ).reshape(SEQ_LENGTH, 1)

    return predictions


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        body = request.get_json()

        ticker = body.get(
            "ticker",
            "GOOG"
        ).upper()

        forecast_days = int(
            body.get(
                "forecast_days",
                7
            )
        )

        if forecast_days < 1 or forecast_days > 30:
            return jsonify({
                "error": "forecast_days must be between 1 and 30"
            }), 400

        data = get_stock_data(ticker)

        current_price = float(
            data["Close"].iloc[-1]
        )

        linear_pred = linear_prediction(
            data
        )

        lstm_preds = lstm_prediction(
            data,
            forecast_days
        )

        return jsonify({

            "ticker": ticker,

            "current_price": current_price,

            "linear_regression_prediction":
                linear_pred,

            "lstm_forecast":
                lstm_preds,

            "forecast_days":
                forecast_days

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )