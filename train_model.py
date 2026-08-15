import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


# ============================================================
# CONFIGURATION
# ============================================================

TICKER = "GOOG"
PERIOD = "5y"

SEQ_LENGTH = 60
TRAIN_RATIO = 0.80

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LINEAR_MODEL_PATH = MODEL_DIR / "linear_regression.pkl"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "lstm_scaler.pkl"

RANDOM_STATE = 42


# ============================================================
# 1. DOWNLOAD STOCK DATA
# ============================================================

def download_data():
    print(f"\nDownloading {TICKER} stock data...")

    data = yf.download(
        TICKER,
        period=PERIOD,
        interval="1d",
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(
            f"No stock data was downloaded for {TICKER}."
        )

    # yfinance can return MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()

    print(f"Downloaded {len(data)} rows.")

    return data


# ============================================================
# 2. FEATURE ENGINEERING FOR LINEAR REGRESSION
# ============================================================

def create_features(data):
    df = data.copy()

    # Previous closing price
    df["Previous_Close"] = df["Close"].shift(1)

    # Moving averages
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    # Daily percentage return
    df["Daily_Return"] = df["Close"].pct_change()

    # Remove rows created with NaN values
    df = df.dropna()

    return df


# ============================================================
# 3. TRAIN LINEAR REGRESSION
# ============================================================

def train_linear_regression(data):
    print("\n" + "=" * 60)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 60)

    features = [
        "Previous_Close",
        "MA_5",
        "MA_10",
        "MA_20",
        "Daily_Return",
        "Volume"
    ]

    X = data[features]
    y = data["Close"]

    # Chronological split
    split_index = int(len(data) * TRAIN_RATIO)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nLinear Regression Results")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Save model
    joblib.dump(model, LINEAR_MODEL_PATH)

    print(f"\nLinear Regression model saved to:")
    print(LINEAR_MODEL_PATH)

    return {
        "model": model,
        "predictions": predictions,
        "actual": y_test.values,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ============================================================
# 4. PREPARE DATA FOR LSTM
# ============================================================

def prepare_lstm_data(data):
    print("\nPreparing LSTM sequences...")

    close_prices = data["Close"].values.reshape(-1, 1)

    # Fit scaler only on training data
    split_index = int(len(close_prices) * TRAIN_RATIO)

    train_prices = close_prices[:split_index]

    scaler = MinMaxScaler()

    scaler.fit(train_prices)

    scaled_prices = scaler.transform(close_prices)

    X = []
    y = []

    # Create sequences
    for i in range(
        SEQ_LENGTH,
        len(scaled_prices)
    ):
        X.append(
            scaled_prices[
                i - SEQ_LENGTH:i
            ]
        )

        y.append(
            scaled_prices[i]
        )

    X = np.array(X)
    y = np.array(y)

    # Adjust split because sequences start after SEQ_LENGTH
    lstm_split = split_index - SEQ_LENGTH

    X_train = X[:lstm_split]
    X_test = X[lstm_split:]

    y_train = y[:lstm_split]
    y_test = y[lstm_split:]

    print(f"LSTM training sequences: {len(X_train)}")
    print(f"LSTM testing sequences : {len(X_test)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )


# ============================================================
# 5. TRAIN LSTM
# ============================================================

def train_lstm(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler
):
    print("\n" + "=" * 60)
    print("TRAINING LSTM")
    print("=" * 60)

    model = Sequential([
        Input(
            shape=(SEQ_LENGTH, 1)
        ),

        LSTM(
            64,
            return_sequences=True
        ),

        LSTM(
            32
        ),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    print("\nTraining LSTM model...")

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Predict test data
    scaled_predictions = model.predict(
        X_test,
        verbose=0
    )

    # Convert predictions back to actual prices
    predictions = scaler.inverse_transform(
        scaled_predictions
    ).flatten()

    actual = scaler.inverse_transform(
        y_test
    ).flatten()

    # Evaluation
    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    r2 = r2_score(
        actual,
        predictions
    )

    print("\nLSTM Results")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Save model
    model.save(LSTM_MODEL_PATH)

    # Save scaler
    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print(f"\nLSTM model saved to:")
    print(LSTM_MODEL_PATH)

    print("\nLSTM scaler saved to:")
    print(SCALER_PATH)

    return {
        "model": model,
        "predictions": predictions,
        "actual": actual,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ============================================================
# 6. COMPARE MODELS
# ============================================================

def compare_models(
    linear_results,
    lstm_results
):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "LSTM"
        ],
        "MAE": [
            linear_results["mae"],
            lstm_results["mae"]
        ],
        "RMSE": [
            linear_results["rmse"],
            lstm_results["rmse"]
        ],
        "R2": [
            linear_results["r2"],
            lstm_results["r2"]
        ]
    })

    print(
        comparison.to_string(
            index=False
        )
    )

    # Determine best model by RMSE
    if (
        lstm_results["rmse"]
        <
        linear_results["rmse"]
    ):
        best_model = "LSTM"
    else:
        best_model = "Linear Regression"

    print(
        f"\nBest model based on RMSE: {best_model}"
    )

    return comparison


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINANCE FORECASTER")
    print("Linear Regression + LSTM")
    print("=" * 60)

    # Download data
    data = download_data()

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    feature_data = create_features(data)

    linear_results = train_linear_regression(
        feature_data
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = prepare_lstm_data(data)

    lstm_results = train_lstm(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison = compare_models(
        linear_results,
        lstm_results
    )

    # Save comparison
    comparison.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    print("\nModel comparison saved to:")
    print(
        MODEL_DIR / "model_comparison.csv"
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()