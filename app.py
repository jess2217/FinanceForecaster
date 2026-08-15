import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta


# ============================================================
# CONFIGURATION
# ============================================================

FLASK_API_URL = "http://127.0.0.1:5000"

VALID_TICKERS = [
    "AAPL",
    "GOOG",
    "MSFT",
    "AMZN",
    "TSLA",
    "RELIANCE.NS",
    "TCS.NS"
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinanceForecaster",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 FinanceForecaster")

st.markdown(
    """
    ### Stock Price Prediction using Regression & LSTM

    FinanceForecaster compares **Linear Regression** and
    **LSTM Neural Networks** to generate short-term stock
    price forecasts using historical market data.
    """
)


# ============================================================
# FLASK API CHECK
# ============================================================

def check_api():

    try:

        response = requests.get(
            f"{FLASK_API_URL}/health",
            timeout=5
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:

        return False


# ============================================================
# GET PREDICTION FROM FLASK
# ============================================================

def get_prediction(ticker, forecast_days):

    payload = {
        "ticker": ticker,
        "forecast_days": forecast_days
    }

    response = requests.post(
        f"{FLASK_API_URL}/predict",
        json=payload,
        timeout=120
    )

    if response.status_code != 200:

        try:

            error = response.json().get(
                "error",
                "Unknown API error"
            )

        except Exception:

            error = response.text

        raise Exception(error)

    return response.json()


# ============================================================
# GET HISTORICAL DATA FOR CHART
# ============================================================

def get_historical_data(ticker):

    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    return data.dropna()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Forecast Settings")

    ticker = st.selectbox(
        "Select Stock",
        VALID_TICKERS,
        index=1
    )

    custom_ticker = st.text_input(
        "Or enter another ticker",
        placeholder="Example: NFLX"
    ).upper()

    if custom_ticker:
        ticker = custom_ticker

    forecast_days = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=7
    )

    st.markdown("---")

    st.markdown(
        """
        **Models used**

        🔵 Linear Regression

        🟠 LSTM Neural Network
        """
    )


# ============================================================
# API STATUS
# ============================================================

if check_api():

    st.success("🟢 Flask API is connected")

else:

    st.error(
        "🔴 Flask API is not running. "
        "Start `flask_api.py` first."
    )

    st.code(
        "python flask_api.py"
    )

    st.stop()


# ============================================================
# RUN PREDICTION
# ============================================================

if st.button(
    "🚀 Run Forecast",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        f"Fetching {ticker} data and generating forecast..."
    ):

        try:

            # ------------------------------------------------
            # Get prediction from Flask
            # ------------------------------------------------

            result = get_prediction(
                ticker,
                forecast_days
            )

            # ------------------------------------------------
            # Extract results
            # ------------------------------------------------

            current_price = result[
                "current_price"
            ]

            linear_prediction = result[
                "linear_regression_prediction"
            ]

            lstm_forecast = result[
                "lstm_forecast"
            ]

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            st.subheader(
                f"📊 {ticker} Forecast"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Current Price",
                f"${current_price:,.2f}"
            )

            col2.metric(
                "Linear Regression",
                f"${linear_prediction:,.2f}",
                delta=(
                    f"${linear_prediction - current_price:,.2f}"
                )
            )

            col3.metric(
                f"LSTM ({forecast_days}-Day)",
                f"${lstm_forecast[-1]:,.2f}",
                delta=(
                    f"${lstm_forecast[-1] - current_price:,.2f}"
                )
            )

            # ------------------------------------------------
            # Historical Data
            # ------------------------------------------------

            historical_data = get_historical_data(
                ticker
            )

            if historical_data is None:

                st.warning(
                    "Historical chart data could not be loaded."
                )

            else:

                # Last 90 days for cleaner chart
                chart_data = historical_data.tail(90)

                historical_dates = chart_data.index

                historical_prices = (
                    chart_data["Close"].values
                )

                # ------------------------------------------------
                # Future dates
                # ------------------------------------------------

                last_date = historical_dates[-1]

                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=forecast_days,
                    freq="D"
                )

                # ------------------------------------------------
                # Plot
                # ------------------------------------------------

                fig = go.Figure()

                # Historical price
                fig.add_trace(
                    go.Scatter(
                        x=historical_dates,
                        y=historical_prices,
                        name="Historical Price",
                        mode="lines"
                    )
                )

                # Linear Regression prediction
                fig.add_trace(
                    go.Scatter(
                        x=[
                            last_date,
                            future_dates[0]
                        ],
                        y=[
                            current_price,
                            linear_prediction
                        ],
                        name="Linear Regression",
                        mode="lines+markers",
                        line=dict(
                            dash="dash"
                        )
                    )
                )

                # LSTM forecast
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=lstm_forecast,
                        name="LSTM Forecast",
                        mode="lines+markers",
                        line=dict(
                            dash="dot"
                        )
                    )
                )

                fig.update_layout(
                    title=(
                        f"{ticker} Stock Price Forecast"
                    ),
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode="x unified",
                    height=600,
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ====================================================
            # FORECAST TABLE
            # ====================================================

            st.subheader(
                "🔮 LSTM Forecast"
            )

            forecast_table = pd.DataFrame({
                "Forecast Day": range(
                    1,
                    forecast_days + 1
                ),
                "Predicted Price": [
                    round(price, 2)
                    for price in lstm_forecast
                ]
            })

            st.dataframe(
                forecast_table,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Download forecast
            # ------------------------------------------------

            csv_data = forecast_table.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv"
            )

            # ====================================================
            # INTERPRETATION
            # ====================================================

            st.subheader(
                "📌 Forecast Summary"
            )

            lstm_change = (
                (
                    lstm_forecast[-1]
                    / current_price
                ) - 1
            ) * 100

            linear_change = (
                (
                    linear_prediction
                    / current_price
                ) - 1
            ) * 100

            st.write(
                f"**Linear Regression:** "
                f"{linear_change:+.2f}% "
                f"from the current price."
            )

            st.write(
                f"**LSTM:** "
                f"{lstm_change:+.2f}% "
                f"over the {forecast_days}-day "
                f"forecast period."
            )

            st.info(
                "⚠️ These predictions are generated "
                "by machine-learning models for "
                "educational purposes and should not "
                "be treated as financial advice."
            )

        except Exception as e:

            st.error(
                f"❌ Forecast failed: {str(e)}"
            )


else:

    st.info(
        "Select a stock and forecast period, "
        "then click **Run Forecast**."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader(
    "🤖 Model Performance"
)

comparison_path = "models/model_comparison.csv"

try:

    comparison = pd.read_csv(
        comparison_path
    )

    st.dataframe(
        comparison.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Determine best model
    best_model = comparison.loc[
        comparison["RMSE"].idxmin(),
        "Model"
    ]

    st.success(
        f"🏆 Best performing model based on RMSE: "
        f"**{best_model}**"
    )

except Exception as e:

    st.warning(
        f"Model performance data could not be loaded: {e}"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "FinanceForecaster | "
    "Linear Regression + LSTM | "
    "Flask + Streamlit"
)