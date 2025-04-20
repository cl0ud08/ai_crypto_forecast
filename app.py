import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------
# 💰 Fetch Real-Time Crypto Data
# ------------------------------
def fetch_crypto_data(symbol, interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades", "taker_buy_base_asset", "taker_buy_quote_asset", "ignore"])

        # Convert time to datetime
        df["ds"] = pd.to_datetime(df["time"], unit="ms")

        # Convert numerical columns to float type for processing
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["y"] = df["close"]  # Target for prediction

        # Return the relevant columns
        return df[["ds", "open", "high", "low", "close", "volume", "y"]]
    else:
        return None

# ------------------------------
# 🔴 Train Random Forest Model
# ------------------------------
def train_random_forest(df, days=7):
    # Ensure 'volume' exists in the DataFrame
    if 'volume' not in df.columns:
        raise KeyError("'volume' column is missing from the DataFrame!")

    # Add technical indicators (like day of the week, hour, and minute)
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["hour"] = df["ds"].dt.hour
    df["minute"] = df["ds"].dt.minute

    df = df.dropna(subset=["y"])  # Drop missing values in the target column

    # Features (X) and Target (y)
    X = df[["open", "high", "low", "close", "volume", "day_of_week", "hour", "minute"]]
    y = df["y"]

    # Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Prepare the last row data to predict the next 'days' predictions
    last_row = df.iloc[-1]  # Get the most recent row

    future_data = {
        "open": [last_row["open"]] * days,
        "high": [last_row["high"]] * days,
        "low": [last_row["low"]] * days,
        "close": [last_row["close"]] * days,
        "volume": [last_row["volume"]] * days,
        "day_of_week": [last_row["day_of_week"]] * days,
        "hour": [last_row["hour"]] * days,
        "minute": [0] * days,  # Starting at minute 0 for predictions
    }

    # Construct a DataFrame for future predictions
    future_df = pd.DataFrame(future_data)

    # Predict the future prices
    prediction = model.predict(future_df)
    return prediction

# ------------------------------
# 🌟 Build Streamlit App with UI/UX Enhancements
# ------------------------------
def main():
    st.set_page_config(page_title="AI Crypto Price Prediction", layout="wide")
    st.title("🔮 AI Crypto Price Prediction")
    
    # Input for cryptocurrency symbol
    symbol = st.text_input("Enter Crypto Symbol (e.g., BTCUSDT):", "BTCUSDT")
    
    if symbol:
        # Fetch data from Binance API
        df = fetch_crypto_data(symbol)

        # Display data and predictions
        if df is not None:
            # Candlestick Chart
            st.subheader("📊 Candlestick Chart (Last 100 Data Points)")
            fig = go.Figure(data=[go.Candlestick(
                x=df["ds"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name='Candlestick'
            )])
            fig.update_layout(
                title=f"{symbol} - Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price (USDT)",
                template="plotly_dark"
            )
            st.plotly_chart(fig)

            # Display Data
            st.subheader("📝 Last 5 Data Points")
            st.write(df.tail())

            # Add a progress bar to show prediction progress
            with st.spinner('Training the model and predicting future prices...'):
                prediction = train_random_forest(df, days=7)
            
            # Display Prediction
            st.subheader("🔮 7-Day Price Prediction")
            future_dates = pd.date_range(df["ds"].max(), periods=8, freq="D")[1:]

            prediction_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Close Price": prediction
            })

            st.write(prediction_df)

            # Plot the prediction
            st.subheader("📉 Prediction Plot")
            fig_pred = px.line(prediction_df, x='Date', y='Predicted Close Price', title="Predicted Crypto Prices (Next 7 Days)")
            st.plotly_chart(fig_pred)

            # Show prediction results as a bar chart
            st.subheader("📊 Price Prediction Bar Chart")
            fig_bar = px.bar(prediction_df, x='Date', y='Predicted Close Price', title="Predicted Close Price for the Next 7 Days")
            st.plotly_chart(fig_bar)

            # Trigger a rerun (after prediction and chart display)
            st.rerun()  # This is the correct method now!

        else:
            st.error("Failed to fetch data from Binance. Please try again later.")

if __name__ == "__main__":
    main()
