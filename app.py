import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests
import time
from prophet import Prophet
from prophet.plot import plot_plotly
import ta  # Technical Indicators Library

# ------------------------------
# 🎨 UI Customization
# ------------------------------
st.set_page_config(page_title="AI Crypto Forecast", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 AI-Powered Crypto Market Forecast</h1>", unsafe_allow_html=True)

st.sidebar.header("🔍 Market Settings")

# ------------------------------
# 🔥 User Inputs
# ------------------------------
crypto_options = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]
selected_cryptos = st.sidebar.multiselect("Select Cryptos", crypto_options, default=["BTCUSDT"])
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
days_to_predict = st.sidebar.slider("Days to Predict", 1, 30, 7)
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=True)

# ------------------------------
# 🔥 Fetch Real-Time Crypto Data
# ------------------------------
def fetch_crypto_data(symbol, interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades", "taker_buy_base_asset", "taker_buy_quote_asset", "ignore"])
        
        df["ds"] = pd.to_datetime(df["time"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["y"] = df["close"]  # For Prophet
        
        return df[["ds", "open", "high", "low", "close", "y"]]
    else:
        return None

# ------------------------------
# 💰 Fetch Real-Time Price
# ------------------------------
def fetch_live_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return float(response.json()["price"])
    return None

# ------------------------------
# 📊 Add Technical Indicators
# ------------------------------
def add_technical_indicators(df):
    df["SMA"] = ta.trend.sma_indicator(df["close"], window=10)  # Simple Moving Average
    df["EMA"] = ta.trend.ema_indicator(df["close"], window=10)  # Exponential Moving Average
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)  # Relative Strength Index
    df["MACD"] = ta.trend.macd(df["close"])  # MACD
    df["Bollinger_High"] = ta.volatility.bollinger_hband(df["close"])  # Bollinger Bands High
    df["Bollinger_Low"] = ta.volatility.bollinger_lband(df["close"])  # Bollinger Bands Low
    return df

# ------------------------------
# 🤖 Train Prophet Model
# ------------------------------
def train_prophet(df, days=7):
    model = Prophet()
    model.fit(df[["ds", "y"]])
    future = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)
    return forecast, model

# ------------------------------
# 🚀 Real-Time Update Loop
# ------------------------------
placeholder = st.empty()

if st.sidebar.button("Start Analysis"):
    while True:
        with placeholder.container():
            for idx, symbol in enumerate(selected_cryptos):
                df = fetch_crypto_data(symbol, interval=interval, limit=200)
                
                if df is not None and not df.empty:
                    df = add_technical_indicators(df)  # Add indicators
                    
                    # Train Prophet model
                    forecast_prophet, model_prophet = train_prophet(df, days=days_to_predict)

                    # 💰 Fetch Live Price
                    live_price = fetch_live_price(symbol)
                    if live_price:
                        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>💰 {symbol} Live Price: ${live_price:,.2f}</h2>", unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    # 🔵 Candlestick Chart (Fixed Key Issue)
                    with col1:
                        st.subheader(f"📈 {symbol} Real-Time Candlestick Chart ({interval})")
                        fig1 = go.Figure(
                            data=[go.Candlestick(
                                x=df["ds"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], 
                                increasing_line_color="green", decreasing_line_color="red"
                            )]
                        )
                        st.plotly_chart(fig1, use_container_width=True, key=f"candlestick_{symbol}_{idx}_{time.time()}")

                    # 🔮 Prophet Prediction Chart (Fixed Key Issue)
                    with col2:
                        st.subheader(f"🔮 {days_to_predict}-Day AI Prediction for {symbol}")
                        fig2 = plot_plotly(model_prophet, forecast_prophet)
                        st.plotly_chart(fig2, use_container_width=True, key=f"prediction_{symbol}_{idx}_{time.time()}")

                    # 📥 Prepare CSV for download
                    csv = forecast_prophet[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)

                    # 📥 Download Forecast Button (Unique Key)
                    st.download_button(
                        label="📥 Download Forecast",
                        data=csv,
                        file_name=f"{symbol}_forecast.csv",
                        mime="text/csv",
                        key=f"download_{symbol}_{idx}_{time.time()}"  # Unique Key
                    )

        if not auto_refresh:
            break
        time.sleep(5)  # Refresh every 5 seconds
