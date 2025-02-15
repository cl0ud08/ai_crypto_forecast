import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests
import time
from prophet import Prophet
from prophet.plot import plot_plotly
import ta  # Technical Indicators Library
import requests
from textblob import TextBlob
# ------------------------------
# 🎨 UI Customization
# ------------------------------
st.set_page_config(page_title="AI Crypto Forecast", layout="wide")
st.markdown("""
    <style>
    .news-container {
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .news-title {
        font-size: 20px;
        font-weight: bold;
        text-decoration: none;
    }
    .news-source {
        font-size: 14px;
        color: gray;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>AI-Powered Crypto Market Forecast</h1>", unsafe_allow_html=True)

# ------------------------------
# 🛠 Initialize Session State
# ------------------------------
if "news_index" not in st.session_state:
    st.session_state.news_index = 0  # Start with the first news item
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()  # Store last update time

# ------------------------------
# 📰 Fetch Latest Crypto News
# ------------------------------
def fetch_crypto_news():
    api_url = "https://cryptopanic.com/api/v1/posts/?auth_token=c540fcf87be58748c11332a92cc7806c1ab4b12a&public=true"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception:
        return []

# ------------------------------
# 🔍 News Sentiment Analysis & Styling
# ------------------------------
def analyze_sentiment(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0.1:
        return "🙂 Positive", "green"
    elif sentiment_score < -0.1:
        return "😟 Negative", "red"
    else:
        return "😐 Neutral", "black"

# ------------------------------
# 📰 Display News Section (Auto-Switch Every 5 Seconds)
# ------------------------------
st.markdown("<h2 style='text-align: center;'>📰 Latest Crypto News</h2>", unsafe_allow_html=True)

news_items = fetch_crypto_news()

if news_items:
    current_time = time.time()
    
    # Check if 5 seconds have passed since last update
    if current_time - st.session_state.last_update >= 0:
        st.session_state.news_index = (st.session_state.news_index + 1) % len(news_items)
        st.session_state.last_update = current_time  # Update the timestamp

    # Get current news item
    news = news_items[st.session_state.news_index]
    title = news["title"]
    url = news["url"]
    source = news["source"]["title"]

    # 🔍 Get Sentiment & Color
    sentiment, color = analyze_sentiment(title)

    # 🎨 Dynamic Styling
    with st.container():
        st.markdown(f'<div class="news-container" style="border-left: 8px solid {color};">', unsafe_allow_html=True)
        st.markdown(f"<p class='news-title' style='color:{color};'>🔹 <a href='{url}' target='_blank' style='text-decoration: none; color: {color};'>{title}</a></p>", unsafe_allow_html=True)
        st.markdown(f"<p class='news-source'>🔸 *Source: {source}*</p>", unsafe_allow_html=True)
        st.markdown(f"📊 **Sentiment Analysis:** {sentiment}")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("No news available at the moment.")

# ------------------------------
# 🎯 Other UI Components (Rest of Your Page)
# ------------------------------
st.sidebar.header("Market Settings")

crypto_options = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]
selected_cryptos = st.sidebar.multiselect("Select Cryptos", crypto_options, default=["BTCUSDT"])
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
days_to_predict = st.sidebar.slider("Days to Predict", 1, 30, 7)
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=True)
import streamlit as st



# Sidebar button with unique key
if st.sidebar.button("Start News Analysis", key="start_analysis_sidebar"):
    placeholder = st.empty()  # Create a placeholder for dynamic content
    while True:
        with placeholder.container():
            st.write("📈 **Running market analysis...**")  
            # Add market analysis logic here

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
    df["SMA"] = ta.trend.sma_indicator(df["close"], window=10)
    df["EMA"] = ta.trend.ema_indicator(df["close"], window=10)
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    df["MACD"] = ta.trend.macd(df["close"])
    df["Bollinger_High"] = ta.volatility.bollinger_hband(df["close"])
    df["Bollinger_Low"] = ta.volatility.bollinger_lband(df["close"])
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
                        st.markdown(f"<h2 style='text-align: center; color: black;'>💰 {symbol} Live Price: ${live_price:,.2f}</h2>", unsafe_allow_html=True)

                    col1, col2 = st.columns([3, 2])  # Adjusted layout

                    # 🔵 Candlestick Chart
                    with col1:
                        st.subheader(f"📈 {symbol} Real-Time Candlestick Chart ({interval})")
                        fig1 = go.Figure(
                            data=[go.Candlestick(
                                x=df["ds"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], 
                                increasing_line_color="green", decreasing_line_color="red"
                            )]
                        )
                        st.plotly_chart(fig1, use_container_width=True, key=f"candlestick_{symbol}_{idx}_{time.time()}")

                    # 🔮 Prophet Prediction Chart
                    with col2:
                        st.subheader(f"🔮 {days_to_predict}-Day AI Prediction for {symbol}")
                        fig2 = plot_plotly(model_prophet, forecast_prophet)
                        st.plotly_chart(fig2, use_container_width=True, key=f"prediction_{symbol}_{idx}_{time.time()}")

                    # 📥 Download Forecast Button
                    csv = forecast_prophet[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast",
                        data=csv,
                        file_name=f"{symbol}_forecast.csv",
                        mime="text/csv",
                        key=f"download_{symbol}_{idx}_{time.time()}"
                    )

        if not auto_refresh:
            break
        time.sleep(1)  # Refresh every 5 seconds
