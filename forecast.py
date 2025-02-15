from prophet import Prophet
from fetch_data import fetch_crypto_data

def train_prophet(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast, model  # ✅ Return both forecast and model

if __name__ == "__main__":
    symbol = "BTCUSDT"  # ✅ Define a symbol
    df = fetch_crypto_data(symbol)  # ✅ Pass the symbol argument

    if df is not None and not df.empty:
        forecast, model = train_prophet(df)
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    else:
        print("Failed to fetch data.")
