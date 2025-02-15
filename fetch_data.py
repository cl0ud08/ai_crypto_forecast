import pandas as pd
import requests

def fetch_crypto_data(symbol, interval="1m", limit=100):
    """
    Fetch real-time crypto price data from Binance API.
    :param symbol: Cryptocurrency symbol (e.g., "BTCUSDT")
    :param interval: Timeframe for data (e.g., "1m", "5m", "1h")
    :param limit: Number of data points to fetch
    :return: DataFrame with datetime ('ds') and price ('y')
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError if status is not 200
        data = response.json()

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
        df = df[['timestamp', 'close']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'], unit='ms')
        df['y'] = df['y'].astype(float)

        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
