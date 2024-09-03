from alpha_vantage.timeseries import TimeSeries
import pandas as pd

api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Ganti dengan API Key Anda
ts = TimeSeries(key=api_key, output_format='pandas')

def load_data(ticker, start, end):
    try:
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        data = data[start:end]
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

ticker = "BBCA.JK"
start = "2015-01-01"
end = "2024-01-01"
data = load_data(ticker, start, end)
print(data.head())
