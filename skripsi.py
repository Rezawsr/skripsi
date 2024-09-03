import yfinance as yf

ticker = "BBCA.JK"  # Ganti dengan ticker yang sesuai
start = "2015-01-01"
end = "2024-01-01"

data = yf.download(ticker, start=start, end=end)
print(data.head())
