import yfinance as yf
df = yf.download("XU100.IS", period="1y", progress=False)
print(df.shape)  # (250, 6) gibi bir şey olmalı
print(df.tail())