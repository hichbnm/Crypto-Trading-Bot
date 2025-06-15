import pandas as pd
import ta
import datetime
from binance.client import Client
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
SYMBOL = "BTCUSDT"
RSI_WINDOW = 14
CANDLE_INTERVAL = Client.KLINE_INTERVAL_1HOUR  # Weekly candles
LIMIT = 200  # Number of weeks to fetch (increased for more data)

# === INIT ===
client = Client(API_KEY, API_SECRET)

# === Fetch Weekly OHLCV with error handling ===
def get_ohlcv(symbol, interval, limit):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return pd.DataFrame()  # return empty dataframe on error

    df = pd.DataFrame(klines, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['time', 'close']]

df = get_ohlcv(SYMBOL, CANDLE_INTERVAL, LIMIT)
if df.empty:
    print("No data fetched, exiting.")
    exit()

# === Calculate RSI ===
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=RSI_WINDOW).rsi()

# Print RSI stats to understand range
print(f"RSI min: {df['rsi'].min():.2f}, RSI max: {df['rsi'].max():.2f}")

# Print recent RSI values where RSI < 30 (oversold points)
oversold = df[df['rsi'] < 30]
print(f"\nNumber of oversold RSI points (<30): {len(oversold)}")
print(oversold[['time', 'rsi', 'close']].tail(10))

# === Detect RSI Bounce (simpler: RSI crosses above 30 from below) ===
df['rsi_bounce'] = (
    (df['rsi'].shift(1) < 30) &  # previous candle RSI below 30
    (df['rsi'] >= 30)            # current candle RSI at or above 30
)

# === Log Bounce Points ===
bounces = df[df['rsi_bounce']]
print(f"\n📊 Detected {len(bounces)} RSI Bounce(s) on 1W candles:")
for index, row in bounces.iterrows():
    print(f"🟢 Bounce at {row['time'].date()} | RSI: {row['rsi']:.2f} | Price: {row['close']:.2f}")

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['rsi'], label='RSI (1W)', color='blue')
plt.axhline(30, color='red', linestyle='--', label='Oversold (30)')
plt.scatter(bounces['time'], bounces['rsi'], color='green', label='RSI Bounce', zorder=5)
plt.title('RSI Bounce Detection - Weekly')
plt.xlabel('Week')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
