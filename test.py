import pandas as pd
import ta
import datetime
from binance.client import Client
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# === CONFIG ===
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
SYMBOL = "BTCUSDT"
RSI_WINDOW = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
TAKE_PROFIT_PERCENTAGE = 6
STOP_LOSS_PERCENTAGE = -10
TRADING_FEE = 0.001  # 0.1% trading fee
CANDLE_INTERVAL = Client.KLINE_INTERVAL_1HOUR
LIMIT = 1000  # Increased to 1000 candles for longer period
INITIAL_INVESTMENT = 10000  # Initial investment in USDT

# === INIT ===
client = Client(API_KEY, API_SECRET)

def get_ohlcv(symbol, interval, limit):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

def backtest_strategy(df):
    trades = []
    portfolio_values = []
    in_position = False
    entry_price = 0
    entry_time = None
    current_portfolio = INITIAL_INVESTMENT
    current_btc = 0
    
    for i in range(RSI_WINDOW, len(df)):
        current_price = df['close'].iloc[i]
        current_time = df['time'].iloc[i]
        current_rsi = df['rsi'].iloc[i]
        
        if not in_position:
            # Buy signal: RSI crosses below oversold
            if df['rsi'].iloc[i-1] >= RSI_OVERSOLD and current_rsi < RSI_OVERSOLD:
                in_position = True
                entry_price = current_price
                entry_time = current_time
                # Calculate BTC amount to buy
                current_btc = (current_portfolio * (1 - TRADING_FEE)) / current_price
                current_portfolio = 0  # All USDT converted to BTC
                trades.append({
                    'type': 'BUY',
                    'time': current_time,
                    'price': current_price,
                    'rsi': current_rsi,
                    'btc_amount': current_btc,
                    'portfolio_value': current_portfolio + (current_btc * current_price)
                })
        else:
            # Sell signals
            current_portfolio_value = current_btc * current_price
            roi = ((current_price - entry_price) / entry_price) * 100
            
            if (current_rsi > RSI_OVERBOUGHT or  # RSI overbought
                roi >= TAKE_PROFIT_PERCENTAGE or  # Take profit
                (roi <= STOP_LOSS_PERCENTAGE and current_rsi >= RSI_OVERSOLD)):  # Stop loss only if not oversold
                
                in_position = False
                # Convert BTC back to USDT
                current_portfolio = current_portfolio_value * (1 - TRADING_FEE)
                current_btc = 0
                trades.append({
                    'type': 'SELL',
                    'time': current_time,
                    'price': current_price,
                    'rsi': current_rsi,
                    'roi': roi,
                    'portfolio_value': current_portfolio
                })
        
        # Record portfolio value at each step
        portfolio_values.append({
            'time': current_time,
            'value': current_portfolio + (current_btc * current_price)
        })
    
    return trades, portfolio_values

# === Main Execution ===
print(f"Fetching {LIMIT} candles for {SYMBOL}...")
df = get_ohlcv(SYMBOL, CANDLE_INTERVAL, LIMIT)
if df.empty:
    print("No data fetched, exiting.")
    exit()

# Calculate RSI
df['rsi'] = calculate_rsi(df['close'].values, RSI_WINDOW)

# Run backtest
trades, portfolio_values = backtest_strategy(df)

# === Analysis ===
print("\n=== Backtest Results ===")
print(f"Initial Investment: {INITIAL_INVESTMENT} USDT")
print(f"Period: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
print(f"Total Trades: {len(trades) // 2}")  # Divide by 2 because each trade has a buy and sell

# Calculate statistics
profitable_trades = 0
total_profit = 0
max_profit = 0
max_loss = 0
final_portfolio = portfolio_values[-1]['value']
total_return = ((final_portfolio - INITIAL_INVESTMENT) / INITIAL_INVESTMENT) * 100

for i in range(0, len(trades), 2):
    if i + 1 < len(trades):  # Ensure we have a complete trade
        buy = trades[i]
        sell = trades[i + 1]
        profit = sell['roi']
        total_profit += profit
        
        if profit > 0:
            profitable_trades += 1
            max_profit = max(max_profit, profit)
        else:
            max_loss = min(max_loss, profit)

print(f"Final Portfolio Value: {final_portfolio:.2f} USDT")
print(f"Total Return: {total_return:.2f}%")
print(f"Profitable Trades: {profitable_trades}")
print(f"Win Rate: {(profitable_trades / (len(trades) // 2)) * 100:.2f}%")
print(f"Max Profit: {max_profit:.2f}%")
print(f"Max Loss: {max_loss:.2f}%")

# === Plotting ===
plt.figure(figsize=(15, 15))

# Plot 1: Price and Trades
plt.subplot(3, 1, 1)
plt.plot(df['time'], df['close'], label='Price', color='blue', alpha=0.5)
for trade in trades:
    if trade['type'] == 'BUY':
        plt.scatter(trade['time'], trade['price'], color='green', marker='^', s=100)
    else:
        plt.scatter(trade['time'], trade['price'], color='red', marker='v', s=100)
plt.title(f'{SYMBOL} Price and Trades')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Plot 2: RSI
plt.subplot(3, 1, 2)
plt.plot(df['time'], df['rsi'], label='RSI', color='purple')
plt.axhline(RSI_OVERBOUGHT, color='red', linestyle='--', label='Overbought')
plt.axhline(RSI_OVERSOLD, color='green', linestyle='--', label='Oversold')
for trade in trades:
    if trade['type'] == 'BUY':
        plt.scatter(trade['time'], trade['rsi'], color='green', marker='^', s=100)
    else:
        plt.scatter(trade['time'], trade['rsi'], color='red', marker='v', s=100)
plt.title('RSI with Trade Signals')
plt.xlabel('Time')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)

# Plot 3: Portfolio Value
plt.subplot(3, 1, 3)
portfolio_df = pd.DataFrame(portfolio_values)
plt.plot(portfolio_df['time'], portfolio_df['value'], label='Portfolio Value', color='green')
plt.axhline(INITIAL_INVESTMENT, color='red', linestyle='--', label='Initial Investment')
plt.title('Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value (USDT)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print detailed trade log
print("\n=== Detailed Trade Log ===")
for i in range(0, len(trades), 2):
    if i + 1 < len(trades):
        buy = trades[i]
        sell = trades[i + 1]
        print(f"\nTrade {i//2 + 1}:")
        print(f"Entry: {buy['time']} | Price: {buy['price']:.2f} | RSI: {buy['rsi']:.2f}")
        print(f"BTC Amount: {buy['btc_amount']:.8f}")
        print(f"Exit:  {sell['time']} | Price: {sell['price']:.2f} | RSI: {sell['rsi']:.2f}")
        print(f"ROI: {sell['roi']:.2f}%")
        print(f"Portfolio Value: {sell['portfolio_value']:.2f} USDT")
