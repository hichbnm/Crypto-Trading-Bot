import logging
from datetime import datetime, timedelta
from typing import List, Any
import numpy as np
from binance.client import Client
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    TRADING_FEE, TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    TURN_POINT_WINDOW, TURNING_POINT_MARGIN
)
import matplotlib.pyplot as plt
from dateutil import parser as dateparser
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# === USER SETTINGS ===
SYMBOL = 'BTCUSDT'         # Symbol to backtest
TRADE_SIZE = 80         # Amount (USDT) you enter with per trade
INITIAL_BALANCE = 100    # Starting balance in USDT
DAYS = 30


                  # Number of days to backtest
# All other user settings are imported from config.py
# === END USER SETTINGS ===

# Initialize Binance client
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def calculate_rsi(prices: List[float], period: int = RSI_PERIOD) -> float:
    if len(prices) < period + 1:
        return 50.0  # Neutral if not enough data
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

# --- Indicator Calculation Functions (ported from main.py) ---
def calculate_sma(prices: List[float], period: int) -> float:
    if len(prices) < period:
        return float(np.mean(prices))  # Return simple average if not enough data
    return float(np.mean(prices[-period:]))

def calculate_ema(prices: List[float], period: int) -> float:
    if len(prices) < period:
        return float(np.mean(prices))  # Return simple average if not enough data
    ema = np.mean(prices[-period:])
    multiplier = 2 / (period + 1)
    for price in reversed(prices[-period:]):
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_macd(prices: List[float], fast_period: int, slow_period: int, signal_period: int) -> tuple:
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema([macd_line], signal_period)
    return macd_line, signal_line

def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int, d_period: int) -> tuple:
    lowest_low = np.min(lows[-k_period:])
    highest_high = np.max(highs[-k_period:])
    k_line = 100 * ((closes[-1] - lowest_low) / (highest_high - lowest_low)) if highest_high != lowest_low else 0.0
    d_line = calculate_sma([k_line], d_period)
    return k_line, d_line

def calculate_bollinger_bands(prices: List[float], period: int, num_std: float) -> tuple:
    sma = calculate_sma(prices, period)
    std_dev = np.std(prices[-period:]) if len(prices) >= period else np.std(prices)
    upper_band = sma + num_std * std_dev
    lower_band = sma - num_std * std_dev
    return upper_band, sma, lower_band

def calculate_average_volume(volumes: List[float], period: int) -> float:
    return float(np.mean(volumes[-period:])) if len(volumes) >= period else float(np.mean(volumes))

def is_high_turn_point(prices: List[float], window: int) -> bool:
    if len(prices) < window:
        return False
    center = window // 2
    center_price = prices[-center-1]
    for i in range(window):
        if i == center:
            continue
        if center_price <= prices[-window + i]:
            return False
    return True

def is_low_turn_point(prices: List[float], window: int) -> bool:
    if len(prices) < window:
        return False
    center = window // 2
    center_price = prices[-center-1]
    for i in range(window):
        if i == center:
            continue
        if center_price >= prices[-window + i]:
            return False
    return True

def fetch_historical_klines(symbol: str, interval: str, start_str: str, end_str: str) -> List[List[Any]]:
    # Use python-binance's generator to fetch all candles robustly
    all_klines = []
    try:
        for kline in binance_client.get_historical_klines_generator(symbol, interval, start_str, end_str):
            all_klines.append(kline)
    except Exception as e:
        logger.error(f"Error fetching historical klines: {str(e)}")
        return []
    logger.info(f"Fetched {len(all_klines)} historical klines for {symbol}")
    if all_klines:
        logger.info(f"First candle: {datetime.fromtimestamp(all_klines[0][0]/1000)}")
        logger.info(f"Last candle: {datetime.fromtimestamp(all_klines[-1][0]/1000)}")
    return all_klines

def simulate_strategy(klines: List[List[Any]]) -> (List[dict], float):
    trades = []
    balance = INITIAL_BALANCE
    in_position = False
    entry_price = 0.0
    quantity = 0.0
    auto_buy_iteration = 1
    margin = TURNING_POINT_MARGIN
    window = TURN_POINT_WINDOW
    take_profit_pct = TAKE_PROFIT_PERCENTAGE
    trade_size = TRADE_SIZE
    i = window
    while i < len(klines):
        prices = [float(k[4]) for k in klines[:i+1]]
        current_price = prices[-1]
        current_time = datetime.fromtimestamp(klines[i][0] / 1000)
        # --- SELL LOGIC ---
        if in_position:
            profit = (current_price - entry_price) * quantity
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            # High turn point + 3 down candles
            is_high_turn = is_high_turn_point(prices, window)
            three_down = len(prices) >= 3 and prices[-1] < prices[-2] < prices[-3]
            # Margin filter
            if is_high_turn and three_down and profit > margin:
                # SELL
                fee = current_price * quantity * TRADING_FEE
                balance += (current_price * quantity) - fee
                trades.append({
                    'type': 'SELL',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'auto_buy_iteration': auto_buy_iteration
                })
                in_position = False
                auto_buy_iteration += 1
                entry_price = 0.0
                quantity = 0.0
                # After sell, immediately look for next buy (auto-buy loop)
                i += 1
                continue
        # --- BUY LOGIC ---
        else:
            is_low_turn = is_low_turn_point(prices, window)
            three_up = len(prices) >= 3 and prices[-1] > prices[-2] > prices[-3]
            if is_low_turn and three_up:
                # First buy: no margin filter
                if auto_buy_iteration == 1:
                    if balance >= trade_size:
                        entry_price = current_price
                        entry_time = current_time
                        quantity = trade_size / entry_price
                        fee = trade_size * TRADING_FEE
                        balance -= (trade_size + fee)
                        in_position = True
                        trades.append({
                            'type': 'BUY',
                            'entry_price': entry_price,
                            'entry_time': entry_time,
                            'auto_buy_iteration': auto_buy_iteration
                        })
                else:
                    # Subsequent buys: require price bounce > margin/quantity
                    trough_price = prices[-(window // 2) - 1]
                    price_rise = current_price - trough_price
                    required_bounce = margin / (trade_size / current_price)
                    if price_rise > required_bounce and balance >= trade_size:
                        entry_price = current_price
                        entry_time = current_time
                        quantity = trade_size / entry_price
                        fee = trade_size * TRADING_FEE
                        balance -= (trade_size + fee)
                        in_position = True
                        trades.append({
                            'type': 'BUY',
                            'entry_price': entry_price,
                            'entry_time': entry_time,
                            'auto_buy_iteration': auto_buy_iteration
                        })
        i += 1
    # If still in position at the end, sell at last price
    if in_position:
        current_price = float(klines[-1][4])
        current_time = datetime.fromtimestamp(klines[-1][0] / 1000)
        profit = (current_price - entry_price) * quantity
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        fee = current_price * quantity * TRADING_FEE
        balance += (current_price * quantity) - fee
        trades.append({
            'type': 'SELL',
            'entry_price': entry_price,
            'exit_price': current_price,
            'profit': profit,
            'profit_pct': profit_pct,
            'entry_time': entry_time,
            'exit_time': current_time,
            'auto_buy_iteration': auto_buy_iteration
        })
    return trades, balance

def main():
    end_time = datetime.now()
    start_time = end_time - timedelta(days=DAYS)
    start_str = start_time.strftime('%Y-%m-%d')
    end_str = end_time.strftime('%Y-%m-%d')
    klines = fetch_historical_klines(SYMBOL, Client.KLINE_INTERVAL_5MINUTE, start_str, end_str)
    if not klines:
        logger.error("No historical data fetched. Exiting.")
        return
    trades, final_balance = simulate_strategy(klines)
    total_trades = len([t for t in trades if t['type'] == 'SELL'])
    profitable_trades = sum(1 for t in trades if t['type'] == 'SELL' and t['profit'] > 0)
    total_profit = sum(t['profit'] for t in trades if t['type'] == 'SELL')
    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    logger.info("\nBacktest Results:")
    logger.info(f"Initial Balance: {INITIAL_BALANCE:.2f} USDT")
    logger.info(f"Trade Size: {TRADE_SIZE:.2f} USDT")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Profitable Trades: {profitable_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Total Profit: {total_profit:.2f} USDT")
    logger.info(f"Final Balance: {final_balance:.2f} USDT")
    # Plot price and trades
    klines_prices = [float(k[4]) for k in klines]
    klines_times = [datetime.fromtimestamp(k[0] / 1000) for k in klines]
    buy_times = [t['entry_time'] for t in trades if t['type'] == 'BUY']
    buy_prices = [t['entry_price'] for t in trades if t['type'] == 'BUY']
    sell_times = [t['exit_time'] for t in trades if t['type'] == 'SELL']
    sell_prices = [t['exit_price'] for t in trades if t['type'] == 'SELL']
    plt.figure(figsize=(14, 8))
    plt.plot(klines_times, klines_prices, label='Price', color='blue', alpha=0.7)
    plt.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy')
    plt.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell')
    plt.title(f'{SYMBOL} Price with Buy/Sell Points')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# Add this function to monitor and execute pending orders