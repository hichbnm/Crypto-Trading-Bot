import logging
from datetime import datetime, timedelta
from typing import List, Any
import numpy as np
from binance.client import Client
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    TRADING_FEE, TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT
)
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# === USER SETTINGS ===
SYMBOL = 'BTCUSDT'         # Symbol to backtest
TRADE_SIZE = 1000          # Amount (USDT) you enter with per trade
INITIAL_BALANCE = 1000     # Starting balance in USDT
DAYS = 30                  # Number of days to backtest
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
    try:
        klines = binance_client.get_historical_klines(symbol, interval, start_str, end_str)
        logger.info(f"Fetched {len(klines)} historical klines for {symbol}")
        return klines
    except Exception as e:
        logger.error(f"Error fetching historical klines: {str(e)}")
        return []

def simulate_trades(klines: List[List[Any]]) -> (List[dict], float):
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    quantity = 0.0
    balance = INITIAL_BALANCE

    # Indicator parameters (can be moved to config)
    SMA_PERIOD = 20
    EMA_SHORT_PERIOD = 9
    EMA_LONG_PERIOD = 20
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3
    BOLLINGER_PERIOD = 20
    BOLLINGER_NUM_STD = 2.0
    VOLUME_AVG_PERIOD = 20
    VOLUME_CONFIRMATION_MULTIPLIER = 1.5
    TURN_POINT_WINDOW = 7

    for i in range(RSI_PERIOD, len(klines)):
        prices = [float(k[4]) for k in klines[:i + 1]]
        highs = [float(k[2]) for k in klines[:i + 1]]
        lows = [float(k[3]) for k in klines[:i + 1]]
        volumes = [float(k[5]) for k in klines[:i + 1]]
        current_price = prices[-1]
        current_time = datetime.fromtimestamp(klines[i][0] / 1000)
        rsi = calculate_rsi(prices)
        sma = calculate_sma(prices, SMA_PERIOD)
        ema_short = calculate_ema(prices, EMA_SHORT_PERIOD)
        ema_long = calculate_ema(prices, EMA_LONG_PERIOD)
        macd_line, macd_signal = calculate_macd(prices, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD)
        stoch_k, stoch_d = calculate_stochastic(highs, lows, prices, STOCH_K_PERIOD, STOCH_D_PERIOD)
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(prices, BOLLINGER_PERIOD, BOLLINGER_NUM_STD)
        avg_vol = calculate_average_volume(volumes, VOLUME_AVG_PERIOD)
        vol_confirm = volumes[-1] > VOLUME_CONFIRMATION_MULTIPLIER * avg_vol if avg_vol > 0 else False
        is_high_turn = is_high_turn_point(prices, TURN_POINT_WINDOW)
        is_low_turn = is_low_turn_point(prices, TURN_POINT_WINDOW)

        # Buy condition (keep simple for now, can enhance later)
        if not in_position and rsi < RSI_OVERSOLD and balance >= TRADE_SIZE:
            entry_price = current_price
            entry_time = current_time
            quantity = TRADE_SIZE / entry_price
            fee = TRADE_SIZE * TRADING_FEE
            balance -= (TRADE_SIZE + fee)
            in_position = True
            # Record the buy trade for plotting
            trades.append({
                'type': 'BUY',
                'entry_time': entry_time,
                'entry_price': entry_price
            })
            logger.info(f"BUY  | Price: {entry_price:.2f} | RSI: {rsi:.2f} | Time: {entry_time} | Trade Size: {TRADE_SIZE} | Balance: {balance:.2f}")

        # Enhanced Sell Logic: Only sell if profit is positive (except stop loss)
        elif in_position:
            profit_percentage = ((current_price - entry_price) / entry_price) * 100
            stop_loss_hit = profit_percentage <= -STOP_LOSS_PERCENTAGE
            should_sell = False
            sell_reason = ""
            signal_strength = 0
            # Always allow stop loss
            if stop_loss_hit:
                should_sell = True
                sell_reason = "Stop Loss"
            # All other sells require strictly positive profit
            elif profit_percentage > 0:
                if profit_percentage >= TAKE_PROFIT_PERCENTAGE:
                    should_sell = True
                    sell_reason = "Take Profit"
                elif is_high_turn and rsi > 70 and vol_confirm and ema_short < ema_long:
                    should_sell = True
                    sell_reason = "Perfect Exit (5/5): High turn, RSI>70, High Vol, Downtrend"
                    signal_strength = 5
                elif is_high_turn and rsi > 60 and vol_confirm:
                    should_sell = True
                    sell_reason = "Strong Exit (4/5): High turn, RSI>60, High Vol"
                    signal_strength = 4
                elif rsi > 70 and vol_confirm:
                    should_sell = True
                    sell_reason = "Good Exit (3/5): RSI>70, High Vol"
                    signal_strength = 3
                elif is_high_turn and rsi > 60:
                    should_sell = True
                    sell_reason = "Basic Exit (2/5): High turn, RSI>60"
                    signal_strength = 2
                elif rsi > 70:
                    should_sell = True
                    sell_reason = "Simple Exit (1/5): RSI>70"
                    signal_strength = 1
                elif macd_line < macd_signal:
                    should_sell = True
                    sell_reason = "MACD Bearish Cross"
                elif stoch_k < stoch_d and stoch_k > 80 and stoch_d > 80:
                    should_sell = True
                    sell_reason = "Stochastic Bearish Cross (Overbought)"
                elif current_price >= bb_upper:
                    should_sell = True
                    sell_reason = "Price above upper Bollinger Band"

            if should_sell:
                exit_price = current_price
                exit_time = current_time
                gross = quantity * exit_price
                fee = gross * TRADING_FEE
                net = gross - fee
                profit = net - (quantity * entry_price)
                balance += net
                trades.append({
                    'type': 'SELL',
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_percentage': profit_percentage,
                    'profit': profit,
                    'trade_size': TRADE_SIZE,
                    'sell_reason': sell_reason,
                    'signal_strength': signal_strength
                })
                in_position = False
                logger.info(f"SELL | Price: {exit_price:.2f} | Profit: {profit_percentage:.2f}% | RSI: {rsi:.2f} | Time: {exit_time} | Reason: {sell_reason} | Balance: {balance:.2f}")

    # If still in position at the end, sell at last price
    if in_position:
        exit_price = float(klines[-1][4])
        exit_time = datetime.fromtimestamp(klines[-1][0] / 1000)
        gross = quantity * exit_price
        fee = gross * TRADING_FEE
        net = gross - fee
        profit = net - (quantity * entry_price)
        profit_percentage = ((exit_price - entry_price) / entry_price) * 100
        balance += net
        trades.append({
            'type': 'SELL',
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_percentage': profit_percentage,
            'profit': profit,
            'trade_size': TRADE_SIZE,
            'sell_reason': 'Final Sell',
            'signal_strength': 0
        })
        logger.info(f"FINAL SELL | Price: {exit_price:.2f} | Time: {exit_time} | Balance: {balance:.2f}")

    return trades, balance

def main():
    end_time = datetime.now()
    start_time = end_time - timedelta(days=DAYS)
    start_str = start_time.strftime('%Y-%m-%d')
    end_str = end_time.strftime('%Y-%m-%d')

    klines = fetch_historical_klines(SYMBOL, Client.KLINE_INTERVAL_1HOUR, start_str, end_str)
    if not klines:
        logger.error("No historical data fetched. Exiting.")
        return

    trades, final_balance = simulate_trades(klines)

    total_trades = len(trades)
    profitable_trades = 0
    total_profit = 0
    max_profit = 0
    max_loss = 0
    final_portfolio = final_balance
    total_return = ((final_portfolio - INITIAL_BALANCE) / INITIAL_BALANCE) * 100

    for trade in trades:
        if trade.get('type') == 'SELL':
            profit = trade['profit']
            total_profit += profit
            if profit > 0:
                profitable_trades += 1
                max_profit = max(max_profit, profit)
            else:
                max_loss = min(max_loss, profit)

    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

    logger.info("\nBacktest Results:")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Profitable Trades: {profitable_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Total Profit: {total_profit:.2f} USDT")
    logger.info(f"Final Balance: {final_balance:.2f} USDT")

    # === Plotting ===
    # Fetch price and time for plotting
    klines_prices = [float(k[4]) for k in klines]
    klines_times = [datetime.fromtimestamp(k[0] / 1000) for k in klines]

    # Prepare buy/sell points
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    for trade in trades:
        if trade['type'] == 'BUY':
            buy_times.append(trade['entry_time'])
            buy_prices.append(trade['entry_price'])
        elif trade['type'] == 'SELL':
            sell_times.append(trade['exit_time'])
            sell_prices.append(trade['exit_price'])

    # Plot price and trades
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

    # Plot portfolio value over time
    # (Optional: can be added if you track portfolio value at each step)

if __name__ == "__main__":
    main()

# Add this function to monitor and execute pending orders