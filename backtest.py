import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Initialize Binance client
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Constants
SYMBOL = 'BTCUSDT'
RSI_PERIOD = 14
RSI_OVERSOLD = 30
TAKE_PROFIT_PERCENTAGE = 1  # 2% take profit
STOP_LOSS_PERCENTAGE = 10    # 1% stop loss
TRADING_FEE = 0.001           # 0.1% trading fee

def calculate_rsi(prices: List[float], period: int = RSI_PERIOD) -> float:
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

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
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi[-1]

def fetch_historical_klines(symbol: str, interval: str, start_str: str, end_str: str) -> List[List[Any]]:
    try:
        klines = binance_client.get_historical_klines(symbol, interval, start_str, end_str)
        logger.info(f"Fetched {len(klines)} historical klines for {symbol}")
        return klines
    except Exception as e:
        logger.error(f"Error fetching historical klines: {str(e)}")
        return []

def simulate_trades(klines: List[List[Any]]) -> List[Dict[str, Any]]:
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None

    for i in range(RSI_PERIOD, len(klines)):
        prices = [float(k[4]) for k in klines[:i + 1]]
        current_price = prices[-1]
        current_time = datetime.fromtimestamp(klines[i][0] / 1000)
        rsi = calculate_rsi(prices)

        if not in_position and rsi < RSI_OVERSOLD:
            entry_price = current_price
            entry_time = current_time
            in_position = True
            logger.info(f"BUY at {entry_price} (RSI: {rsi:.2f}) at {entry_time}")

        elif in_position:
            profit_percentage = ((current_price - entry_price) / entry_price) * 100
            if profit_percentage >= TAKE_PROFIT_PERCENTAGE:
                exit_price = current_price
                exit_time = current_time
                profit = (exit_price - entry_price) * (1 - TRADING_FEE)
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_percentage': profit_percentage,
                    'profit': profit
                })
                in_position = False
                logger.info(f"SELL at {exit_price} (Profit: {profit_percentage:.2f}%, RSI: {rsi:.2f}) at {exit_time}")

            elif profit_percentage <= -STOP_LOSS_PERCENTAGE:
                exit_price = current_price
                exit_time = current_time
                loss = (entry_price - exit_price) * (1 - TRADING_FEE)
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_percentage': profit_percentage,
                    'profit': -loss
                })
                in_position = False
                logger.info(f"SELL at {exit_price} (Loss: {profit_percentage:.2f}%, RSI: {rsi:.2f}) at {exit_time}")

    return trades

def main():
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    start_str = start_time.strftime('%Y-%m-%d')
    end_str = end_time.strftime('%Y-%m-%d')

    klines = fetch_historical_klines(SYMBOL, Client.KLINE_INTERVAL_1HOUR, start_str, end_str)
    if not klines:
        logger.error("No historical data fetched. Exiting.")
        return

    trades = simulate_trades(klines)

    total_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if trade['profit'] > 0)
    total_profit = sum(trade['profit'] for trade in trades)
    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

    logger.info("\nBacktest Results:")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Profitable Trades: {profitable_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Total Profit: {total_profit:.2f} USDT")

if __name__ == "__main__":
    main()

# Add this function to monitor and execute pending orders