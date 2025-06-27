import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Trading Configuration
TRADING_FEE = 0.001  # 0.1% trading fee
TAKE_PROFIT_PERCENTAGE = 4  # 4% take profit (match backtest.py)
STOP_LOSS_PERCENTAGE = 10  # 10% stop loss (match backtest.py)
RSI_PERIOD = 16  # RSI lookback period (match backtest.py)
RSI_OVERSOLD = 25  # RSI buy threshold (match backtest.py)
RSI_OVERBOUGHT = 70  # RSI sell threshold (match backtest.py)

# Price Fetching Configuration
PRICE_FETCH_DELAY = 0.2  # Delay between price fetches in seconds 

# Indicator Periods and Thresholds
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
RSI_CONFIRMATION_LEVEL = 35  # For buy confirmation after oversold
RSI_OVERBOUGHT_CONFIRMATION = 65  # For sell confirmation after overbought 