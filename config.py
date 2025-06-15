import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Trading Configuration
TRADING_FEE = 0.001  # 0.1% trading fee
TAKE_PROFIT_PERCENTAGE = 10  # 10% take profit
STOP_LOSS_PERCENTAGE = 5  # 5% stop loss 

# Price Fetching Configuration
PRICE_FETCH_DELAY = 0.2  # Delay between price fetches in seconds 