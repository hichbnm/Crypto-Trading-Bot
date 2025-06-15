# RSI Trading Bot

## Project Structure
```
production/
├── main.py           # Main trading bot application
├── config.py         # Configuration and API keys
├── test.py          # Backtesting script
├── trades.json      # Trade history storage
├── requirements.txt # Python dependencies
├── templates/       # HTML templates
└── static/         # Static files (CSS, JS)
```

## Installation

1. Install Python 3.7 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your Binance API keys in `config.py`:
```python
API_KEY = "your_binance_api_key"
API_SECRET = "your_binance_api_secret"
```

## Running the Bot

1. Start the server:
```bash
uvicorn main:app --reload
```
2. Open your browser and go to `http://localhost:8000`
3. Log in with your credentials

## Trading Features

### Auto Buy
1. Click "Auto Buy" button
2. Enter amount in USDT
3. The bot will:
   - Check your USDT balance
   - Calculate RSI
   - Create a pending order
   - Execute when RSI < 30

### Manual Trading
1. Click "New Trade"
2. Fill in:
   - Coin (e.g., BTC, ETH)
   - Amount in USDT
   - Order type (market/limit)
   - Take profit (default 4%)
   - Stop loss (default -20%)

## Strategy Details

### Entry Conditions
- RSI < 30 (oversold)
- 1-hour timeframe
- RSI period: 14

### Exit Conditions
1. Take Profit: 4%
2. Stop Loss: -20%
3. RSI > 70 (overbought)

## Monitoring Trades
- View active trades on the dashboard
- See real-time price updates
- Monitor RSI values
- Track profit/loss

## Backtesting
Run the backtest script to test the strategy:
```bash
python test.py
```
This will:
- Test the strategy on historical data
- Show performance metrics
- Display charts

## Important Notes

### Risk Management
- Start with small amounts
- Monitor trades regularly
- Don't invest more than you can afford to lose

### Best Practices
1. Keep API keys secure
2. Monitor trade history
3. Check bot performance
4. Adjust parameters if needed

### Common Issues
1. **API Connection**
   - Check internet connection
   - Verify API keys
   - Check Binance API status

2. **Order Execution**
   - Ensure sufficient USDT balance
   - Check minimum order amounts
   - Verify sufficient funds

## Security
- API keys stored in config.py
- Session-based authentication
- Secure WebSocket connections

## Maintenance
1. Regular updates
2. Monitor error logs
3. Check Binance API status
4. Backup trade history

## Support
If you have issues:
1. Check error logs
2. Verify API keys
3. Check balance
4. Check Binance API status

## Disclaimer
This bot is for educational purposes. Cryptocurrency trading involves risk. Always do your own research.

## Requirements
```
fastapi
uvicorn
python-binance
python-dotenv
numpy
pandas
matplotlib
```

## License
This project is licensed under the MIT License - see the LICENSE file for details. 