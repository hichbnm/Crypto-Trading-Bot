# Crypto Trading Bot Documentation

## Overview
This is an automated cryptocurrency trading bot that supports both manual and automated trading strategies. The bot integrates with Binance and provides features like take profit, stop loss, and RSI-based auto-trading.

## Prerequisites
- Python 3.8 or higher
- Binance account with API keys
- Sufficient USDT balance for trading

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your Binance API credentials:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

4. Start the bot:
```bash
python main.py
```

## Features

### 1. Manual Trading
- **Market Orders**: Execute trades at current market price
- **Limit Orders**: Set specific entry prices
- **Take Profit**: Set profit targets in percentage or dollar amount
- **Stop Loss**: Optional stop loss protection

### 2. Auto Trading
- **RSI Strategy**: Automatically enters trades when RSI is oversold (< 30)
- **Auto Take Profit**: Closes trades when profit target is reached
- **Auto Stop Loss**: Optional protection against losses

### 3. Trade Management
- Real-time price monitoring
- Profit/Loss tracking
- Trade history
- Active trade management

## How to Use

### Manual Trading

1. **Place a Trade**:
   - Select the cryptocurrency (e.g., Bitcoin, Ethereum)
   - Enter the amount in USDT
   - Choose order type (Market or Limit)
   - Set take profit (percentage or dollar amount)
   - Optionally enable and set stop loss
   - Click "Place Order"

2. **Take Profit Options**:
   - **Percentage**: Enter a percentage (e.g., 10%)
   - **Dollar Amount**: Enter a dollar value (e.g., $100)
   - Default is 10% if not specified

3. **Stop Loss**:
   - Check "Enable" to activate stop loss
   - Enter stop loss percentage
   - Default is 5% if not specified

### Auto Trading

1. **Start Auto Trading**:
   - Select the cryptocurrency
   - Enter the amount in USDT
   - Set take profit and stop loss (optional)
   - Click "Auto Buy"

2. **How Auto Trading Works**:
   - Bot monitors RSI (Relative Strength Index)
   - Enters trade when RSI < 30 (oversold condition)
   - Automatically closes when:
     - Take profit is reached
     - Stop loss is triggered
     - RSI becomes overbought (> 70)

### Managing Trades

1. **Active Trades**:
   - View all open trades in the "Active Trades" table
   - Monitor current prices and profit/loss
   - Close trades manually if needed

2. **Trade History**:
   - View all closed trades
   - Track performance and profitability
   - Analyze trading patterns

## Important Notes

1. **Risk Management**:
   - Always use stop loss for protection
   - Start with small amounts
   - Monitor trades regularly

2. **Trading Fees**:
   - Binance charges 0.1% per trade
   - Fees are automatically calculated and displayed

3. **Best Practices**:
   - Test with small amounts first
   - Monitor the bot's performance
   - Keep your API keys secure
   - Don't share your credentials

## Troubleshooting

1. **Common Issues**:
   - "Insufficient Balance": Ensure you have enough USDT
   - "Invalid API Key": Check your Binance API credentials
   - "Connection Error": Check your internet connection

2. **Error Messages**:
   - All errors are displayed in the interface
   - Check the console for detailed logs

## Security

1. **API Security**:
   - Never share your API keys
   - Use API keys with trading permissions only
   - Enable IP restrictions in Binance

2. **Account Security**:
   - Use strong passwords
   - Enable 2FA on your Binance account
   - Regularly monitor your trades

## Support

For issues or questions:
1. Check the console logs
2. Review the error messages
3. Contact support with:
   - Error message
   - Steps to reproduce
   - Screenshots if applicable

## Updates

The bot is regularly updated with:
- New features
- Bug fixes
- Performance improvements
- Security enhancements

Check the repository for the latest updates and improvements. 