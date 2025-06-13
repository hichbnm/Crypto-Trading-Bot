from fastapi import FastAPI, Form, HTTPException, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
from typing import Optional, List, Dict, Set, Tuple
import asyncio
import logging
from datetime import datetime
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
import json
import uuid
import time
import httpx
from collections import deque
from fastapi import WebSocketDisconnect
from pydantic import BaseModel
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active trades and trade history
active_trades = []
trade_history = []
connected_clients: Set[WebSocket] = set()

# Price cache with timestamp
price_cache = {}
CACHE_DURATION = 60  # Cache duration in seconds
RATE_LIMIT_DELAY = 1.5  # Delay between requests in seconds
last_request_time = 0

# Price history for technical analysis
PRICE_HISTORY_LENGTH = 100
price_history = {}

# Coin mapping for CoinGecko API
COIN_MAPPING = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "binancecoin": "binancecoin",
    "ripple": "ripple",
    "dogecoin": "dogecoin",
    "polygon": "polygon",
    "chainlink": "chainlink"
}

# File paths for persistent storage
TRADES_FILE = "trades.json"

def load_trades():
    """Load trades from the JSON file."""
    try:
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r') as f:
                data = json.load(f)
                return data.get('active_trades', []), data.get('trade_history', [])
    except Exception as e:
        logger.error(f"Error loading trades: {str(e)}")
    return [], []

def save_trades():
    """Save trades to the JSON file."""
    try:
        data = {
            'active_trades': active_trades,
            'trade_history': trade_history
        }
        with open(TRADES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving trades: {str(e)}")

# Initialize trades from file
active_trades, trade_history = load_trades()

# Technical Analysis Functions
def calculate_sma(prices: List[float], period: int) -> float:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return None
        
    try:
        multiplier = 2 / (period + 1)
        ema = prices[0]  # Start with the first price
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
            
        return ema
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        return None

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: List[float]) -> Tuple[float, float]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < 26:
        return 0.0, 0.0  # Return default values if not enough data
        
    # Calculate EMAs
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    
    if ema12 is None or ema26 is None:
        return 0.0, 0.0  # Return default values if EMA calculation fails
    
    # Calculate MACD line
    macd_line = ema12 - ema26
    
    # Calculate signal line (9-day EMA of MACD)
    signal_line = calculate_ema([macd_line], 9)
    if signal_line is None:
        signal_line = macd_line  # Use MACD line as signal if calculation fails
    
    return macd_line, signal_line

def analyze_trading_signals(coin: str) -> Dict:
    """Analyze trading signals based on technical indicators."""
    if coin not in price_history or len(price_history[coin]) < 26:
        return None
    
    prices = price_history[coin]
    
    # Calculate indicators
    sma20 = calculate_sma(prices, 20)
    sma50 = calculate_sma(prices, 50)
    rsi = calculate_rsi(prices)
    macd = calculate_macd(prices)
    
    if not all([sma20, sma50, rsi, macd]):
        return None
    
    # Generate trading signals
    signals = {
        "trend": "bullish" if sma20 > sma50 else "bearish",
        "rsi_signal": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
        "macd_signal": "buy" if macd[0] > macd[1] and macd[0] > 0 else "sell",
        "strength": 0
    }
    
    # Calculate signal strength (-100 to 100)
    strength = 0
    
    # Trend contribution
    strength += 30 if signals["trend"] == "bullish" else -30
    
    # RSI contribution
    if signals["rsi_signal"] == "oversold":
        strength += 30
    elif signals["rsi_signal"] == "overbought":
        strength -= 30
    
    # MACD contribution
    if signals["macd_signal"] == "buy":
        strength += 40
    elif signals["macd_signal"] == "sell":
        strength -= 40
    
    signals["strength"] = strength
    return signals

async def fetch_price(coin: str) -> Optional[float]:
    """Fetch current price from CoinGecko API with rate limiting and caching."""
    global last_request_time
    
    # Check cache first
    current_time = time.time()
    if coin in price_cache:
        cache_time, cached_price = price_cache[coin]
        if current_time - cache_time < CACHE_DURATION:
            return cached_price

    # Rate limiting
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < RATE_LIMIT_DELAY:
        await asyncio.sleep(RATE_LIMIT_DELAY - time_since_last_request)

    try:
        # Get CoinGecko ID - convert to lowercase for consistency
        coin_id = COIN_MAPPING.get(coin.lower())
        if not coin_id:
            logger.error(f"Unsupported coin symbol: {coin}")
            return None

        # Add timestamp to prevent caching
        timestamp = int(time.time() * 1000)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&_={timestamp}"
        
        # Add proper headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if coin_id in data and "usd" in data[coin_id]:
                price = data[coin_id]["usd"]
                # Update cache
                price_cache[coin] = (current_time, price)
                last_request_time = time.time()
                
                # Update price history
                if coin not in price_history:
                    price_history[coin] = deque(maxlen=PRICE_HISTORY_LENGTH)
                price_history[coin].append(price)
                
                return price
            else:
                logger.error(f"Price data not found for {coin}")
                return None

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning(f"Rate limit hit for {coin}, waiting before retry...")
            await asyncio.sleep(5)  # Wait longer on rate limit
            return await fetch_price(coin)  # Retry once
        logger.error(f"HTTP error while fetching price for {coin}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching price for {coin}: {str(e)}")
        return None

# Enhanced Turn Point Detection Logic
async def detect_turn_points(prices: List[float], window_size: int = 3) -> List[Dict]:
    try:
        turn_points = []
        for i in range(window_size, len(prices) - window_size):
            # Check for high turn point
            if all(prices[i] > prices[i-j] for j in range(1, window_size+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, window_size+1)):
                turn_points.append({
                    "type": "high",
                    "price": prices[i],
                    "index": i,
                    "timestamp": datetime.now()
                })
            # Check for low turn point
            elif all(prices[i] < prices[i-j] for j in range(1, window_size+1)) and \
                 all(prices[i] < prices[i+j] for j in range(1, window_size+1)):
                turn_points.append({
                    "type": "low",
                    "price": prices[i],
                    "index": i,
                    "timestamp": datetime.now()
                })
        return turn_points
    except Exception as e:
        logger.error(f"Error in turn point detection: {e}")
        return []

# Enhanced Trading Logic
async def trading_logic(amount_usd: float, prices: List[float], fee_rate: float = 0.001):
    try:
        turn_points = await detect_turn_points(prices)
        trades = []
        current_position = None
        
        for point in turn_points:
            if point["type"] == "low" and not current_position:
                # Buy at low point
                current_position = {
                    "entry_price": point["price"],
                    "amount_usd": amount_usd,
                    "entry_time": datetime.now()
                }
                trades.append({
                    "type": "buy",
                    "price": point["price"],
                    "amount": amount_usd,
                    "fee": amount_usd * fee_rate,
                    "timestamp": datetime.now()
                })
                
            elif point["type"] == "high" and current_position:
                # Calculate potential profit
                gross_profit = (point["price"] - current_position["entry_price"]) * (amount_usd / current_position["entry_price"])
                fee = amount_usd * fee_rate
                net_profit = gross_profit - fee
                
                # Check if profit target is met (10% net profit)
                if net_profit >= amount_usd * 0.1:
                    trades.append({
                        "type": "sell",
                        "price": point["price"],
                        "amount": amount_usd,
                        "fee": fee,
                        "gross_profit": gross_profit,
                        "net_profit": net_profit,
                        "timestamp": datetime.now()
                    })
                    current_position = None
                    
        return trades
    except Exception as e:
        logger.error(f"Error in trading logic: {e}")
        return []

async def broadcast_trade_update(trade):
    """Broadcast a trade update to all connected clients."""
    if not connected_clients:
        return

    try:
        # Prepare trade data for broadcast
        trade_data = {
            "id": trade.get("id"),
            "coin": trade.get("coin"),
            "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),  # Use amount_usdt, fallback to amount
            "entry_price": trade.get("entry_price", 0.0),
            "current_price": trade.get("current_price", 0.0),
            "status": trade.get("status"),
            "profit_loss": trade.get("profit_loss", 0.0),
            "fees": trade.get("fees", 0.0)
        }
        
        # Broadcast to all connected clients
        for client in connected_clients:
            try:
                await client.send_json(trade_data)
            except Exception as e:
                logger.error(f"Error sending trade update to client: {str(e)}")
                # Don't remove the client here, let the disconnect handler handle it
    except Exception as e:
        logger.error(f"Error preparing trade update broadcast: {str(e)}")

async def broadcast_price_updates():
    """Broadcast price updates to all connected clients."""
    while True:
        try:
            for trade in active_trades:
                if trade["status"] == "Open":
                    try:
                        current_price = await fetch_price(trade["coin"])
                        if current_price:
                            trade["current_price"] = current_price
                            await broadcast_trade_update(trade)
                    except Exception as e:
                        logger.error(f"Error updating price for trade {trade.get('id')}: {str(e)}")
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in price update broadcast: {str(e)}")
            await asyncio.sleep(5)  # Wait before retrying

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    async def monitor_trades():
        """Monitor active trades and update prices"""
        while True:
            try:
                for trade in active_trades[:]:  # Create a copy of the list to avoid modification during iteration
                    if trade["status"] == "Open":
                        try:
                            # Get current price
                            current_price = await fetch_price(trade["coin"])
                            if current_price is None:
                                continue
                            
                            # Update trade data
                            trade["current_price"] = current_price
                            
                            # Calculate metrics
                            metrics = calculate_trade_metrics(trade, current_price)
                            trade.update(metrics)
                            trade["fees"] = metrics["fees"] # Ensure total fees are updated in the trade object
                            
                            # Check for auto-close conditions
                            if trade["roi"] >= 10:  # Take profit at 10%
                                await close_trade(trade["id"])
                            elif trade["roi"] <= -5:  # Stop loss at -5%
                                await close_trade(trade["id"])
                            
                            # Save trades to file
                            save_trades()
                            
                            # Broadcast update
                            await broadcast_trade_update(trade)
                            
                        except Exception as e:
                            logger.error(f"Error updating trade {trade['id']}: {str(e)}")
                            continue
                
                # Automated Trading Logic for each coin
                for coin_symbol in COIN_MAPPING.keys():
                    if coin_symbol in price_history and len(price_history[coin_symbol]) >= PRICE_HISTORY_LENGTH:
                        prices = list(price_history[coin_symbol])
                        current_price = prices[-1] # Most recent price
                        
                        turn_points = await detect_turn_points(prices)
                        
                        # Get active trade for this coin, if any
                        active_trade_for_coin = next((t for t in active_trades if t["coin"] == coin_symbol and t["status"] == "Open"), None)
                        
                        for point in turn_points:
                            # Automated Buy Logic
                            if point["type"] == "low" and not active_trade_for_coin:
                                # Determine amount for automated buy (e.g., 100 USD equivalent of coin)
                                # For simplicity, let's assume a fixed USD amount for automated trades.
                                # In a real scenario, this would come from an available balance.
                                auto_trade_amount_usd = 100.0
                                
                                # Calculate coin amount based on current price
                                if current_price > 0:
                                    amount_in_coin = auto_trade_amount_usd / current_price
                                else:
                                    logger.warning(f"Current price for {coin_symbol} is zero, cannot place buy order.")
                                    continue

                                new_trade_request = {
                                    "coin": coin_symbol,
                                    "amount": amount_in_coin, # This is in coin units
                                    "order_type": "market",
                                    "limit_price": None
                                }
                                try:
                                    # Call create_trade function directly as if from an API call
                                    trade_data = await create_trade_internal(new_trade_request)
                                    logger.info(f"AUTO-BUY: Placed new trade for {coin_symbol} at {trade_data.get("trade", {}).get("entry_price")}")
                                    # After creating a trade, refresh the active_trade_for_coin variable
                                    active_trade_for_coin = next((t for t in active_trades if t["coin"] == coin_symbol and t["status"] == "Open"), None)
                                except Exception as e:
                                    logger.error(f"AUTO-BUY: Error placing automated buy for {coin_symbol}: {str(e)}")

                            # Automated Sell Logic
                            elif point["type"] == "high" and active_trade_for_coin:
                                # Calculate metrics at the high turn point
                                metrics_at_high = calculate_trade_metrics(active_trade_for_coin, current_price)
                                net_profit_loss = metrics_at_high["profit_loss"]
                                roi = metrics_at_high["roi"]

                                # Check if 10% net profit requirement is met
                                if roi >= 10:  # 10% ROI target
                                    try:
                                        await close_trade(active_trade_for_coin["id"])
                                        logger.info(f"AUTO-SELL: Closed trade {active_trade_for_coin['id']} for {coin_symbol} with {roi:.2f}% ROI (Net P/L: {net_profit_loss:.2f})")
                                        active_trade_for_coin = None # Clear current position after closing
                                    except Exception as e:
                                        logger.error(f"AUTO-SELL: Error closing automated trade {active_trade_for_coin['id']}: {str(e)}")
                                else:
                                    logger.info(f"AUTO-SELL: Holding trade {active_trade_for_coin['id']} for {coin_symbol}. High turn point detected but ROI ({roi:.2f}%) < 10% target.")

                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in monitor_trades: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error

    async def broadcast_price_updates():
        """Broadcast price updates to all connected clients."""
        while True:
            try:
                for trade in active_trades:
                    current_price = await fetch_price(trade["coin"])
                    if current_price:
                        await broadcast_trade_update(trade)
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in price update broadcast: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    # Start both monitoring tasks
    asyncio.create_task(monitor_trades())
    asyncio.create_task(broadcast_price_updates())
    yield

# Initialize FastAPI app with lifespan
app = FastAPI(title="Crypto Trading Bot", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
# WebSocket connection handler
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"New WebSocket connection. Total clients: {len(connected_clients)}")
    
    try:
        # Send initial trade data
        for trade in active_trades:
            try:
                trade_data = {
                    "id": trade.get("id"),
                    "coin": trade.get("coin"),
                    "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),  # Use amount_usdt, fallback to amount
                    "entry_price": trade.get("entry_price", 0.0),
                    "current_price": trade.get("current_price", 0.0),
                    "status": trade.get("status"),
                    "profit_loss": trade.get("profit_loss", 0.0),
                    "fees": trade.get("fees", 0.0)
                }
                await websocket.send_json(trade_data)
            except Exception as e:
                logger.error(f"Error sending initial trade data: {str(e)}")
                break
        
        # Keep the connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle any incoming messages if needed
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket connection closed. Total clients: {len(connected_clients)}")

# Route: Home Page with Trade Reporting
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    # Ensure all trades have the correct field names
    processed_trades = []
    for trade in active_trades:
        processed_trade = {
            "id": trade.get("id"),
            "coin": trade.get("coin"),
            "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),  # Use amount_usdt, fallback to amount
            "entry_price": trade.get("entry_price", 0.0),
            "current_price": trade.get("current_price", 0.0),
            "exit_price": trade.get("exit_price"),
            "status": trade.get("status"),
            "profit_loss": trade.get("profit_loss", 0.0),
            "fees": trade.get("fees", 0.0),
            "entry_time": trade.get("entry_time"),
            "exit_time": trade.get("exit_time")
        }
        processed_trades.append(processed_trade)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_trades": processed_trades,
        "trade_history": trade_history
    })

class TradeRequest(BaseModel):
    coin: str
    amount: float
    order_type: str
    limit_price: Optional[float] = None

async def _create_trade_logic(trade_request: TradeRequest):
    try:
        # Validate amount
        if trade_request.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        
        # Validate limit price for limit orders
        if trade_request.order_type == "limit" and (trade_request.limit_price is None or trade_request.limit_price <= 0):
            raise HTTPException(status_code=400, detail="Limit price must be greater than 0 for limit orders")
        
        # Get current price
        current_price = await fetch_price(trade_request.coin)
        if current_price is None:
            raise HTTPException(status_code=400, detail="Invalid coin code or price unavailable")
        
        # Convert values to float
        amount_usdt = float(trade_request.amount)
        current_price = float(current_price)
        
        # Calculate quantity based on USDT amount
        quantity = amount_usdt / current_price
        
        # Calculate initial fees (0.1% entry fee)
        entry_value = amount_usdt
        entry_fee = round(entry_value * 0.001, 2)  # 0.1% entry fee

        # Create trade data
        trade = {
            "id": str(uuid.uuid4()),
            "coin": trade_request.coin,
            "amount_usdt": amount_usdt,  # Store the USDT amount
            "quantity": round(quantity, 8),  # Store the calculated quantity
            "order_type": trade_request.order_type,
            "entry_price": current_price,
            "current_price": current_price,
            "status": "Open",
            "limit_price": float(trade_request.limit_price) if trade_request.limit_price else None,
            "entry_time": datetime.now().isoformat(),
            "exit_time": None,
            "profit_loss": 0.0,
            "fees": entry_fee  # Initial fee is just the entry fee
        }
        
        # Add to active trades
        active_trades.append(trade)
        
        # Save trades to file
        save_trades()
        
        # Broadcast trade update
        await broadcast_trade_update(trade)
        
        return {
            "status": "success",
            "message": "Trade created successfully",
            "trade": trade
        }
        
    except Exception as e:
        logger.error(f"Error creating trade: {str(e)}")
        raise

@app.post("/trade")
async def create_trade(trade_request: TradeRequest):
    try:
        response = await _create_trade_logic(trade_request)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error handling trade creation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_trade_metrics(trade, current_price):
    """Calculate trade metrics including profit/loss and fees."""
    try:
        # Convert all values to float and ensure they are positive
        amount_usdt = abs(float(trade.get("amount_usdt", 0.0)))
        entry_price = abs(float(trade.get("entry_price", 0.0)))
        current_price = abs(float(current_price))

        # Calculate quantity if not present
        if "quantity" not in trade:
            trade["quantity"] = amount_usdt / entry_price if entry_price > 0 else 0.0
        quantity = abs(float(trade["quantity"]))

        # Calculate values
        entry_value = quantity * entry_price
        current_value = quantity * current_price

        # Calculate gross profit/loss
        gross_profit_loss = current_value - entry_value

        # Calculate fees (0.1% on entry and exit)
        entry_fee = entry_value * 0.001  # 0.1% entry fee
        exit_fee = current_value * 0.001  # 0.1% exit fee
        total_fees = entry_fee + exit_fee

        # Calculate net profit/loss
        net_profit_loss = gross_profit_loss - total_fees

        # Calculate ROI
        roi = (net_profit_loss / entry_value) * 100 if entry_value > 0 else 0

        return {
            "current_price": current_price,
            "profit_loss": round(net_profit_loss, 2),
            "fees": round(total_fees, 2),
            "roi": round(roi, 2)
        }

    except Exception as e:
        logger.error(f"Error calculating trade metrics: {str(e)}")
        return {
            "current_price": current_price,
            "profit_loss": 0.0,
            "fees": 0.0,
            "roi": 0.0
        }

@app.post("/close-trade/{trade_id}")
async def close_trade(trade_id: str):
    """Close a trade and move it to history."""
    try:
        # Find the trade
        trade = next((t for t in active_trades if t["id"] == trade_id), None)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        # Get current price
        current_price = await fetch_price(trade["coin"])
        if current_price is None:
            raise HTTPException(status_code=400, detail="Could not fetch current price")
        
        # Update trade data
        trade["current_price"] = current_price
        trade["status"] = "Closed"
        trade["exit_time"] = datetime.now().isoformat()
        
        # Calculate final metrics
        metrics = calculate_trade_metrics(trade, current_price)
        trade.update(metrics)
        
        # Move to history
        active_trades.remove(trade)
        trade_history.append(trade)
        
        # Save trades to file
        save_trades()
        
        # Broadcast update
        await broadcast_trade_update(trade)
        
        return {"status": "success", "message": "Trade closed successfully"}
        
    except Exception as e:
        logger.error(f"Error closing trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-trades")
async def get_active_trades():
    """Get all active trades."""
    return active_trades

@app.get("/trade-history")
async def get_trade_history():
    """Get trade history."""
    return trade_history

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
