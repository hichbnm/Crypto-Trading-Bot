from fastapi import FastAPI, Form, HTTPException, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
from typing import Optional, List, Dict, Set, Tuple
import asyncio
import logging
from datetime import datetime
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
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_FEE, TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to format datetime
def format_datetime(iso_timestamp: str) -> str:
    try:
        dt_object = datetime.fromisoformat(iso_timestamp)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return iso_timestamp # Return original if format is incorrect

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

# Binance symbol mapping
SYMBOL_MAPPING = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "ripple": "XRPUSDT",
    "dogecoin": "DOGEUSDT",
    "polygon": "MATICUSDT",
    "chainlink": "LINKUSDT"
}

# File paths for persistent storage
TRADES_FILE = "trades.json"

# Initialize Binance client with API credentials
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def load_trades():
    """Load trades from the JSON file."""
    try:
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r') as f:
                data = json.load(f)
                loaded_active_trades = data.get('active_trades', [])
                loaded_trade_history = data.get('trade_history', [])

                # Ensure 'roi' field is present in all loaded active trades
                for trade in loaded_active_trades:
                    if 'roi' not in trade:
                        trade['roi'] = 0.0

                # Ensure 'roi' field is present in all loaded trade history
                for trade in loaded_trade_history:
                    if 'roi' not in trade:
                        trade['roi'] = 0.0
                        
                return loaded_active_trades, loaded_trade_history
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

async def fetch_price(coin: str) -> Optional[float]:
    """Fetch current price from Binance API with rate limiting and caching."""
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
        # Get Binance symbol
        symbol = SYMBOL_MAPPING.get(coin.lower())
        if not symbol:
            logger.error(f"Unsupported coin symbol: {coin}")
            return None

        # Fetch price from Binance
        ticker = binance_client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        
        # Update cache
        price_cache[coin] = (current_time, price)
        last_request_time = time.time()
        
        # Update price history
        if coin not in price_history:
            price_history[coin] = deque(maxlen=PRICE_HISTORY_LENGTH)
        price_history[coin].append(price)
        
        return price

    except BinanceAPIException as e:
        logger.error(f"Binance API error while fetching price for {coin}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching price for {coin}: {str(e)}")
        return None

async def broadcast_trade_update(trade):
    """Broadcast a trade update to all connected clients."""
    if not connected_clients:
        return

    try:
        # Prepare trade data for broadcast
        trade_data = {
            "id": trade.get("id"),
            "coin": trade.get("coin"),
            "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),
            "entry_price": trade.get("entry_price", 0.0),
            "current_price": trade.get("current_price", 0.0),
            "status": trade.get("status"),
            "profit_loss": trade.get("profit_loss", 0.0),
            "fees": trade.get("fees", 0.0),
            "roi": trade.get("roi", 0.0),
            "entry_time": format_datetime(trade.get("entry_time")) if trade.get("entry_time") else "N/A"
        }
        
        # Broadcast to all connected clients
        for client in connected_clients:
            try:
                await client.send_json(trade_data)
            except Exception as e:
                logger.error(f"Error sending trade update to client: {str(e)}")
    except Exception as e:
        logger.error(f"Error preparing trade update broadcast: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    async def monitor_trades():
        """Monitor active trades and update prices"""
        while True:
            try:
                for trade in active_trades[:]:
                    try:
                        # Get current price
                        current_price = await fetch_price(trade["coin"])
                        if current_price is None:
                            continue
                        
                        # Update trade data
                        trade["current_price"] = current_price
                        
                        # Handle limit orders
                        if trade["status"] == "Pending" and trade["order_type"] == "limit":
                            # Check if limit price is reached
                            if current_price <= trade["limit_price"]:
                                trade["status"] = "Open"
                                trade["entry_price"] = current_price
                                logger.info(f"Limit order executed for trade {trade['id']} at price {current_price}")
                        
                        # Only calculate metrics for open trades
                        if trade["status"] == "Open":
                            # Calculate metrics
                            metrics = calculate_trade_metrics(trade, current_price)
                            trade.update(metrics)
                            
                            # Check for auto-close conditions
                            if trade["roi"] >= TAKE_PROFIT_PERCENTAGE:  # Take profit
                                await close_trade(trade["id"])
                            elif trade["roi"] <= -STOP_LOSS_PERCENTAGE:  # Stop loss
                                await close_trade(trade["id"])
                        
                        # Save trades to file
                        save_trades()
                        
                        # Broadcast update
                        await broadcast_trade_update(trade)
                        
                    except Exception as e:
                        logger.error(f"Error updating trade {trade['id']}: {str(e)}")
                        continue
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in monitor_trades: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error

    # Start monitoring task
    asyncio.create_task(monitor_trades())
    yield

# Initialize FastAPI app with lifespan
app = FastAPI(title="Crypto Trading Bot", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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
                    "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),
                    "entry_price": trade.get("entry_price", 0.0),
                    "current_price": trade.get("current_price", 0.0),
                    "status": trade.get("status"),
                    "profit_loss": trade.get("profit_loss", 0.0),
                    "fees": trade.get("fees", 0.0),
                    "roi": trade.get("roi", 0.0),
                    "entry_time": format_datetime(trade.get("entry_time")) if trade.get("entry_time") else "N/A"
                }
                await websocket.send_json(trade_data)
            except Exception as e:
                logger.error(f"Error sending initial trade data: {str(e)}")
                break
        
        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket connection closed. Total clients: {len(connected_clients)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    processed_trades = []
    for trade in active_trades:
        # Ensure current_price is available for ROI calculation
        current_price = await fetch_price(trade["coin"])
        if current_price is None:
            current_price = trade.get("current_price", 0.0) # Use existing if fetch fails

        # Calculate metrics for the trade including ROI
        metrics = calculate_trade_metrics(trade, current_price)

        processed_trade = {
            "id": trade.get("id"),
            "coin": trade.get("coin"),
            "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),
            "entry_price": trade.get("entry_price", 0.0),
            "current_price": current_price,
            "status": trade.get("status"),
            "profit_loss": metrics.get("profit_loss", 0.0),
            "fees": metrics.get("fees", 0.0),
            "roi": metrics.get("roi", 0.0), # Add ROI here
            "entry_time": format_datetime(trade.get("entry_time")) if trade.get("entry_time") else "N/A"
        }
        processed_trades.append(processed_trade)

    processed_history = []
    for trade in trade_history:
        processed_history.append({
            "id": trade.get("id"),
            "coin": trade.get("coin"),
            "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),
            "entry_price": trade.get("entry_price", 0.0),
            "exit_price": trade.get("exit_price", None),
            "profit_loss": trade.get("profit_loss", 0.0),
            "fees": trade.get("fees", 0.0),
            "status": trade.get("status"),
            "entry_time": format_datetime(trade.get("entry_time")) if trade.get("entry_time") else "N/A",
            "exit_time": format_datetime(trade.get("exit_time")) if trade.get("exit_time") else "N/A",
            "roi": trade.get("roi", 0.0) # Ensure ROI is passed to history as well
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_trades": processed_trades,
        "trade_history": processed_history # Pass processed history
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
            "amount_usdt": amount_usdt,
            "quantity": round(quantity, 8),
            "order_type": trade_request.order_type,
            "entry_price": current_price if trade_request.order_type == "market" else float(trade_request.limit_price),
            "current_price": current_price,
            "status": "Pending" if trade_request.order_type == "limit" else "Open",
            "limit_price": float(trade_request.limit_price) if trade_request.order_type == "limit" else None,
            "entry_time": format_datetime(datetime.now().isoformat()),
            "profit_loss": 0.0,
            "fees": entry_fee,
            "roi": 0.0 # Initialize ROI to 0.0
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
    """Close a trade."""
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
        trade["exit_time"] = format_datetime(datetime.now().isoformat())
        
        # Calculate final metrics
        metrics = calculate_trade_metrics(trade, current_price)
        trade.update(metrics)
        
        # Move to trade history
        trade_history.append(trade)
        
        # Remove from active trades
        active_trades.remove(trade)
        
        # Save trades to file
        save_trades()
        
        # Broadcast update
        await broadcast_trade_update(trade)
        
        return {"status": "success", "message": "Trade closed successfully"}
        
    except Exception as e:
        logger.error(f"Error closing trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decline-trade/{trade_id}")
async def decline_trade(trade_id: str):
    """Decline a pending trade."""
    try:
        # Find the trade
        trade = next((t for t in active_trades if t["id"] == trade_id), None)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        # Check if trade is pending
        if trade["status"] != "Pending":
            raise HTTPException(status_code=400, detail="Can only decline pending trades")
        
        # Remove from active trades
        active_trades.remove(trade)
        
        # Save trades to file
        save_trades()
        
        # Broadcast update
        await broadcast_trade_update(trade)
        
        return {"status": "success", "message": "Trade declined successfully"}
        
    except Exception as e:
        logger.error(f"Error declining trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-trades")
async def get_active_trades():
    """Get all active trades."""
    return active_trades

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
