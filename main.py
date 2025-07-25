from fastapi import FastAPI, Form, HTTPException, WebSocket, Request, Depends, Response, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import requests
from typing import Optional, List, Dict, Set, Tuple
import asyncio
import logging
from datetime import datetime, timedelta, timezone
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
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_FEE, 
    TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE, PRICE_FETCH_DELAY,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT, TURNING_POINT_MARGIN, TURN_POINT_WINDOW,
    DEFAULT_MARGIN
)
import math
import secrets
from starlette.middleware.sessions import SessionMiddleware
import numpy as np
import argparse
import pandas as pd
import hmac
import hashlib
import requests as pyrequests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_server_time():
    """Get the current server time from Binance"""
    try:
        response = requests.get('https://api.binance.com/api/v3/time')
        if response.status_code == 200:
            return response.json()['serverTime']
        return None
    except Exception as e:
        logger.error(f"Error getting server time: {str(e)}")
        return None

def sync_time():
    """Synchronize local time with Binance server time"""
    server_time = get_server_time()
    if server_time:
        local_time = int(time.time() * 1000)
        time_diff = server_time - local_time
        logger.info(f"Time difference with Binance server: {time_diff}ms")
        return time_diff
    return 0

# Add session middleware secret key
SECRET_KEY = secrets.token_hex(32)

# Add security
security = HTTPBasic()

# Store active trades and trade history
active_trades = []
trade_history = []
connected_clients: Set[WebSocket] = set()

# User credentials (in production, use a proper database)
USERS = {
    "armen": "armen123"  # username: password
}

# Session management
sessions = {}

def create_session(username: str) -> str:
    """Create a new session for a user"""
    session_id = secrets.token_hex(16)
    sessions[session_id] = {
        "username": username,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=24)
    }
    return session_id

def get_session(session_id: str) -> Optional[dict]:
    """Get session data if it exists and is not expired"""
    if session_id in sessions:
        session = sessions[session_id]
        if datetime.utcnow() < session["expires_at"]:
            return session
        else:
            del sessions[session_id]
    return None

def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]

# Authentication dependency
async def get_current_user(request: Request) -> Optional[str]:
    """Get the current user from the session"""
    session_id = request.cookies.get("session_id")
    if session_id:
        session = get_session(session_id)
        if session:
            return session["username"]
    return None

# Authentication required dependency
async def require_auth(request: Request) -> str:
    """Require authentication for protected routes"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=302,
            detail="Not authenticated",
            headers={"Location": "/login"}
        )
    return user

async def get_symbol_precision(symbol: str) -> int:
    """Get the quantity precision for a symbol"""
    try:
        exchange_info = binance_client.get_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            # Get the quantity precision from the symbol info
            quantity_precision = 0
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    quantity_precision = len(str(step_size).rstrip('0').split('.')[-1])
                    break
            return quantity_precision
        return 8  # Default precision
    except Exception as e:
        logger.error(f"Error getting symbol precision: {str(e)}")
        return 8  # Default precision

def handle_binance_error(error: BinanceAPIException) -> HTTPException:
    """Handle common Binance API errors"""
    error_codes = {
        -1010: "Invalid quantity",
        -1013: "Invalid price",
        -2010: "Insufficient balance",
        -2015: "Invalid API key",
        -2014: "Invalid signature",
        -1021: "Timestamp for this request is outside of the recvWindow",
        -1022: "Signature for this request is not valid"
    }
    
    error_message = error_codes.get(error.code, str(error))
    logger.error(f"Binance API error: {error_message}")
    
    # Return appropriate HTTP status code based on error type
    if error.code in [-2015, -2014, -1021, -1022]:  # Authentication errors
        return HTTPException(status_code=401, detail=error_message)
    elif error.code in [-2010]:  # Balance errors
        return HTTPException(status_code=400, detail=error_message)
    else:  # Other errors
        return HTTPException(status_code=500, detail=error_message)

async def validate_order_parameters(symbol: str, quantity: float, price: float = None) -> bool:
    """Validate order parameters against Binance requirements"""
    try:
        exchange_info = binance_client.get_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        
        if not symbol_info:
            return False
            
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                min_qty = float(filter['minQty'])
                max_qty = float(filter['maxQty'])
                step_size = float(filter['stepSize'])
                
                # Check quantity against min/max
                if quantity < min_qty:
                    logger.error(f"Quantity {quantity} is below minimum {min_qty}")
                    return False
                if quantity > max_qty:
                    logger.error(f"Quantity {quantity} is above maximum {max_qty}")
                    return False
                    
                # Check if quantity is a multiple of step size
                remainder = quantity % step_size
                if remainder > 0:
                    # Round to nearest valid step
                    quantity = round(quantity / step_size) * step_size
                    logger.info(f"Adjusted quantity to {quantity} to match step size {step_size}")
                    
            elif filter['filterType'] == 'PRICE_FILTER' and price:
                min_price = float(filter['minPrice'])
                max_price = float(filter['maxPrice'])
                tick_size = float(filter['tickSize'])
                
                # Check price against min/max
                if price < min_price:
                    logger.error(f"Price {price} is below minimum {min_price}")
                    return False
                if price > max_price:
                    logger.error(f"Price {price} is above maximum {max_price}")
                    return False
                    
                # Check if price is a multiple of tick size
                remainder = price % tick_size
                if remainder > 0:
                    # Round to nearest valid tick
                    price = round(price / tick_size) * tick_size
                    logger.info(f"Adjusted price to {price} to match tick size {tick_size}")
                    
        return True
    except Exception as e:
        logger.error(f"Error validating order parameters: {str(e)}")
        return False

# Helper function to format datetime
def format_datetime(iso_timestamp: str) -> str:
    try:
        # Accept both with and without 'Z'
        if iso_timestamp.endswith('Z'):
            dt_object = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        else:
            dt_object = datetime.fromisoformat(iso_timestamp)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return iso_timestamp # Return original if format is incorrect

# Price cache with timestamp
price_cache = {}
CACHE_DURATION = 60  # Cache duration in seconds
RATE_LIMIT_DELAY = PRICE_FETCH_DELAY  # Use the configured delay
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

# File path for orange points persistence
ORANGE_POINTS_FILE = "orange_points.json"

# Initialize Binance client with API credentials
try:
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise ValueError("Binance API credentials are not configured")
    
    # Initialize client
    binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    #binance_client.API_URL = 'https://testnet.binance.vision/api'
    
    # Synchronize time with Binance server
    time_diff = sync_time()
    if time_diff != 0:
        logger.info(f"Adjusting for time difference of {time_diff}ms")
        # Set the time offset in the client's request handler
        binance_client.timestamp_offset = time_diff
    
    # Test the connection
    binance_client.get_account()
    logger.info("Successfully connected to Binance API")
except BinanceAPIException as e:
    if e.code == -2015:  # Invalid API key
        logger.error("Invalid Binance API key. Please check your API credentials.")
        raise ValueError("Invalid Binance API key. Please check your API credentials.")
    elif e.code == -2014:  # Invalid signature
        logger.error("Invalid Binance API signature. Please check your API secret.")
        raise ValueError("Invalid Binance API signature. Please check your API secret.")
    else:
        logger.error(f"Binance API error during initialization: {str(e)}")
        raise ValueError(f"Binance API error: {str(e)}")
except Exception as e:
    logger.error(f"Error initializing Binance client: {str(e)}")
    raise ValueError(f"Error initializing Binance client: {str(e)}")

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
                    # Ensure each trade has a unique ID
                    if 'id' not in trade:
                        trade['id'] = str(uuid.uuid4())

                # Ensure 'roi' field is present in all loaded trade history
                for trade in loaded_trade_history:
                    if 'roi' not in trade:
                        trade['roi'] = 0.0
                    # Ensure each trade has a unique ID
                    if 'id' not in trade:
                        trade['id'] = str(uuid.uuid4())
                        
                return loaded_active_trades, loaded_trade_history
    except Exception as e:
        logger.error(f"Error loading trades: {str(e)}")
    return [], []

def save_trades():
    """Save trades to the JSON file."""
    try:
        # Remove any duplicate trades based on ID
        unique_active_trades = []
        seen_ids = set()
        for trade in active_trades:
            if trade['id'] not in seen_ids:
                seen_ids.add(trade['id'])
                unique_active_trades.append(trade)

        unique_trade_history = []
        seen_ids = set()
        for trade in trade_history:
            if trade['id'] not in seen_ids:
                seen_ids.add(trade['id'])
                unique_trade_history.append(trade)

        data = {
            'active_trades': unique_active_trades,
            'trade_history': unique_trade_history
        }
        with open(TRADES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving trades: {str(e)}")

# Initialize trades from file
active_trades, trade_history = load_trades()

# Add this global set near the top of the file, after other globals
canceled_auto_buy_coins = set()

# Add global dict to store orange points per symbol
orange_points_by_symbol = {}

def load_orange_points():
    """Load orange points from the JSON file."""
    try:
        if os.path.exists(ORANGE_POINTS_FILE):
            with open(ORANGE_POINTS_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                try:
                    data = json.loads(content)
                    return data
                except json.JSONDecodeError:
                    logger.error(f"orange_points.json contains invalid JSON. Initializing as empty.")
                    return {}
    except Exception as e:
        logger.error(f"Error loading orange points: {str(e)}")
    return {}

def save_orange_points():
    """Save orange points to the JSON file."""
    try:
        with open(ORANGE_POINTS_FILE, 'w') as f:
            json.dump(orange_points_by_symbol, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving orange points: {str(e)}")

# Load orange points from file on startup
orange_points_by_symbol = load_orange_points()

async def execute_binance_order(symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> dict:
    """
    Execute a real order on Binance
    side: 'BUY' or 'SELL'
    order_type: 'MARKET' or 'LIMIT'
    """
    try:
        # Map coin name to Binance symbol only if provided a coin (e.g., "bitcoin").
        # If the caller already supplies a Binance symbol like "BTCUSDT", keep it.
        mapped_symbol = SYMBOL_MAPPING.get(symbol.lower())
        if mapped_symbol:
            symbol = mapped_symbol
        # Normalise symbol to upper case for downstream processing / logging consistency
        symbol = symbol.upper()
        # Validation: a Binance symbol must end with USDT for our spot trading logic
        if not symbol or not symbol.endswith('USDT'):
            raise HTTPException(status_code=400, detail=f"Unsupported coin or symbol: {symbol}")
        
        # Get exchange info for the symbol
        try:
            exchange_info = binance_client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if not symbol_info:
                raise HTTPException(status_code=400, detail=f"Could not get exchange info for {symbol}")
        except Exception as e:
            logger.error(f"Error getting exchange info: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error getting exchange info: {str(e)}")
        
        # Get current price directly from Binance for the symbol
        try:
            ticker = binance_client.get_symbol_ticker(symbol=symbol)
            if not ticker or 'price' not in ticker:
                raise HTTPException(status_code=400, detail="Could not fetch current price")
            current_price = float(ticker['price'])
            logger.info(f"Current price for {symbol}: {current_price}")
        except BinanceAPIException as e:
            logger.error(f"Binance API error while fetching price: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error fetching price: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while fetching price: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error fetching price: {str(e)}")
        
        # Convert values to float
        quantity = float(quantity)  # quantity is already the asset amount to trade
        logger.info(
            f"Received quantity parameter: {quantity} {symbol.replace('USDT', '')} (will validate & format)"
        )
        
        # Get symbol precision and format quantity
        try:
            # Get LOT_SIZE filter
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if not lot_size_filter:
                raise HTTPException(status_code=400, detail=f"Could not get LOT_SIZE filter for {symbol}")
            
            step_size = float(lot_size_filter['stepSize'])
            min_qty = float(lot_size_filter['minQty'])
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            
            # Round DOWN to the nearest valid step size
            quantity = math.floor(quantity / step_size) * step_size
            formatted_quantity = format(quantity, f'.{precision}f')
            logger.info(f"Formatted quantity: {formatted_quantity} {symbol.replace('USDT', '')}")
            
            # Calculate actual amount that will be used
            actual_amount = float(formatted_quantity) * current_price
            if float(formatted_quantity) < min_qty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Calculated quantity {formatted_quantity} is below minimum {min_qty} for {symbol}"
                )
            # Warn if actual required amount is less than a reasonable threshold (optional)
            if actual_amount < 0.01:  # You can adjust this threshold as needed
                raise HTTPException(
                    status_code=400,
                    detail=f"After rounding, the order value is too small: {actual_amount:.2f} USDT. Please enter a higher amount."
                )
        except Exception as e:
            logger.error(f"Error formatting quantity: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error formatting quantity: {str(e)}")
        
        # Check balance before executing order
        if side == 'BUY':
            usdt_balance = await get_binance_balance('USDT')
            required_usdt = float(formatted_quantity) * (price if price else current_price)
            if usdt_balance < required_usdt:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient USDT balance. Required: {required_usdt:.2f} USDT, Available: {usdt_balance:.2f} USDT"
                )
        else:  # SELL order
            # Extract the base asset (e.g., BTC from BTCUSDT)
            base_asset = symbol.replace('USDT', '')
            
            # Get account information to check actual balance
            account = binance_client.get_account()
            coin_balance = None
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    coin_balance = float(balance['free'])
                    break
                    
            if not coin_balance or coin_balance <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No {base_asset} balance available"
                )
            
            # Use the actual balance as quantity, ensuring it meets minimum quantity
            if coin_balance < min_qty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Balance {coin_balance} {base_asset} is below minimum quantity {min_qty} {base_asset}"
                )
            
            # Round down to nearest step size to ensure we don't exceed balance
            quantity = math.floor(coin_balance / step_size) * step_size
            formatted_quantity = format(quantity, f'.{precision}f')
            logger.info(f"Using available balance: {formatted_quantity} {base_asset} for sell order")
            
            # Double check the balance is still available
            if float(formatted_quantity) > coin_balance:
                raise HTTPException(
                    status_code=400,
                    detail=f"Calculated quantity {formatted_quantity} exceeds available balance {coin_balance} {base_asset}"
                )
        
        # Execute the order
        try:
            if order_type == 'MARKET':
                # For market orders, we only need symbol, side, type, and quantity
                params = {
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': formatted_quantity
                }
                logger.info(f"Executing market order with params: {params}")
                order = binance_client.create_order(**params)
            else:  # LIMIT order
                if not price:
                    raise HTTPException(status_code=400, detail="Price is required for limit orders")
                    
                # Get price filter info
                price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if not price_filter:
                    raise HTTPException(status_code=400, detail=f"Could not get price filter for {symbol}")
                
                # Validate price range
                min_price = float(price_filter['minPrice'])
                max_price = float(price_filter['maxPrice'])
                tick_size = float(price_filter['tickSize'])
                
                if price < min_price:
                    raise HTTPException(status_code=400, detail=f"Price {price} is below minimum {min_price}")
                if price > max_price:
                    raise HTTPException(status_code=400, detail=f"Price {price} is above maximum {max_price}")
                
                # Round price to nearest tick
                price = round(price / tick_size) * tick_size
                
                # Get price precision from tick size
                price_precision = len(str(tick_size).rstrip('0').split('.')[-1])
                formatted_price = format(price, f'.{price_precision}f')
                
                logger.info(f"Creating limit order with formatted price {formatted_price} and quantity {formatted_quantity}")
                
                params = {
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'timeInForce': 'GTC',
                    'quantity': formatted_quantity,
                    'price': formatted_price
                }
                order = binance_client.create_order(**params)
            return order
        except BinanceAPIException as e:
            if e.code == -1010:  # Invalid quantity
                raise HTTPException(status_code=400, detail=f"Invalid quantity: {formatted_quantity}")
            elif e.code == -1013:  # Invalid price
                raise HTTPException(status_code=400, detail=f"Invalid price for {symbol}")
            else:
                raise handle_binance_error(e)
                
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error executing order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_binance_balance(asset: str = 'USDT') -> float:
    """Get the balance for a specific asset from Binance"""
    try:
        account = binance_client.get_account()
        for balance in account['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0
    except BinanceAPIException as e:
        logger.error(f"Binance API error while getting balance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
            logger.error(f"Unsupported coin: {coin}")
            return None

        # Fetch price from Binance
        ticker = binance_client.get_symbol_ticker(symbol=symbol)
        if not ticker or 'price' not in ticker:
            logger.error(f"Invalid ticker response for {symbol}: {ticker}")
            return None
            
        price = float(ticker['price'])
        if price <= 0:
            logger.error(f"Invalid price for {symbol}: {price}")
            return None
        
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

async def broadcast_trade_update(trade=None):
    """Broadcast a trade update to all connected clients."""
    if not connected_clients:
        return

    try:
        # Prepare all active trades data
        trades_data = []
        for active_trade in active_trades:
            trade_data = {
                "id": active_trade.get("id"),
                "coin": active_trade.get("coin"),
                "amount_usdt": active_trade.get("amount_usdt", active_trade.get("amount", 0.0)),
                "filled_amount_usdt": active_trade.get("filled_amount_usdt", active_trade.get("amount_usdt", 0.0)),
                "entry_price": active_trade.get("entry_price", 0.0),
                "current_price": active_trade.get("current_price", 0.0),
                "status": active_trade.get("status"),
                "profit_loss": active_trade.get("profit_loss", 0.0),
                "fees": active_trade.get("buy_fee", active_trade.get("fees", 0.0)),
                "roi": active_trade.get("roi", 0.0),
                "take_profit": active_trade.get("take_profit", TAKE_PROFIT_PERCENTAGE),
                "take_profit_type": active_trade.get("take_profit_type", "dollar"),
                "stop_loss": active_trade.get("stop_loss"),  # Just pass the value as is
                "entry_time": format_datetime(active_trade.get("entry_time")) if active_trade.get("entry_time") else "N/A",
                "binance_order_id": active_trade.get("binance_order_id", ""),
                "sell_binance_order_id": active_trade.get("sell_binance_order_id", "")
            }
            trades_data.append(trade_data)
        
        # Broadcast to all connected clients
        for client in connected_clients:
            try:
                await client.send_json({
                    "type": "trade_update",
                    "trades": trades_data
                })
            except Exception as e:
                logger.error(f"Error sending trade update to client: {str(e)}")
    except Exception as e:
        logger.error(f"Error preparing trade update broadcast: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Verify Binance connection on startup
    try:
        binance_client.get_account()
        logger.info("Successfully verified Binance API connection")
    except BinanceAPIException as e:
        logger.error(f"Failed to verify Binance API connection: {str(e)}")
        raise ValueError(f"Failed to verify Binance API connection: {str(e)}")
    except Exception as e:
        logger.error(f"Error verifying Binance API connection: {str(e)}")
        raise ValueError(f"Error verifying Binance API connection: {str(e)}")

    async def monitor_trades():
        while True:
            try:
                for trade in active_trades[:]:
                    try:
                        # Get current price
                        current_price = await fetch_price(trade["coin"])
                        if current_price is None:
                            continue
                        if "price_history" not in trade:
                            trade["price_history"] = []
                        last_price = trade["price_history"][-1] if trade["price_history"] else None
                        if last_price is not None and current_price == last_price:
                            continue
                        trade["price_history"].append(current_price)
                        if len(trade["price_history"]) > 20:
                            trade["price_history"] = trade["price_history"][-20:]
                        trade["current_price"] = current_price
                        if trade["status"] == "Pending" and trade["order_type"] == "limit":
                            symbol = SYMBOL_MAPPING.get(trade["coin"].lower())
                            if current_price <= trade["limit_price"]:
                                trade["status"] = "Open"
                                trade["entry_price"] = current_price
                                logger.info(f"Limit order executed for trade {trade['id']} at price {current_price}")
                                # Fetch the order trades from Binance and set filled_amount_usdt
                                try:
                                    if symbol and trade.get("binance_order_id"):
                                        trades = binance_client.get_my_trades(symbol=symbol)
                                        order_trades = [t for t in trades if t['orderId'] == trade["binance_order_id"]]
                                        filled_amount_usdt = sum(float(t['qty']) * float(t['price']) for t in order_trades)
                                        trade["filled_amount_usdt"] = filled_amount_usdt
                                except Exception as e:
                                    logger.error(f"Error fetching trades for filled_amount_usdt: {str(e)}")
                        if trade["status"] == "Open":
                            # Refresh filled_amount_usdt from Binance
                            try:
                                symbol = SYMBOL_MAPPING.get(trade["coin"].lower())
                                if symbol and trade.get("binance_order_id"):
                                    trades = binance_client.get_my_trades(symbol=symbol)
                                    order_trades = [t for t in trades if t['orderId'] == trade["binance_order_id"]]
                                    filled_amount_usdt = sum(float(t['qty']) * float(t['price']) for t in order_trades)
                                    trade["filled_amount_usdt"] = filled_amount_usdt
                            except Exception as e:
                                logger.error(f"Error refreshing filled_amount_usdt for open trade: {str(e)}")
                            metrics = calculate_trade_metrics(trade, current_price)
                            trade.update(metrics)
                            symbol = SYMBOL_MAPPING.get(trade["coin"].lower())
                            if symbol:
                                klines = binance_client.get_klines(
                                    symbol=symbol,
                                    interval=Client.KLINE_INTERVAL_5MINUTE,
                                    limit=100
                                )
                                closes = [float(k[4]) for k in klines]
                                logger.info(f"[SELL LOGIC] Last {TURN_POINT_WINDOW} closes: {closes[-TURN_POINT_WINDOW:]}")
                                should_sell = False
                                sell_reason = ""
                                profit_percentage = trade["roi"]
                                stop_loss_hit = trade.get("stop_loss") is not None and profit_percentage <= -trade["stop_loss"]
                                is_high_turn = is_high_turn_point(closes, TURN_POINT_WINDOW)
                                logger.info(f"[SELL LOGIC] Checking for high turn point: is_high_turn={is_high_turn}")
                                # Always log the required price drop and margin info
                                margin_value = trade.get("take_profit", TURNING_POINT_MARGIN)
                                margin_type = trade.get("take_profit_type", "dollar")
                                # Use filled_amount_usdt if available, else amount_usdt
                                base_amount = trade.get("filled_amount_usdt")
                                if base_amount is None or base_amount == 0:
                                    base_amount = trade.get("amount_usdt", 0)
                                if margin_type == "percentage":
                                    required_profit = (margin_value / 100) * base_amount
                                    required_price_drop = required_profit / trade['quantity']
                                else:
                                    required_profit = margin_value * trade['quantity']
                                    required_price_drop = margin_value
                                logger.info(
                                    f"[SELL LOGIC] Required price drop for margin: {required_price_drop:.6f} "
                                    f"Turning Point Sell (Profit: ${trade['profit_loss']:.2f} > Required Profit: ${required_profit:.2f} | Margin Param: {margin_value:.2f}{'%' if margin_type == 'percentage' else '$'})"
                                )
                                if is_high_turn and len(closes) >= TURN_POINT_WINDOW and closes[-1] < closes[-2] and closes[-2] < closes[-3]:
                                    peak_price = closes[- (TURN_POINT_WINDOW // 2) - 1]
                                    price_drop_from_peak = peak_price - closes[-1]
                                    margin_type = trade.get("take_profit_type", "dollar")
                                    margin_value = trade.get("take_profit", TURNING_POINT_MARGIN)
                                    # Use filled_amount_usdt if available, else amount_usdt
                                    base_amount = trade.get("filled_amount_usdt")
                                    if base_amount is None or base_amount == 0:
                                        base_amount = trade.get("amount_usdt", 0)
                                    if margin_type == "percentage":
                                        required_profit = (margin_value / 100) * base_amount
                                        required_price_drop = required_profit / trade['quantity']
                                    else:
                                        required_profit = margin_value * trade['quantity']
                                        required_price_drop = margin_value
                                    logger.info(f"[SELL LOGIC] peak_price={peak_price}, current_price={closes[-1]}, price_drop_from_peak={price_drop_from_peak}, required_price_drop={required_price_drop}, margin_value={margin_value}")
                                    # Check if current profit exceeds the margin threshold
                                    if trade['profit_loss'] > margin_value:
                                        should_sell = True
                                        sell_reason = f"Turning Point Sell (Profit: ${trade['profit_loss']:.2f} > Margin: {margin_value:.2f}{'%' if margin_type == 'percentage' else '$'})"
                                    else:
                                        logger.info(
                                            f"[SELL LOGIC] High turn detected and 3 down candles, but profit condition not met: "
                                            f"profit_loss={trade['profit_loss']:.2f}, margin_value={margin_value:.2f}{'%' if margin_type == 'percentage' else '$'}"
                                        )
                                        # Record orange point for this symbol
                                        symbol_orange_points = orange_points_by_symbol.setdefault(symbol, [])
                                        peak_candle = klines[- (TURN_POINT_WINDOW // 2) - 1]
                                        peak_time = peak_candle[6] // 1000  # close time in seconds
                                        peak_time_iso = datetime.fromtimestamp(peak_time, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
                                        if not any(pt['time'] == peak_time_iso for pt in symbol_orange_points):
                                            symbol_orange_points.append({
                                                "time": peak_time_iso,
                                                "price": peak_price
                                            })
                                            save_orange_points()
                                elif stop_loss_hit:
                                    should_sell = True
                                    sell_reason = "Stop Loss"
                                if should_sell:
                                    logger.info(f"Selling {symbol} due to {sell_reason} (ROI: {trade['roi']:.2f}%)")
                                    await close_trade(trade["id"])
                                    continue
                        # Broadcast update whenever price changes
                        try:
                            await broadcast_trade_update(trade)
                        except Exception as e:
                            logger.error(f"Error broadcasting trade update: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error monitoring trade {trade.get('id', 'unknown')}: {str(e)}")
                        continue
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in monitor_trades: {str(e)}")
                await asyncio.sleep(60)

    # Start monitoring task
    asyncio.create_task(monitor_trades())

    # Add this function to monitor and execute pending orders based on low turn point logic
    async def monitor_pending_orders():
        while True:
            try:
                for trade in active_trades:
                    if trade["status"] == "Pending" and trade.get("auto_buy_loop"):
                        symbol = SYMBOL_MAPPING.get(trade["coin"].lower())
                        if not symbol:
                            continue
                        klines = binance_client.get_klines(
                            symbol=symbol,
                            interval=Client.KLINE_INTERVAL_5MINUTE,
                            limit=100
                        )
                        closes = [float(k[4]) for k in klines]
                        log1 = f"[BUY LOGIC] Last {TURN_POINT_WINDOW} closes: {closes[-TURN_POINT_WINDOW:]}"
                        logger.info(log1)
                        await broadcast_logic_log(log1)
                        is_low_turn = is_low_turn_point(closes, TURN_POINT_WINDOW)
                        log2 = f"[BUY LOGIC] Checking for low turn point: is_low_turn={is_low_turn}"
                        logger.info(log2)
                        await broadcast_logic_log(log2)
                        should_buy = False  # Initialize here to avoid unbound variable error
                        # Check if this is the initial buy (auto_buy_iteration == 1) or a subsequent auto-buy
                        is_initial_buy = trade.get("auto_buy_iteration", 1) == 1
                        
                        # Different logic for initial vs subsequent auto-buy
                        if is_initial_buy:
                            # Initial buy: only low turn point logic (no price rise threshold)
                            if is_low_turn and len(closes) >= TURN_POINT_WINDOW and closes[-1] > closes[-2] and closes[-2] > closes[-3]:
                                log3 = f"[BUY LOGIC] Initial buy: Low turn point detected - executing buy order immediately"
                                logger.info(log3)
                                await broadcast_logic_log(log3)
                                should_buy = True
                        else:
                            # Subsequent auto-buy: low turn point + price rise threshold
                            if is_low_turn and len(closes) >= TURN_POINT_WINDOW and closes[-1] > closes[-2] and closes[-2] > closes[-3]:
                                trough_price = closes[- (TURN_POINT_WINDOW // 2) - 1]
                                price_rise_from_trough = closes[-1] - trough_price
                                # Calculate required price increase based on margin (always in USDT)
                                margin_value = trade.get("take_profit", TURNING_POINT_MARGIN)
                                margin_type = trade.get("take_profit_type", "dollar")
                                # Use filled_amount_usdt if available, else amount_usdt
                                base_amount = trade.get("filled_amount_usdt")
                                if base_amount is None or base_amount == 0:
                                    base_amount = trade.get("amount_usdt", 0)
                                if margin_type == "percentage":
                                    required_profit = (margin_value / 100) * base_amount
                                    required_price_increase = required_profit / trade['quantity']
                                else:
                                    required_profit = margin_value * trade['quantity']
                                    required_price_increase = margin_value
                                logger_msg = (
                                    f"[BUY LOGIC] Required price increase for margin: {required_price_increase:.6f} "
                                    f"Turning Point Buy (Price Rise: {price_rise_from_trough:.6f} > Required Profit: ${required_profit:.2f} | Margin Param: {margin_value:.2f}{'%' if margin_type == 'percentage' else '$'})"
                                )
                                logger.info(logger_msg)
                                await broadcast_logic_log(logger_msg)
                                if price_rise_from_trough > required_price_increase:
                                    should_buy = True
                                    log4 = f"[BUY LOGIC] Subsequent buy: Low turn point detected with sufficient price rise - executing buy order"
                                    logger.info(log4)
                                    await broadcast_logic_log(log4)
                                else:
                                    log5 = (
                                        f"[BUY LOGIC] Low turn detected and 3 up candles, but price rise/profit condition not met: "
                                        f"price_rise_from_trough={price_rise_from_trough:.2f}, required_price_increase={required_price_increase:.2f}, required_profit={required_profit:.2f}"
                                    )
                                    logger.info(log5)
                                    await broadcast_logic_log(log5)
                            
                        if should_buy:
                                try:
                                    order = await execute_binance_order(
                                        symbol=symbol,
                                        side='BUY',
                                        order_type='MARKET',
                                        quantity=trade["quantity"]
                                    )
                                    trade["status"] = "Open"
                                    trade["binance_order_id"] = order['orderId']
                                    trade["entry_price"] = float(order['fills'][0]['price'])
                                    trade["current_price"] = float(order['fills'][0]['price'])
                                    trade["fees"] = sum(float(fill['commission']) for fill in order['fills'])
                                    trade["entry_time"] = format_datetime(datetime.utcnow().isoformat())
                                    trade["auto_buy_iteration"] = trade["auto_buy_iteration"] + 1
                                    save_trades()
                                    await broadcast_trade_update(trade)
                                    logger.info(f"Executed pending order for {symbol} at low turn point.")
                                except Exception as e:
                                    logger.error(f"Error executing pending order: {str(e)}")
            except Exception as e:
                logger.error(f"Error in monitor_pending_orders: {str(e)}")
            await asyncio.sleep(60)  # Check every 30 seconds

    # Start monitoring task
    asyncio.create_task(monitor_pending_orders())

    yield

# Initialize FastAPI app with lifespan
app = FastAPI(title="Crypto Trading Bot", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"New WebSocket connection. Total clients: {len(connected_clients)}")
    try:
        # Send all active trades in a single trade_update message
        try:
            trades_data = []
            for trade in active_trades:
                trade_data = {
                    "id": trade.get("id"),
                    "coin": trade.get("coin"),
                    "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),
                    "filled_amount_usdt": trade.get("filled_amount_usdt", trade.get("amount_usdt", 0.0)),
                    "entry_price": trade.get("entry_price", 0.0),
                    "current_price": trade.get("current_price", 0.0),
                    "status": trade.get("status"),
                    "profit_loss": trade.get("profit_loss", 0.0),
                    "fees": trade.get("buy_fee", trade.get("fees", 0.0)),
                    "roi": trade.get("roi", 0.0),
                    "take_profit": trade.get("take_profit", TAKE_PROFIT_PERCENTAGE),
                    "take_profit_type": trade.get("take_profit_type", "dollar"),
                    "stop_loss": trade.get("stop_loss"),  # Just pass the value as is
                    "entry_time": format_datetime(trade.get("entry_time")) if trade.get("entry_time") else "N/A",
                    "binance_order_id": trade.get("binance_order_id", ""),
                    "sell_binance_order_id": trade.get("sell_binance_order_id", "")
                }
                trades_data.append(trade_data)
            await websocket.send_json({
                "type": "trade_update",
                "trades": trades_data
            })
        except Exception as e:
            logger.error(f"Error sending initial trade data: {str(e)}")

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
async def home(request: Request, user: str = Depends(require_auth)):
    """Render the home page."""
    def trade_key(trade):
        return (
            trade.get("coin"),
            float(trade.get("amount_usdt", trade.get("amount", 0.0))),
            float(trade.get("entry_price") or 0.0),
            trade.get("status")
        )
    unique_trades = {}
    for trade in active_trades:
        key = trade_key(trade)
        if key not in unique_trades:
            current_price = await fetch_price(trade["coin"])
            if current_price is None:
                current_price = trade.get("current_price", 0.0)
            metrics = calculate_trade_metrics(trade, current_price)
            processed_trade = {
                "id": trade.get("id"),
                "coin": trade.get("coin"),
                "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),
                "filled_amount_usdt": trade.get("filled_amount_usdt", trade.get("amount_usdt", 0.0)),
                "entry_price": trade.get("entry_price", 0.0),
                "current_price": current_price,
                "status": trade.get("status"),
                "profit_loss": metrics.get("profit_loss", 0.0),
                "fees": round(metrics.get("fees", 0.0), 3),
                "roi": metrics.get("roi", 0.0),
                "take_profit": trade.get("take_profit", TAKE_PROFIT_PERCENTAGE),
                "take_profit_type": trade.get("take_profit_type", "percentage"),
                "stop_loss": trade.get("stop_loss"),
                "entry_time": format_datetime(trade.get("entry_time")) if trade.get("entry_time") else "N/A",
                "binance_order_id": trade.get("binance_order_id", ""),
                "sell_binance_order_id": trade.get("sell_binance_order_id", "")
            }
            unique_trades[key] = processed_trade
    processed_trades = list(unique_trades.values())

    def history_key(trade):
        return (
            trade.get("coin"),
            float(trade.get("amount_usdt", trade.get("amount", 0.0))),
            float(trade.get("entry_price", 0.0)),
            float(trade.get("exit_price", 0.0)),
            trade.get("status")
        )
    unique_history = {}
    for trade in trade_history:
        key = history_key(trade)
        if key not in unique_history:
            processed_history = {
                "id": trade.get("id"),
                "coin": trade.get("coin"),
                "amount_usdt": trade.get("amount_usdt", trade.get("amount", 0.0)),
                "filled_amount_usdt": trade.get("filled_amount_usdt", trade.get("amount_usdt", 0.0)),
                "entry_price": trade.get("entry_price", 0.0),
                "exit_price": trade.get("exit_price", None),
                "profit_loss": trade.get("profit_loss", 0.0),
                "fees": trade.get("fees", 0.0),
                "buy_fee": trade.get("buy_fee", 0.0),
                "sell_fee": trade.get("sell_fee", 0.0),
                "status": trade.get("status"),
                "entry_time": format_datetime(trade.get("entry_time")) if trade.get("entry_time") else "N/A",
                "exit_time": format_datetime(trade.get("exit_time")) if trade.get("exit_time") else "N/A",
                "roi": trade.get("roi", 0.0),
                "take_profit": trade.get("take_profit", TAKE_PROFIT_PERCENTAGE),
                "stop_loss": trade.get("stop_loss"),
                "binance_order_id": trade.get("binance_order_id", ""),
                "sell_binance_order_id": trade.get("sell_binance_order_id", ""),
                "duration": calculate_duration(trade.get("entry_time"), trade.get("exit_time")),
                "sell_amount_usdt": trade.get("sell_amount_usdt")
            }
            unique_history[key] = processed_history
    processed_history = list(unique_history.values())

    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_trades": processed_trades,
        "trade_history": processed_history,
        "config": {"DEFAULT_MARGIN": DEFAULT_MARGIN}
    })

class TradeRequest(BaseModel):
    coin: str
    amount: float
    order_type: str
    limit_price: Optional[float] = None
    take_profit: Optional[float] = None
    take_profit_type: Optional[str] = "percentage"
    stop_loss: Optional[float] = None

async def _create_trade_logic(trade_request: TradeRequest):
    try:
        # Validate amount
        if trade_request.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        
        # Map coin name to Binance symbol only if provided a coin (e.g., "bitcoin").
        # If the caller already supplies a Binance symbol like "BTCUSDT", keep it.
        mapped_symbol = SYMBOL_MAPPING.get(trade_request.coin.lower())
        if mapped_symbol:
            symbol = mapped_symbol
        # Normalise symbol to upper case for downstream processing / logging consistency
        symbol = symbol.upper()
        # Validation: a Binance symbol must end with USDT for our spot trading logic
        if not symbol or not symbol.endswith('USDT'):
            raise HTTPException(status_code=400, detail=f"Unsupported coin or symbol: {symbol}")
        
        # Get exchange info for the symbol
        try:
            exchange_info = binance_client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if not symbol_info:
                raise HTTPException(status_code=400, detail=f"Could not get exchange info for {symbol}")
        except Exception as e:
            logger.error(f"Error getting exchange info: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error getting exchange info: {str(e)}")
        
        # Get current price directly from Binance for the symbol
        try:
            ticker = binance_client.get_symbol_ticker(symbol=symbol)
            if not ticker or 'price' not in ticker:
                raise HTTPException(status_code=400, detail="Could not fetch current price")
            current_price = float(ticker['price'])
            logger.info(f"Current price for {symbol}: {current_price}")
        except BinanceAPIException as e:
            logger.error(f"Binance API error while fetching price: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error fetching price: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while fetching price: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error fetching price: {str(e)}")
        
        # Convert values to float (USDT amount)
        amount_usdt = float(trade_request.amount)
        logger.info(f"Creating trade with amount: {amount_usdt} USDT")
        
        # Calculate quantity based on USDT amount
        quantity = amount_usdt / current_price
        logger.info(f"Calculated quantity: {quantity} {trade_request.coin}")
        
        # Get symbol precision and format quantity
        try:
            # Get LOT_SIZE filter
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if not lot_size_filter:
                raise HTTPException(status_code=400, detail=f"Could not get LOT_SIZE filter for {symbol}")
            
            step_size = float(lot_size_filter['stepSize'])
            min_qty = float(lot_size_filter['minQty'])
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            
            # Round DOWN to the nearest valid step size
            quantity = math.floor(quantity / step_size) * step_size
            formatted_quantity = format(quantity, f'.{precision}f')
            logger.info(f"Formatted quantity: {formatted_quantity} {symbol.replace('USDT', '')}")
            
            # Calculate actual amount that will be used
            actual_amount = float(formatted_quantity) * current_price
            if float(formatted_quantity) < min_qty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Calculated quantity {formatted_quantity} is below minimum {min_qty} for {symbol}"
                )
            # Warn if actual required amount is less than a reasonable threshold (optional)
            if actual_amount < 0.01:  # You can adjust this threshold as needed
                raise HTTPException(
                    status_code=400,
                    detail=f"After rounding, the order value is too small: {actual_amount:.2f} USDT. Please enter a higher amount."
                )
        except Exception as e:
            logger.error(f"Error formatting quantity: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error formatting quantity: {str(e)}")
        
        # Check if we have enough USDT balance (already checked above, but keep for logic consistency)
        try:
            usdt_balance = await get_binance_balance('USDT')
            logger.info(f"Current USDT balance: {usdt_balance}")
            if usdt_balance < amount_usdt:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient USDT balance. Required: {amount_usdt:.2f} USDT, Available: {usdt_balance:.2f} USDT"
                )
        except Exception as e:
            logger.error(f"Error checking USDT balance: {str(e)}")
            raise HTTPException(status_code=400, detail=f"An issue occurred while verifying your USDT balance. Please try again.")
        
        # Execute the order on Binance
        try:
            if trade_request.order_type == "market":
                logger.info(f"Creating market order for {symbol} with quantity {formatted_quantity}")
                order = await execute_binance_order(
                    symbol=symbol,
                    side='BUY',
                    order_type='MARKET',
                    quantity=float(formatted_quantity)
                )
                entry_price = float(order['fills'][0]['price'])
                status = "Open"
                binance_order_id = order['orderId']
                logger.info(f"Buy order ID: {binance_order_id}")
                # Calculate actual filled amount in USDT
                try:
                    trades = binance_client.get_my_trades(symbol=symbol)
                    order_trades = [t for t in trades if t['orderId'] == order['orderId']]
                    filled_amount_usdt = sum(float(t['qty']) * float(t['price']) for t in order_trades)
                except Exception as e:
                    logger.error(f"Error fetching trades for filled_amount_usdt (market order): {str(e)}")
                    if 'fills' in order and order['fills']:
                        filled_amount_usdt = sum(float(fill['qty']) * float(fill['price']) for fill in order['fills'])
                    else:
                        filled_amount_usdt = amount_usdt  # fallback
            else:  # limit order
                if not trade_request.limit_price or trade_request.limit_price <= 0:
                    raise HTTPException(status_code=400, detail="Limit price must be greater than 0 for limit orders")
                
                # Get price filter info
                price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if not price_filter:
                    raise HTTPException(status_code=400, detail=f"Could not get price filter for {symbol}")
                
                # Validate and format limit price
                min_price = float(price_filter['minPrice'])
                max_price = float(price_filter['maxPrice'])
                tick_size = float(price_filter['tickSize'])
                
                limit_price = float(trade_request.limit_price)
                
                # Validate price range
                if limit_price < min_price:
                    raise HTTPException(status_code=400, detail=f"Limit price {limit_price} is below minimum {min_price}")
                if limit_price > max_price:
                    raise HTTPException(status_code=400, detail=f"Limit price {limit_price} is above maximum {max_price}")
                
                # Validate price is within reasonable range of current price (within 20%)
                price_difference_percent = abs(limit_price - current_price) / current_price * 100
                if price_difference_percent > 20:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Limit price {limit_price} is too far from current price {current_price}. "
                              f"Price must be within 20% of current price."
                    )
                
                # Round price to nearest tick
                limit_price = round(limit_price / tick_size) * tick_size
                
                logger.info(f"Creating limit order with price {limit_price} (current price: {current_price}) and quantity {formatted_quantity}")
                
                # Create the limit order
                order = await execute_binance_order(
                    symbol=symbol,
                    side='BUY',
                    order_type='LIMIT',
                    quantity=float(formatted_quantity),
                    price=limit_price
                )
                
                # For limit orders, we use the limit price as entry price and set status to Pending
                entry_price = limit_price
                status = "Pending"
        except BinanceAPIException as e:
            logger.error(f"Binance API error while creating order: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating order: {str(e)}")
        except HTTPException as e:
            # Propagate HTTP errors (generated inside execute_binance_order) with full detail
            logger.error(f"HTTP error while creating order: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error while creating order: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating order: {str(e)}")
        
        # Calculate initial fees (only for market orders)
        entry_fee = round(amount_usdt * TRADING_FEE, 3) if status == "Open" else 0.0

        # Take profit logic disabled - no take profit price calculation
        take_profit_price = None
        
        # Create trade data
        trade = {
            "id": str(uuid.uuid4()),
            "coin": trade_request.coin,
            "amount_usdt": amount_usdt,
            "filled_amount_usdt": filled_amount_usdt,
            "quantity": float(formatted_quantity),
            "order_type": trade_request.order_type,
            "entry_price": entry_price,
            "current_price": current_price,
            "status": status,
            "limit_price": float(trade_request.limit_price) if trade_request.order_type == "limit" else None,
            "entry_time": utc_iso_now(),
            "profit_loss": 0.0,
            "fees": entry_fee,
            "buy_fee": entry_fee,
            "roi": 0.0,
            "binance_order_id": order['orderId'],
            "take_profit": trade_request.take_profit if trade_request.take_profit is not None else DEFAULT_MARGIN,
            "take_profit_type": trade_request.take_profit_type if trade_request.take_profit_type else "dollar",
            "stop_loss": trade_request.stop_loss,
            "auto_buy_loop": True,  # Mark this trade for auto-rebuy
            "auto_buy_iteration": 2 if trade_request.order_type == "market" else 1  # For market: next buy is subsequent, for limit: next buy is initial
        }
        
        # Add to active trades
        active_trades.append(trade)
        
        # Save trades to file
        save_trades()
        
        # Broadcast the new trade immediately
        await broadcast_trade_update(trade)
        
        return trade
        
    except HTTPException as e:
        logger.error(f"Error creating trade: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error creating trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade")
async def create_trade(trade_request: TradeRequest, user: str = Depends(require_auth)):
    """Create a new trade."""
    try:
        trade = await _create_trade_logic(trade_request)
        await broadcast_trade_update(trade)
        return {
            "status": "success",
            "message": "Trade created successfully",
            "trade": trade
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error handling trade creation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_trade_metrics(trade, current_price):
    """Calculate trade metrics including profit/loss and fees."""
    try:
        # Ensure current_price is a float, default to 0 if None
        safe_current_price = float(current_price) if current_price is not None else 0.0

        # For pending trades or trades without a valid entry price, we can't calculate metrics.
        entry_price_val = trade.get("entry_price")
        if entry_price_val is None or float(entry_price_val) <= 0:
            return {
                "current_price": safe_current_price,
                "profit_loss": 0.0,
                "fees": 0.0,
                "roi": 0.0,
                "buy_fee": 0.0,
                "sell_fee": 0.0
            }

        # Convert all values to float and ensure they are positive
        amount_usdt = abs(float(trade.get("amount_usdt", 0.0)))
        entry_price = abs(float(entry_price_val))

        # Calculate quantity if not present
        if "quantity" not in trade:
            trade["quantity"] = amount_usdt / entry_price
        quantity = abs(float(trade.get("quantity", 0.0)))

        # Calculate values
        entry_value = quantity * entry_price
        current_value = quantity * safe_current_price

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
            "current_price": safe_current_price,
            "profit_loss": round(net_profit_loss, 2),
            "fees": round(total_fees, 3),
            "roi": round(roi, 2),
            "buy_fee": round(entry_fee, 3),
            "sell_fee": round(exit_fee, 3)
        }

    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating trade metrics for trade {trade.get('id')}: {str(e)}")
        # Fallback for any other conversion errors
        safe_current_price = float(current_price) if current_price is not None else 0.0
        return {
            "current_price": safe_current_price,
            "profit_loss": 0.0,
            "fees": 0.0,
            "roi": 0.0,
            "buy_fee": 0.0,
            "sell_fee": 0.0
        }

@app.post("/close-trade/{trade_id}")
async def close_trade(trade_id: str, manual: bool = Query(False), user: str = Depends(require_auth)):
    """Close a trade with real Binance sell order."""
    try:
        # Find the trade
        trade = next((t for t in active_trades if t["id"] == trade_id), None)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        if trade["status"] != "Open":
            raise HTTPException(status_code=400, detail="Can only close open trades")
        
        # Get Binance symbol
        symbol = SYMBOL_MAPPING.get(trade["coin"].lower())
        if not symbol:
            raise HTTPException(status_code=400, detail=f"Unsupported coin: {trade['coin']}")
        
        # Get the base asset (e.g., BTC for BTCUSDT)
        base_asset = symbol[:-4]
        
        try:
            # Get symbol info for quantity precision
            exchange_info = binance_client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if not symbol_info:
                raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found in exchange info")
            
            # Get quantity precision and step size
            quantity_precision = 0
            min_qty = 0
            step_size = 0
            
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    min_qty = float(filter['minQty'])
                    quantity_precision = len(str(step_size).rstrip('0').split('.')[-1])
                    break

            # Get current balance
            account = binance_client.get_account()
            coin_balance = None
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    coin_balance = float(balance['free'])
                    break

            if not coin_balance or coin_balance <= 0:
                raise HTTPException(status_code=400, detail=f"No {base_asset} balance available")


            # Use the trade's quantity directly
            quantity_to_sell = float(trade["quantity"])
            
            # Format quantity according to step size
            quantity_to_sell = math.floor(quantity_to_sell / step_size) * step_size
            formatted_quantity = format(quantity_to_sell, f'.{quantity_precision}f')
            
            logger.info(f"Attempting to sell {formatted_quantity} {base_asset} from trade {trade_id}")
            
            # Execute the sell order
            order = await execute_binance_order(
                symbol=symbol,
                side='SELL',
                order_type='MARKET',
                quantity=float(formatted_quantity)
            )
            trade['sell_binance_order_id'] = order.get('orderId', '')
            
            if not order or 'fills' not in order or not order['fills']:
                raise HTTPException(status_code=400, detail="Failed to execute sell order")

            # Calculate total filled quantity and cost
            total_filled_quantity = sum(float(fill['qty']) for fill in order['fills'])
            total_cost = sum(float(fill['price']) * float(fill['qty']) for fill in order['fills'])
            exit_price = total_cost / total_filled_quantity if total_filled_quantity > 0 else 0

            # Update trade data
            trade["current_price"] = exit_price
            trade["status"] = "Closed"
            trade["exit_time"] = utc_iso_now()
            trade["exit_price"] = exit_price
            trade["exit_quantity"] = total_filled_quantity
            trade["sell_amount_usdt"] = total_cost
            # Calculate final metrics
            metrics = calculate_trade_metrics(trade, exit_price)
            trade.update(metrics)
            
            # Move to trade history
            trade_history.append(trade)
            
            # Remove from active trades
            active_trades.remove(trade)
            
            # --- AUTO-BUY LOOP: Create new pending trade if enabled ---
            if not manual and trade.get('auto_buy_loop', False):
                previous_amount = trade.get("filled_amount_usdt", trade.get("amount_usdt", 0))
                previous_profit_loss = trade.get("profit_loss", 0)
                next_amount = previous_amount + previous_profit_loss
                if next_amount <= 0:
                    next_amount = previous_amount  # Prevent negative or zero entry
                new_trade = {
                    "id": str(uuid.uuid4()),
                    "coin": trade["coin"],
                    "amount_usdt": next_amount,
                    "quantity": next_amount / exit_price if exit_price else 0,
                    "order_type": trade["order_type"],
                    "entry_price": None,
                    "current_price": exit_price,
                    "status": "Pending",
                    "limit_price": None,
                    "entry_time": utc_iso_now(),
                    "profit_loss": 0.0,
                    "fees": 0.0,
                    "roi": 0.0,
                    "binance_order_id": None,
                    "take_profit": trade.get("take_profit"),
                    "take_profit_type": trade.get("take_profit_type", "dollar"),
                    "stop_loss": trade.get("stop_loss"),
                    "auto_buy_loop": True,
                    "auto_buy_iteration": trade.get("auto_buy_iteration", 1) + 1
                }
                active_trades.append(new_trade)
                save_trades()
                await broadcast_trade_update(new_trade)

            # Save trades to file
            save_trades()
            
            # Broadcast update
            await broadcast_trade_update(trade)
            
            return {
                "status": "success",
                "message": "Trade closed successfully",
                "trade": trade
            }
            
        except BinanceAPIException as e:
            error_msg = f"Binance API error while closing trade: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        except Exception as e:
            error_msg = f"Error executing sell order: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException as e:
        logger.error(f"Error closing trade: {str(e)}")
        raise e
    except Exception as e:
        error_msg = f"Unexpected error closing trade: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/decline-trade/{trade_id}")
async def decline_trade(trade_id: str, user: str = Depends(require_auth)):
    """Decline a pending trade."""
    try:
        # Find the trade
        trade = next((t for t in active_trades if t["id"] == trade_id), None)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        # Check if trade is pending
        if trade["status"] != "Pending":
            raise HTTPException(status_code=400, detail="Can only decline pending trades")
        # Get Binance symbol
        symbol = SYMBOL_MAPPING.get(trade["coin"].lower())
        if not symbol:
            raise HTTPException(status_code=400, detail=f"Unsupported coin: {trade['coin']}")
        # If it's an auto-buy order (has rsi_trigger)
        if trade.get("rsi_trigger") is not None:
            logger.info(f"Declining auto-buy order {trade_id} for {symbol}")
            # Mark this coin as canceled for auto-buy loop
            canceled_auto_buy_coins.add(trade["coin"].lower())
            # Just remove from active trades since no Binance order exists yet
            active_trades.remove(trade)
        else:
            # For regular limit orders, try to cancel on Binance
            try:
                if trade["binance_order_id"]:
                    binance_client.cancel_order(
                        symbol=symbol,
                        orderId=trade["binance_order_id"]
                    )
                    logger.info(f"Successfully canceled Binance order {trade['binance_order_id']} for trade {trade_id}")
            except BinanceAPIException as e:
                if e.code == -2011:  # Order does not exist
                    logger.warning(f"Order {trade['binance_order_id']} already canceled or filled")
                else:
                    raise HTTPException(status_code=500, detail=f"Error canceling Binance order: {str(e)}")
            # Remove from active trades
            active_trades.remove(trade)
        # Save trades to file
        save_trades()
        # Broadcast update
        await broadcast_trade_update(trade)
        return {
            "status": "success", 
            "message": "Trade declined successfully",
            "trade_id": trade_id
        }
    except Exception as e:
        logger.error(f"Error declining trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-trades")
async def get_active_trades(user: str = Depends(require_auth)):
    """Get all active trades."""
    # Remove any duplicate trades based on ID
    unique_trades = []
    seen_ids = set()
    for trade in active_trades:
        if trade['id'] not in seen_ids:
            seen_ids.add(trade['id'])
            # Ensure buy_fee is present and correct
            trade = dict(trade)  # Copy to avoid mutating global
            if 'buy_fee' not in trade or trade['buy_fee'] is None:
                trade['buy_fee'] = round(trade.get('amount_usdt', 0.0) * TRADING_FEE, 3)
            # For active trades, set 'fees' to 'buy_fee' so frontend always shows only the buy fee
            trade['fees'] = trade['buy_fee']
            # Add sell_amount_usdt if present
            trade['sell_amount_usdt'] = trade.get('sell_amount_usdt')
            unique_trades.append(trade)
    return unique_trades

@app.get("/account-balance")
async def get_account_balance(user: str = Depends(require_auth)):
    """Get account balance from Binance."""
    try:
        # Get account information from Binance
        account_info = binance_client.get_account()
        
        # Get all balances, including zero balances
        balances = [
            {
                "asset": balance["asset"],
                "free": float(balance["free"]),
                "locked": float(balance["locked"]),
                "total": float(balance["free"]) + float(balance["locked"])
            }
            for balance in account_info["balances"]
        ]
        
        # Sort balances by total amount (highest first)
        balances.sort(key=lambda x: x["total"], reverse=True)
        
        return {
            "status": "success", 
            "balances": balances,
            "message": "Showing all balances, including zero balances"
        }
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error while fetching account balance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching account balance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    if username in USERS and USERS[username] == password:
        session_id = create_session(username)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=86400,  # 24 hours
            samesite="lax"
        )
        return response
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password"}
    )

@app.get("/logout")
async def logout(response: Response):
    """Handle user logout"""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session_id")
    return response

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index) for a list of prices."""
    if len(prices) < period + 1:
        return 50.0  # Return neutral RSI if not enough data
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

def calculate_macd(prices: List[float], fast_period: int, slow_period: int, signal_period: int) -> Tuple[float, float]:
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    macd_line = ema_fast - ema_slow
    # For signal line, use a simple EMA of the macd_line (not a rolling window)
    signal_line = calculate_ema([macd_line], signal_period)
    return macd_line, signal_line

def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int, d_period: int) -> Tuple[float, float]:
    lowest_low = np.min(lows[-k_period:])
    highest_high = np.max(highs[-k_period:])
    k_line = 100 * ((closes[-1] - lowest_low) / (highest_high - lowest_low)) if highest_high != lowest_low else 0.0
    d_line = calculate_sma([k_line], d_period)
    return k_line, d_line

def calculate_bollinger_bands(prices: List[float], period: int, num_std: float) -> Tuple[float, float, float]:
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

def print_rsi_status(symbol: str, rsi: float, is_oversold: bool = False):
    """Print RSI status in a clear, visible format."""
    print("\n" + "="*50)
    print(f"RSI Status for {symbol}")
    print("="*50)
    print(f"Current RSI: {rsi:.2f}")
    if is_oversold:
        print("Status: OVERSOLD - Ready to execute order!")
    else:
        print("Status: Waiting for oversold condition")
    print("="*50 + "\n")

@app.get("/price-history")
async def get_price_history(symbol: str = 'BTCUSDT', range: int = 1):
    """Return OHLCV price history for the given symbol and range in days (1, 7, 30). Also returns orange points for high turn peaks where profit condition is not met."""
    try:
        klines = binance_client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_5MINUTE,
            f"{range} day ago UTC"
        )
        times = [datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).isoformat().replace('+00:00', 'Z') for k in klines]
        opens = [float(k[1]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        # Normalize orange point times to match candle times
        def normalize_time(t):
            # If candles have 'Z', add 'Z' to orange points; else, remove 'Z'
            candle_has_z = times[0].endswith('Z')
            if t.endswith('Z') and not candle_has_z:
                return t[:-1]
            elif not t.endswith('Z') and candle_has_z:
                return t + 'Z'
            return t
        orange_points = [
            {**pt, "time": normalize_time(pt["time"])}
            for pt in orange_points_by_symbol.get(symbol, [])
            if normalize_time(pt['time']) >= times[0] and normalize_time(pt['time']) <= times[-1]
        ]
        return JSONResponse({
            "times": times,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "volumes": volumes,
            "orange_points": orange_points
        })
    except Exception as e:
        logger.error(f"Error fetching price history: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/trade-history")
async def get_trade_history(user: str = Depends(require_auth)):
    """Return the trade history (closed trades) as JSON."""
    return JSONResponse(trade_history)

@app.post("/auto-buy")
async def auto_buy(trade_request: TradeRequest, user: str = Depends(require_auth)):
    """Create a pending trade that will execute when a low turn point (turning point for buy) is detected."""
    try:
        symbol = SYMBOL_MAPPING.get(trade_request.coin.lower())
        if not symbol:
            raise HTTPException(status_code=400, detail=f"Unsupported coin: {trade_request.coin}")
        logger.info(f"Starting auto-buy process for {symbol}")
        usdt_balance = await get_binance_balance('USDT')
        if usdt_balance < float(trade_request.amount):
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient USDT balance. Required: {float(trade_request.amount):.2f} USDT, Available: {usdt_balance:.2f} USDT"
            )
        ticker = binance_client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        quantity = float(trade_request.amount) / current_price
        exchange_info = binance_client.get_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        step_size = float(lot_size_filter['stepSize'])
        min_qty = float(lot_size_filter['minQty'])
        precision = len(str(step_size).rstrip('0').split('.')[-1])
        quantity = math.ceil(quantity / step_size) * step_size
        formatted_quantity = format(quantity, f'.{precision}f')
        if float(formatted_quantity) < min_qty:
            raise HTTPException(
                status_code=400,
                detail=f"Calculated quantity {formatted_quantity} is below minimum {min_qty} for {symbol}"
            )
        trade = {
            "id": str(uuid.uuid4()),
            "coin": trade_request.coin,
            "amount_usdt": float(trade_request.amount),
            "quantity": float(formatted_quantity),
            "order_type": trade_request.order_type,
            "entry_price": None,
            "current_price": current_price,
            "status": "Pending",
            "limit_price": None,
            "entry_time": utc_iso_now(),
            "profit_loss": 0.0,
            "fees": 0.0,
            "roi": 0.0,
            "binance_order_id": None,
            "take_profit": trade_request.take_profit if trade_request.take_profit is not None else DEFAULT_MARGIN,
            "take_profit_type": trade_request.take_profit_type if trade_request.take_profit_type else "dollar",
            "stop_loss": trade_request.stop_loss,
            "auto_buy_loop": True,  # Mark this trade for auto-rebuy
            "auto_buy_iteration": 1  # Initial auto-buy
        }
        active_trades.append(trade)
        save_trades()
        await broadcast_trade_update(trade)
        return {
            "status": "success",
            "message": f"Pending order created. Will execute when a low turn point is detected.",
            "trade": trade
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in auto-buy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    asyncio.create_task(monitor_trades())
    asyncio.create_task(monitor_pending_orders())
    logger.info("Background tasks started")

@app.get("/logs/trading_bot.log")
def get_trading_log():
    """Serve the trading log file as plain text in the browser."""
    log_path = os.path.join("logs", "trading_bot.log")
    if not os.path.exists(log_path):
        return JSONResponse({"error": "Log file not found."}, status_code=404)
    def iterfile():
        with open(log_path, mode="r", encoding="utf-8") as file:
            for line in file:
                yield line
    return StreamingResponse(iterfile(), media_type="text/plain")

@app.get("/api/chart-data")
async def get_chart_data(days: int = 1, symbol: str = 'BTCUSDT'):
    """Return OHLCV price history for the given symbol and number of days, always including the latest candles."""
    try:
        # Use Binance's relative time string to always get up-to-date candles
        klines = binance_client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_5MINUTE,
            f"{days} day ago UTC"
        )
        times = [datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).isoformat().replace('+00:00', 'Z') for k in klines]
        opens = [float(k[1]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        return JSONResponse({
            "times": times,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "volumes": volumes
        })
    except Exception as e:
        logger.error(f"Error fetching chart data: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

def calculate_duration(entry_time, exit_time):
    if not entry_time or not exit_time or entry_time == "N/A" or exit_time == "N/A":
        return "N/A"
    try:
        entry_dt = datetime.fromisoformat(entry_time)
        exit_dt = datetime.fromisoformat(exit_time)
        duration = exit_dt - entry_dt
        days = duration.days
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        else:
            return f"{hours}h {minutes}m"
    except Exception:
        return "N/A"

def utc_iso_now():
    """Return the current UTC time as an ISO 8601 string with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

@app.post("/convert-btc-to-usdt")
async def convert_btc_to_usdt(user: str = Depends(require_auth)):
    """Convert all BTC in the account to USDT using Binance Convert API."""
    try:
        # Get BTC balance
        btc_balance = await get_binance_balance('BTC')
        if btc_balance is None or btc_balance <= 0:
            return {"status": "error", "detail": "No BTC balance to convert."}
        
        # Get quote
        api_key = BINANCE_API_KEY
        api_secret = BINANCE_API_SECRET
        base_url = 'https://api.binance.com'
        endpoint = '/sapi/v1/convert/getQuote'
        timestamp = int(time.time() * 1000)
        params = f'fromAsset=BTC&toAsset=USDT&fromAmount={btc_balance}&timestamp={timestamp}'
        signature = hmac.new(api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
        headers = {'X-MBX-APIKEY': api_key}
        quote_resp = pyrequests.post(
            base_url + endpoint + '?' + params + f'&signature={signature}',
            headers=headers
        )
        quote_data = quote_resp.json()
        if 'quoteId' not in quote_data:
            return {"status": "error", "detail": quote_data.get('msg', 'Failed to get quote.')}
        quote_id = quote_data['quoteId']
        # Accept quote
        endpoint2 = '/sapi/v1/convert/acceptQuote'
        timestamp2 = int(time.time() * 1000)
        params2 = f'quoteId={quote_id}&timestamp={timestamp2}'
        signature2 = hmac.new(api_secret.encode(), params2.encode(), hashlib.sha256).hexdigest()
        accept_resp = pyrequests.post(
            base_url + endpoint2 + '?' + params2 + f'&signature={signature2}',
            headers=headers
        )
        accept_data = accept_resp.json()
        if accept_resp.status_code == 200 and accept_data.get('orderStatus') == 'SUCCESS':
            return {"status": "success", "detail": "Converted all BTC to USDT."}
        else:
            return {"status": "error", "detail": accept_data.get('msg', 'Conversion failed.')}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/account-total-value")
async def get_account_total_value(user: str = Depends(require_auth)):
    """Get total account value in USDT and BTC balance."""
    try:
        account_info = binance_client.get_account()
        balances = [
            {
                "asset": balance["asset"],
                "free": float(balance["free"]),
                "locked": float(balance["locked"]),
                "total": float(balance["free"]) + float(balance["locked"])
            }
            for balance in account_info["balances"]
        ]
        total_usdt = 0.0
        btc_balance = 0.0
        for bal in balances:
            asset = bal["asset"]
            total = bal["total"]
            if asset == "USDT":
                total_usdt += total
            elif asset == "BTC":
                btc_balance = total
                # Get BTCUSDT price
                ticker = binance_client.get_symbol_ticker(symbol="BTCUSDT")
                price = float(ticker["price"])
                total_usdt += total * price
            elif total > 0:
                # Try to get price in USDT
                symbol = asset + "USDT"
                try:
                    ticker = binance_client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker["price"])
                    total_usdt += total * price
                except Exception:
                    pass  # skip if no direct USDT pair
        return {"status": "success", "total_value": total_usdt, "btc_balance": btc_balance}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

last_broadcasted_log = None

async def broadcast_logic_log(message: str):
    # Write to log file
    try:
        log_path = os.path.join("logs", "trading_bot.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        logger.error(f"Failed to write logic log to file: {e}")
    # Existing code to broadcast to WebSocket clients
    for ws in connected_clients:
        try:
            await ws.send_json({"type": "logic_log", "message": message})
        except Exception:
            pass

# Example usage in buy/sell logic:
# await broadcast_logic_log(log_message)
# Replace logger.info(f"[BUY LOGIC] ...") with both logger.info and broadcast_logic_log

@app.get("/test-logic-log")
async def test_logic_log():
    await broadcast_logic_log("[BUY LOGIC] Test message from /test-logic-log endpoint")
    return {"status": "sent"}

# Patch the logger to broadcast all logs to WebSocket clients
class WebSocketBroadcastHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        # Schedule the broadcast in the event loop
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(broadcast_logic_log(log_entry))
        except Exception:
            pass

# Add the handler to the root logger
ws_handler = WebSocketBroadcastHandler()
ws_handler.setLevel(logging.INFO)
ws_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(ws_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the FastAPI application')
    parser.add_argument('--port', type=int, default=9000, help='Port to run the server on (default: 8000)')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
