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
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_FEE, 
    TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE, PRICE_FETCH_DELAY
)
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize Binance client with API credentials
try:
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise ValueError("Binance API credentials are not configured")
    binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
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

async def execute_binance_order(symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> dict:
    """
    Execute a real order on Binance
    side: 'BUY' or 'SELL'
    order_type: 'MARKET' or 'LIMIT'
    """
    try:
        # Get exchange info for step size and precision
        exchange_info = binance_client.get_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if not symbol_info:
            raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
            
        # Get step size and precision
        step_size = None
        min_qty = None
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                step_size = float(filter['stepSize'])
                min_qty = float(filter['minQty'])
                break
                
        if not step_size or not min_qty:
            raise HTTPException(status_code=400, detail=f"Could not get lot size info for {symbol}")
            
        # Get symbol precision
        precision = len(str(step_size).rstrip('0').split('.')[-1])
        
        # Format quantity according to precision
        formatted_quantity = format(quantity, f'.{precision}f')
        
        # Check balance before executing order
        if side == 'BUY':
            usdt_balance = await get_binance_balance('USDT')
            # Get current price directly from Binance for the symbol
            try:
                ticker = binance_client.get_symbol_ticker(symbol=symbol)
                if not ticker or 'price' not in ticker:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not fetch current price for balance check"
                    )
                current_price = float(ticker['price'])
            except BinanceAPIException as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error fetching price: {str(e)}"
                )
                
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
        
        # Validate final quantity
        if float(formatted_quantity) < min_qty:
            raise HTTPException(
                status_code=400,
                detail=f"Quantity {formatted_quantity} is below minimum {min_qty}"
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
        
        # Get Binance symbol first
        symbol = SYMBOL_MAPPING.get(trade_request.coin.lower())
        if not symbol:
            raise HTTPException(status_code=400, detail=f"Unsupported coin: {trade_request.coin}")
        
        # Get current price directly from Binance for the symbol
        try:
            ticker = binance_client.get_symbol_ticker(symbol=symbol)
            if not ticker or 'price' not in ticker:
                raise HTTPException(status_code=400, detail="Could not fetch current price")
            current_price = float(ticker['price'])
        except BinanceAPIException as e:
            raise HTTPException(status_code=400, detail=f"Error fetching price: {str(e)}")
        
        # Convert values to float
        amount_usdt = float(trade_request.amount)
        
        # Calculate quantity based on USDT amount
        quantity = amount_usdt / current_price
        
        # Check if we have enough USDT balance
        usdt_balance = await get_binance_balance('USDT')
        if usdt_balance < amount_usdt:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient USDT balance. Required: {amount_usdt:.2f} USDT, Available: {usdt_balance:.2f} USDT"
            )
        
        # Get symbol precision and format quantity
        precision = await get_symbol_precision(symbol)
        formatted_quantity = format(quantity, f'.{precision}f')
        
        # Execute the order on Binance
        if trade_request.order_type == "market":
            order = await execute_binance_order(
                symbol=symbol,
                side='BUY',
                order_type='MARKET',
                quantity=float(formatted_quantity)
            )
            entry_price = float(order['fills'][0]['price'])
            status = "Open"
        else:  # limit order
            if not trade_request.limit_price or trade_request.limit_price <= 0:
                raise HTTPException(status_code=400, detail="Limit price must be greater than 0 for limit orders")
            
            # Get exchange info for price validation
            exchange_info = binance_client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if not symbol_info:
                raise HTTPException(status_code=400, detail=f"Could not get exchange info for {symbol}")
            
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
        
        # Calculate initial fees (only for market orders)
        entry_fee = round(amount_usdt * TRADING_FEE, 2) if status == "Open" else 0.0

        # Create trade data
        trade = {
            "id": str(uuid.uuid4()),
            "coin": trade_request.coin,
            "amount_usdt": amount_usdt,
            "quantity": float(formatted_quantity),
            "order_type": trade_request.order_type,
            "entry_price": entry_price,
            "current_price": current_price,
            "status": status,
            "limit_price": float(trade_request.limit_price) if trade_request.order_type == "limit" else None,
            "entry_time": format_datetime(datetime.now().isoformat()),
            "profit_loss": 0.0,
            "fees": entry_fee,
            "roi": 0.0,
            "binance_order_id": order['orderId']  # Store Binance order ID
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
        
    except HTTPException as e:
        logger.error(f"Error creating trade: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error creating trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade")
async def create_trade(trade_request: TradeRequest):
    """Create a new trade."""
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
    """Close a trade with real Binance sell order."""
    try:
        # Find the trade
        trade = next((t for t in active_trades if t["id"] == trade_id), None)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        # Get Binance symbol
        symbol = SYMBOL_MAPPING.get(trade["coin"].lower())
        if not symbol:
            raise HTTPException(status_code=400, detail=f"Unsupported coin: {trade['coin']}")
        
        try:
            # Get exchange info for step size and minimum quantity
            exchange_info = binance_client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if not symbol_info:
                raise HTTPException(status_code=400, detail=f"Could not get exchange info for {symbol}")
            
            # Get step size and minimum quantity
            step_size = None
            min_qty = None
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    min_qty = float(filter['minQty'])
                    break
            
            if not step_size or not min_qty:
                raise HTTPException(status_code=400, detail=f"Could not get lot size info for {symbol}")
            
            # Get current balance for the coin
            base_asset = symbol.replace('USDT', '')
            account = binance_client.get_account()
            coin_balance = None
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    coin_balance = float(balance['free'])
                    break
            
            if not coin_balance:
                raise HTTPException(
                    status_code=400,
                    detail=f"No {base_asset} balance available"
                )
            
            # Get precision from step size
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            
            # Round down to nearest step size to ensure we don't exceed balance
            quantity = math.floor(coin_balance / step_size) * step_size
            
            # Format quantity according to precision
            formatted_quantity = format(quantity, f'.{precision}f')
            
            logger.info(f"Available balance: {coin_balance} {base_asset}")
            logger.info(f"Minimum quantity: {min_qty} {base_asset}")
            logger.info(f"Formatted quantity: {formatted_quantity} {base_asset}")
            
            # Check if quantity meets minimum requirement
            if float(formatted_quantity) < min_qty:
                logger.warning(f"Quantity {formatted_quantity} {base_asset} is below minimum {min_qty} {base_asset}")
                # If quantity is too small, just mark the trade as closed without selling
                trade["status"] = "Closed"
                trade["exit_time"] = format_datetime(datetime.now().isoformat())
                trade["exit_price"] = trade["current_price"]
                trade["exit_quantity"] = float(formatted_quantity)
                
                # Calculate final metrics
                metrics = calculate_trade_metrics(trade, trade["current_price"])
                trade.update(metrics)
                
                # Move to trade history
                trade_history.append(trade)
                
                # Remove from active trades
                active_trades.remove(trade)
                
                # Save trades to file
                save_trades()
                
                # Broadcast update
                await broadcast_trade_update(trade)
                
                return {
                    "status": "success",
                    "message": f"Trade closed (quantity too small to sell: {formatted_quantity} {base_asset})",
                    "trade": trade
                }
            
            # Execute market sell order
            logger.info(f"Executing market sell order for {formatted_quantity} {base_asset}")
            order = await execute_binance_order(
                symbol=symbol,
                side='SELL',
                order_type='MARKET',
                quantity=float(formatted_quantity)
            )
            
            if not order or 'fills' not in order or not order['fills']:
                raise HTTPException(status_code=500, detail="Invalid order response from Binance")
            
            # Get actual sell price from order
            exit_price = float(order['fills'][0]['price'])
            
            # Update trade data
            trade["current_price"] = exit_price
            trade["status"] = "Closed"
            trade["exit_time"] = format_datetime(datetime.now().isoformat())
            trade["exit_price"] = exit_price
            trade["exit_quantity"] = float(formatted_quantity)
            
            # Calculate final metrics
            metrics = calculate_trade_metrics(trade, exit_price)
            trade.update(metrics)
            
            # Move to trade history
            trade_history.append(trade)
            
            # Remove from active trades
            active_trades.remove(trade)
            
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

@app.get("/account-balance")
async def get_account_balance():
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
