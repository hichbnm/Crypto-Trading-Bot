let logicLogs = [];

function updateLogicLogArea() {
    const area = document.getElementById('logic-log-area');
    if (area) {
        // Remove consecutive duplicates
        let deduped = [];
        for (let i = 0; i < logicLogs.length; i++) {
            if (i === 0 || logicLogs[i] !== logicLogs[i - 1]) {
                deduped.push(logicLogs[i]);
            }
        }
        area.innerHTML = deduped.length
            ? deduped.map(msg => {
                let tag = '';
                let rest = msg;
                if (msg.startsWith('[BUY LOGIC]')) {
                    tag = '<span class="log-tag buy-log">[BUY LOGIC]</span>';
                    rest = msg.replace('[BUY LOGIC]', '');
                } else if (msg.startsWith('[SELL LOGIC]')) {
                    tag = '<span class="log-tag sell-log">[SELL LOGIC]</span>';
                    rest = msg.replace('[SELL LOGIC]', '');
                }
                return `<div class="log-line">${tag}${rest}</div>`;
            }).join('')
            : '<div style="color:#aaa;">No logs yet.</div>';
        area.scrollTop = area.scrollHeight;
    }
}

async function fetchInitialLogs() {
    try {
        const response = await fetch('/logs/trading_bot.log');
        if (!response.ok) return;
        const text = await response.text();
        // Split by lines, filter for [SELL LOGIC] or [BUY LOGIC]
        const lines = text.trim().split('\n');
        const filtered = lines.filter(line => line.includes('[SELL LOGIC]') || line.includes('[BUY LOGIC]'));
        const lastLines = filtered.slice(-50); // Show last 50 filtered logs
        logicLogs = lastLines;
        updateLogicLogArea();
    } catch (e) {
        // Optionally handle error
    }
}

document.addEventListener('DOMContentLoaded', function() {
    fetchInitialLogs();
    const toggleLogicLogBtn = document.getElementById('toggleLogicLogBtn');
    const logicLogArea = document.getElementById('logic-log-area');
    if (toggleLogicLogBtn && logicLogArea) {
        toggleLogicLogBtn.addEventListener('click', function() {
            if (logicLogArea.style.display === 'none' || logicLogArea.style.maxHeight === '0px' || logicLogArea.style.maxHeight === '0') {
                logicLogArea.style.display = 'block';
                setTimeout(() => {
                    logicLogArea.style.maxHeight = '200px';
                    logicLogArea.style.opacity = 1;
                }, 10);
                toggleLogicLogBtn.textContent = 'Hide Logic Logs';
                updateLogicLogArea();
            } else {
                logicLogArea.style.maxHeight = '0';
                logicLogArea.style.opacity = 0;
                setTimeout(() => {
                    logicLogArea.style.display = 'none';
                }, 400);
                toggleLogicLogBtn.textContent = 'Show Logic Logs';
            }
        });
        // Initial state
        logicLogArea.style.maxHeight = '0';
        logicLogArea.style.opacity = 0;
        logicLogArea.style.display = 'none';
    }

    // WebSocket handler for real logs
    // The connectWebSocket function now handles its own WebSocket connection
    // This listener is kept to ensure logic_log messages are processed
    // and the log area is updated.
    // The actual WebSocket connection is managed by connectWebSocket.
});

// Account Balance Functions
async function fetchAccountBalance() {
    try {
        console.log('Fetching account balance...');
        const response = await fetch('/account-balance');
        const data = await response.json();
        console.log('Balance data received:', data);
        
        if (data.status === 'success') {
            updateAccountBalance(data.balances);
        } else {
            console.error('Error in balance response:', data);
            document.getElementById('account-balance').textContent = 'Error';
        }
    } catch (error) {
        console.error('Error fetching account balance:', error);
        document.getElementById('account-balance').textContent = 'Error';
    }
}

function updateAccountBalance(balances) {
    console.log('Updating balance display with:', balances);
    const balanceElement = document.getElementById('account-balance');
    
    if (!balanceElement) {
        console.error('Balance element not found in DOM');
        return;
    }
    
    if (!balances || balances.length === 0) {
        console.log('No balances found, setting to 0.00');
        balanceElement.textContent = '0.00';
        return;
    }
    
    // Find USDT balance
    const usdtBalance = balances.find(balance => balance.asset === 'USDT');
    console.log('USDT balance found:', usdtBalance);
    
    if (!usdtBalance) {
        console.log('No USDT balance found, setting to 0.00');
        balanceElement.textContent = '0.00';
        return;
    }
    
    // Format USDT balance with 2 decimal places
    const total = parseFloat(usdtBalance.total);
    console.log('Setting balance to:', total.toFixed(2));
    balanceElement.textContent = total.toFixed(2);
}

// Update balance every 30 seconds
setInterval(fetchAccountBalance, 30000);

// Fetch balance immediately when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded');
    console.log('Page loaded, fetching initial balance...');
    fetchAccountBalance();
    fetchAccountTotalValue();

    // Main WebSocket for both trade updates and logic logs
    // The connectWebSocket function now handles its own WebSocket connection
    // This listener is kept to ensure trade_update and logic_log messages are processed
    // and the table/log area are updated.
    // The actual WebSocket connection is managed by connectWebSocket.
});

async function fetchAccountTotalValue() {
    try {
        const response = await fetch('/account-total-value');
        const data = await response.json();
        if (data.status === 'success') {
            const btcElem = document.getElementById('btc-balance');
            const totalElem = document.getElementById('total-account-value');
            if (btcElem) btcElem.textContent = parseFloat(data.btc_balance).toFixed(8);
            if (totalElem) totalElem.textContent = '$' + parseFloat(data.total_value).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
    } catch (e) {
        // Optionally handle error
    }
}

// Update both balances every 30 seconds
setInterval(fetchAccountTotalValue, 30000);
document.addEventListener('DOMContentLoaded', () => {
    // ... existing code ...
    fetchAccountTotalValue();
});

// Global variables
let ws = null;
let isConnected = false;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 3000; // 3 seconds
let activeTradesMap = new Map(); // Keep track of active trades

// Global variables for Stop Loss elements
let enableStopLossCheckbox;
let stopLossInput;

// Symbol mapping for chart functionality
const SYMBOL_MAPPING = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "ripple": "XRPUSDT",
    "dogecoin": "DOGEUSDT",
    "polygon": "MATICUSDT",
    "chainlink": "LINKUSDT"
};

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const activeTradesCount = document.getElementById('active-trades-count');
const totalPL = document.getElementById('total-pl');
const activePL = document.getElementById('active-pl');
const historyPL = document.getElementById('history-pl');
const tradeForm = document.getElementById('tradeForm'); // Corrected ID from 'trade-form'
const tradeTable = document.querySelector('table tbody');

// Utility Functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleString();
}

function addAnimation(element, className) {
    element.classList.add(className);
    setTimeout(() => element.classList.remove(className), 1000);
}

function calculatePL(rows) {
    let total = 0;
    rows.forEach(row => {
        const plCell = row.querySelector('.profit-loss');
        if (plCell) {
            const plValue = parseFloat(plCell.textContent.replace(/[^0-9.-]+/g, ''));
            if (!isNaN(plValue)) {
                total += plValue;
            }
        }
    });
    return total;
}

function updateTotalPL() {
    const totalPL = document.getElementById('total-pl');
    const activePL = document.getElementById('active-pl');
    const historyPL = document.getElementById('history-pl');
    
    if (!totalPL || !activePL || !historyPL) return;
    
    let activeTotal = 0;
    let historyTotal = 0;
    
    // Calculate active trades P/L
    const activeRows = document.querySelectorAll('#active-trades tr');
    activeRows.forEach(row => {
        const plCell = row.querySelector('.profit-loss');
        if (plCell) {
            activeTotal += parseFloat(plCell.textContent.replace(/[^0-9.-]+/g, '')) || 0;
        }
    });
    
    // Calculate history P/L
    const historyRows = document.querySelectorAll('#historyTableBody tr');
    historyRows.forEach(row => {
        const plCell = row.querySelector('.profit-loss');
        if (plCell) {
            historyTotal += parseFloat(plCell.textContent.replace(/[^0-9.-]+/g, '')) || 0;
        }
    });
    
    const total = activeTotal + historyTotal;
    
    // Animate P/L updates
    const oldTotal = parseFloat(totalPL.textContent.replace('$', ''));
    const oldActive = parseFloat(activePL.textContent.split(': $')[1]);
    const oldHistory = parseFloat(historyPL.textContent.split(': $')[1]);
    
    // Update displays with animations
    animateValue(totalPL, oldTotal, total, '$');
    animateValue(activePL, oldActive, activeTotal, 'Active: $');
    animateValue(historyPL, oldHistory, historyTotal, 'History: $');
    
    totalPL.className = `stat-value total-pl ${total >= 0 ? 'positive' : 'negative'}`;
}

function updateActiveTradesCount() {
    const tradesTableBody = document.getElementById('tradesTableBody');
    if (!tradesTableBody) return;

    const rows = tradesTableBody.querySelectorAll('tr[data-trade-id]');
    const count = rows.length;
    
    activeTradesCount.textContent = count;
    
    // If no active trades, display the message
    if (count === 0) {
        // Check if the 'No active trades' message is already present to avoid duplicates
        const existingNoTradesRow = tradesTableBody.querySelector('tr td[colspan="9"]');
        if (!existingNoTradesRow) {
            const noTradesRow = document.createElement('tr');
            noTradesRow.innerHTML = '<td colspan="9" style="text-align: center; padding: 20px; color: #777;">No active trades to display.</td>';
            tradesTableBody.appendChild(noTradesRow);
        }
    } else {
        // Ensure the 'No active trades' message is removed if trades exist
        const noTradesRow = tradesTableBody.querySelector('tr td[colspan="9"]');
        if (noTradesRow) {
            noTradesRow.parentElement.remove();
        }
    }
    
    // Update total P/L when count changes
    updateTotalPL();
}

// Add smooth scrolling for table rows
function scrollToRow(row) {
    row.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Add price change animation
function animatePriceChange(element, newValue, oldValue) {
    if (newValue > oldValue) {
        element.classList.add('price-up');
    } else if (newValue < oldValue) {
        element.classList.add('price-down');
    }
    
    setTimeout(() => {
        element.classList.remove('price-up', 'price-down');
    }, 1000);
}

// Add automatic scroll to top
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Enhanced updateTradeData function
function updateTradeData(trade) {
    const row = document.querySelector(`#active-trades tr[data-trade-id="${trade.id}"]`);
    if (row) {
        // Update existing row
        row.querySelector('.coin').textContent = trade.coin;
        row.querySelector('.amount').textContent = `${parseFloat(trade.amount_usdt).toFixed(2)} USDT`;
        row.querySelector('.entry-price').textContent = parseFloat(trade.entry_price).toFixed(2);
        row.querySelector('.current-price').textContent = trade.current_price != null ? `$${Number(trade.current_price).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : 'N/A';
        
        // Update P/L
        const plCell = row.querySelector('.profit-loss');
        plCell.textContent = `$${parseFloat(trade.profit_loss).toFixed(2)}`;
        plCell.className = `profit-loss ${parseFloat(trade.profit_loss) >= 0 ? 'positive' : 'negative'}`;
        
        // Update fees
        row.querySelector('.fees').textContent = parseFloat(trade.buy_fee ?? trade.fees).toFixed(3);
        
        row.querySelector('.take-profit').textContent = `${parseFloat(trade.take_profit).toFixed(1)}%`;
        row.querySelector('.stop-loss').textContent = `${parseFloat(trade.stop_loss).toFixed(1)}%`;
        
        // Update ROI
        const roiCell = row.querySelector('.roi');
        roiCell.textContent = `${parseFloat(trade.roi).toFixed(2)}%`;
        roiCell.className = `roi ${parseFloat(trade.roi) > 0 ? 'profit' : 'loss'}`;
        
        row.querySelector('.status').textContent = trade.status;
        
        // Update action button
        const actionCell = row.querySelector('.action');
        if (trade.status === "Open") {
            actionCell.innerHTML = `<button class="close-btn" onclick="closeTrade('${trade.id}')">Close</button>`;
        } else if (trade.status === "Pending") {
            actionCell.innerHTML = `<button class="decline-btn" onclick="declineTrade('${trade.id}')">Decline</button>`;
        } else {
            actionCell.innerHTML = '';
        }

        // Update take profit display
        const takeProfitCell = row.querySelector('.take-profit');
        if (takeProfitCell) {
            if (trade.take_profit_type === 'dollar') {
                takeProfitCell.textContent = `$${parseFloat(trade.take_profit).toFixed(2)}`;
            } else {
                takeProfitCell.textContent = `${parseFloat(trade.take_profit).toFixed(2)}%`;
            }
        }
    } else {
        // Add new row only if it doesn't exist
        const newRow = document.createElement('tr');
        newRow.setAttribute('data-trade-id', trade.id);
        newRow.innerHTML = `
            <td class="coin">${trade.coin}</td>
            <td class="amount">$${parseFloat(trade.amount_usdt).toFixed(2)} USDT</td>
            <td class="entry-price">${trade.status === "Pending" ? 'N/A' : `$${parseFloat(trade.entry_price).toFixed(2)}`}</td>
            <td class="current-price">$${trade.current_price != null ? Number(trade.current_price).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}) : 'N/A'}</td>
            <td class="profit-loss ${parseFloat(trade.profit_loss) >= 0 ? 'positive' : 'negative'}">
                $${parseFloat(trade.profit_loss).toFixed(2)}
            </td>
            <td class="fees">$${parseFloat(trade.buy_fee ?? trade.fees).toFixed(3)}</td>
            <td class="take-profit">${parseFloat(trade.take_profit).toFixed(1)}%</td>
            <td class="stop-loss">${parseFloat(trade.stop_loss).toFixed(1)}%</td>
            <td class="roi ${parseFloat(trade.roi) > 0 ? 'profit' : 'loss'}">${parseFloat(trade.roi).toFixed(2)}%</td>
            <td class="status">${trade.status}</td>
            <td class="action">
                ${trade.status === "Open" ? `<button class="close-btn" onclick="closeTrade('${trade.id}')">Close</button>` : 
                  trade.status === "Pending" ? `<button class="decline-btn" onclick="declineTrade('${trade.id}')">Decline</button>` : ''}
            </td>
        `;
        document.getElementById('active-trades').appendChild(newRow);
    }
}

// WebSocket Functions
function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    const statusText = statusElement.querySelector('.status-text');
    
    if (connected) {
        statusElement.className = 'connection-status connected';
        statusText.textContent = 'Connected';
        reconnectAttempts = 0;
    } else {
        statusElement.className = 'connection-status disconnected';
        statusText.textContent = 'Disconnected';
    }
}

function connectWebSocket() {
    if (ws) {
        ws.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    
    ws = new WebSocket(wsUrl);

    ws.onopen = function() {
        console.log('WebSocket connected');
        isConnected = true;
        updateConnectionStatus(true);
    };

    ws.onclose = function(event) {
        console.log('WebSocket disconnected:', event.code, event.reason);
        isConnected = false;
        updateConnectionStatus(false);
        
        if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})...`);
            setTimeout(connectWebSocket, reconnectDelay);
        }
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        isConnected = false;
        updateConnectionStatus(false);
    };

    ws.onmessage = function(event) {
        console.log('[DEBUG] WebSocket message received:', event.data);
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'trade_update' && Array.isArray(data.trades)) {
                // Update the table with new trade data
                updateActiveTradesTable(data.trades);
                
                // Update P/L and ROI for each trade
                data.trades.forEach(trade => {
                    const row = document.querySelector(`#active-trades tr[data-trade-id="${trade.id}"]`);
                    if (row) {
                        // Update P/L
                        const plCell = row.querySelector('.profit-loss');
                        plCell.textContent = `$${parseFloat(trade.profit_loss).toFixed(2)}`;
                        plCell.className = `profit-loss ${parseFloat(trade.profit_loss) >= 0 ? 'positive' : 'negative'}`;
                        
                        // Update ROI
                        const roiCell = row.querySelector('.roi');
                        roiCell.textContent = `${parseFloat(trade.roi).toFixed(2)}%`;
                        roiCell.className = `roi ${parseFloat(trade.roi) > 0 ? 'profit' : 'loss'}`;
                        
                        // Update current price
                        const priceCell = row.querySelector('.current-price');
                        priceCell.textContent = trade.current_price != null ? `$${Number(trade.current_price).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : 'N/A';
                    }
                });
                
                // Update total P/L and counts
                updateActiveTradesCount();
                updateTotalPL();
            } else if (data.type === 'logic_log') {
                console.log('Received logic log:', data.message);
                if (data.message.includes('[SELL LOGIC]') || data.message.includes('[BUY LOGIC]')) {
                    logicLogs.push(data.message);
                    if (logicLogs.length > 100) logicLogs.shift();
                    updateLogicLogArea();
                }
            }
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    };
}

// Trade Form Functions
// Add value animation function
function animateValue(element, oldValue, newValue, prefix = '') {
    if (oldValue === newValue) return;
    
    const duration = 1000; // 1 second
    const startTime = performance.now();
    const startValue = oldValue;
    const endValue = newValue;
    
    function updateValue(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeProgress = progress < 0.5
            ? 4 * progress * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 3) / 2;
        
        const currentValue = startValue + (endValue - startValue) * easeProgress;
        element.textContent = `${prefix}${currentValue.toFixed(2)}`;
        
        if (progress < 1) {
            requestAnimationFrame(updateValue);
        }
    }
    
    requestAnimationFrame(updateValue);
}

// Enhanced form handling
function handleTradeSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitButton = form.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    
    // Show loading state
    submitButton.disabled = true;
    submitButton.textContent = 'Processing...';
    
    const formData = new FormData(form);
    
    let amountRaw = formData.get('amount');
    if (typeof amountRaw === 'string') {
        amountRaw = amountRaw.replace(',', '.');
    }
    const tradeData = {
        coin: formData.get('coin'),
        amount: parseFloat(amountRaw),
        order_type: formData.get('order_type'),
        limit_price: formData.get('order_type') === 'limit' ? parseFloat(formData.get('limit_price')) : null,
        take_profit: parseFloat(formData.get('take_profit')),
        take_profit_type: formData.get('take_profit_type'),
        stop_loss: enableStopLossCheckbox.checked ? parseFloat(formData.get('stop_loss')) : null // Send null if disabled
    };
    
    // Remove any existing error messages
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.innerHTML = '';
    }
    
    fetch('/trade', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(tradeData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Show success state with actual amount used
            const actualAmount = data.trade.amount_usdt;
            const originalAmount = tradeData.amount;
            const difference = originalAmount - actualAmount;
            
            submitButton.textContent = 'Success!';
            submitButton.style.backgroundColor = 'var(--success-color)';
            
            // Show info about the actual amount used
            if (errorContainer) {
                let message = `
                    <div class="info-message">
                        Trade executed successfully!<br>
                        Requested amount: $${originalAmount.toFixed(2)}<br>
                        Actual amount used: $${actualAmount.toFixed(2)}<br>
                        Difference: $${difference.toFixed(2)} (due to fees and minimum quantity requirements)
                    </div>
                `;
                
                // Add warning if difference is significant
                if (Math.abs(difference) > 1.0) {
                    message = `
                        <div class="warning-message">
                            <strong>Note:</strong> The actual amount used is different from your requested amount due to minimum quantity requirements.<br>
                            ${message}
                        </div>
                    `;
                }
                
                errorContainer.innerHTML = message;
            }
            
            // Reset form
            form.reset();
            document.getElementById('limit_price_container').style.display = 'none';
            // Reset stop loss toggle and input
            if (enableStopLossCheckbox && stopLossInput) { // Ensure elements exist before trying to reset
                enableStopLossCheckbox.checked = false;  // Keep it unchecked by default
                stopLossInput.disabled = true;  // Keep it disabled by default
                stopLossInput.value = '';  // Clear the value
            }
            
            // Reset button after delay
            setTimeout(() => {
                submitButton.disabled = false;
                submitButton.textContent = originalText;
                submitButton.style.backgroundColor = '';
                if (errorContainer) {
                    errorContainer.innerHTML = '';
                }
            }, 5000); // Show the message for 5 seconds
        } else {
            // Show error state
            submitButton.textContent = 'Error';
            submitButton.style.backgroundColor = 'var(--danger-color)';
            
            // Show error message
            if (errorContainer) {
                errorContainer.innerHTML = `<div class="error-message">${data.message || 'Failed to create trade'}</div>`;
            }
            
            // Reset button after delay
            setTimeout(() => {
                submitButton.disabled = false;
                submitButton.textContent = originalText;
                submitButton.style.backgroundColor = '';
            }, 2000);
        }
    })
    .catch(error => {
        console.error('Error creating trade:', error);
        
        // Show error state
        submitButton.textContent = 'Error';
        submitButton.style.backgroundColor = 'var(--danger-color)';
        
        // Show error message
        if (errorContainer) {
            errorContainer.innerHTML = `<div class="error-message">${error.message || 'Failed to create trade'}</div>`;
        }
        
        // Reset button after delay
        setTimeout(() => {
            submitButton.disabled = false;
            submitButton.textContent = originalText;
            submitButton.style.backgroundColor = '';
        }, 2000);
    });
}

async function declineTrade(tradeId) {
    try {
        const response = await fetch(`/decline-trade/${tradeId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to decline trade');
        }

        // Remove the trade row from the table
        const tradeRow = document.querySelector(`tr[data-trade-id="${tradeId}"]`);
        if (tradeRow) {
            tradeRow.remove();
            updateActiveTradesCount();
            updateTotalPL(); // Update total P/L when trade is declined
        }
    } catch (error) {
        console.error('Error declining trade:', error);
        alert(error.message);
    }

}

// Function to validate trade data
function isValidTrade(trade) {
    // Basic validation - only check for required fields
    return trade && 
           trade.id && 
           trade.coin && 
           trade.status &&
           trade.amount_usdt !== undefined &&  // Make sure amount exists
           trade.entry_price !== undefined;    // Make sure entry price exists
    // Note: stop_loss is not required
}

// Function to update the active trades table
function updateActiveTradesTable(trades) {
    const tbody = document.getElementById('active-trades');
    if (!tbody) return;
    
    // Clear existing rows
    tbody.innerHTML = '';
    
    // Update the map and table
    activeTradesMap.clear();
    
    if (!trades || trades.length === 0) {
        // No active trades, show centered message
        const row = document.createElement('tr');
        row.innerHTML = `<td colspan="14" style="text-align: center; padding: 20px; color: #777;">No active trades to display.</td>`;
        tbody.appendChild(row);
        document.getElementById('active-trades-count').textContent = '0';
        // Also update header P/L values to 0
        const totalPLSpan = document.getElementById('total-pl');
        const activePLSpan = document.getElementById('active-pl');
        if (totalPLSpan) totalPLSpan.textContent = '$0.00';
        if (activePLSpan) activePLSpan.textContent = 'Active: $0.00';
        return;
    } else {
        trades.forEach(trade => {
            // Store in map regardless of validation
            activeTradesMap.set(trade.id, trade);
            // Add to table if it passes validation
            if (isValidTrade(trade)) {
                addTradeToTable(trade);
                console.log("Adding trade to table:", trade);
            }
        });
    }
    
    updateActiveTradesCount();
    updateTotalPL();
}

// Function to add a single trade to the table
function addTradeToTable(trade) {
    console.log("Adding trade to table:", trade);
    if (!isValidTrade(trade)) return;

    const tbody = document.getElementById('active-trades');
    if (!tbody) return;

    // Remove "No active trades" message if it exists
    const noTradesRow = tbody.querySelector('tr td[colspan="11"]');
    if (noTradesRow) {
        noTradesRow.parentElement.remove();
    }

    const row = document.createElement('tr');
    row.setAttribute('data-trade-id', trade.id);
    
    // Format numbers with commas and 2 decimal places
    const formatNumber = (num) => {
        if (num === null || num === undefined) return '0.00';
        const parsed = parseFloat(num);
        return isNaN(parsed) ? '0.00' : parsed.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    };

    // Format take profit display
    let takeProfitDisplay = '';
    if (trade.take_profit_type === 'dollar') {
        takeProfitDisplay = `$${formatNumber(trade.take_profit)}`;
    } else {
        takeProfitDisplay = `${formatNumber(trade.take_profit)}%`;
    }

    // Format stop loss display - always show N/A if stop_loss is null or undefined
    const stopLossDisplay = (trade.stop_loss === null || trade.stop_loss === undefined) 
        ? 'N/A' 
        : `${formatNumber(trade.stop_loss)}%`;

    // For the Entry Time cell, show 'N/A' if the trade is not filled (status not 'Open') or entry_time is missing
    const entryTimeDisplay = (trade.status === 'Open' && trade.entry_time) ? trade.entry_time : 'N/A';
    row.innerHTML = `
        <td class="buy-order-id">${trade.binance_order_id ? trade.binance_order_id : 'N/A'}</td>
        <td class="coin">${trade.coin}</td>
        <td class="amount">$${formatNumber(trade.amount_usdt)}</td>
        <td class="filled-amount">${trade.status === 'Pending' ? 'N/A' : `$${typeof trade.filled_amount_usdt !== 'undefined' ? formatNumber(trade.filled_amount_usdt) : formatNumber(trade.amount_usdt)}`}</td>
        <td class="entry-price">${trade.status === "Pending" ? 'N/A' : `$${formatNumber(trade.entry_price)}`}</td>
        <td class="current-price">$${trade.current_price != null ? Number(trade.current_price).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}) : 'N/A'}</td>
        <td class="profit-loss ${parseFloat(trade.profit_loss || 0) >= 0 ? 'positive' : 'negative'}">
            $${formatNumber(trade.profit_loss)}
        </td>
        <td class="fees">$${parseFloat(trade.buy_fee ?? trade.fees).toFixed(3)}</td>
        <td class="take-profit">${takeProfitDisplay}</td>
        <td class="stop-loss">${stopLossDisplay}</td>
        <td class="roi ${parseFloat(trade.roi || 0) > 0 ? 'profit' : 'loss'}">${formatNumber(trade.roi)}%</td>
        <td class="status">${trade.status || 'Unknown'}</td>
        <td class="entry-time">${entryTimeDisplay}</td>
        <td class="action">
            ${trade.status === "Open" ? 
                `<button class="close-btn" onclick="closeTrade('${trade.id}')">Close</button>` : 
                trade.status === "Pending" ? 
                `<button class="decline-btn" onclick="declineTrade('${trade.id}')">Decline</button>` : 
                ''}
        </td>
    `;

    tbody.appendChild(row);
}

function closeTrade(tradeId) {
    const row = document.querySelector(`tr[data-trade-id="${tradeId}"]`);
    if (row) {
        const closeButton = row.querySelector('.close-btn');
        if (closeButton) {
            closeButton.disabled = true;
            closeButton.textContent = 'Closing...';
        }
    }

    fetch(`/close-trade/${tradeId}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            const row = document.querySelector(`tr[data-trade-id="${tradeId}"]`);
            if (row) {
                row.classList.add('fadeOut');
                setTimeout(() => {
                    row.remove();
                    updateActiveTradesCount();
                    updateTotalPL();
                    
                    // Show success message with details
                    const message = `
                        Trade closed successfully!
                        Sold: ${data.trade.exit_quantity.toFixed(8)} ${data.trade.coin}
                        Price: $${data.trade.exit_price.toFixed(2)}
                        Total: $${(data.trade.exit_quantity * data.trade.exit_price).toFixed(2)}
                        P/L: $${data.trade.profit_loss.toFixed(2)}
                    `;
                    alert(message);
                }, 500);
            }
        } else {
            alert('Error closing trade: ' + data.message);
            // Reset button if there was an error
            const row = document.querySelector(`tr[data-trade-id="${tradeId}"]`);
            if (row) {
                const closeButton = row.querySelector('.close-btn');
                if (closeButton) {
                    closeButton.disabled = false;
                    closeButton.textContent = 'Close';
                }
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error closing trade. Please try again.');
        // Reset button if there was an error
        const row = document.querySelector(`tr[data-trade-id="${tradeId}"]`);
        if (row) {
            const closeButton = row.querySelector('.close-btn');
            if (closeButton) {
                closeButton.disabled = false;
                closeButton.textContent = 'Close';
            }
        }
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize WebSocket connection
    connectWebSocket();
    
    // Get Stop Loss elements and set initial state
    enableStopLossCheckbox = document.getElementById('enable_stop_loss');
    stopLossInput = document.getElementById('stop_loss');

    if (enableStopLossCheckbox && stopLossInput) {
        // Set initial state
        stopLossInput.disabled = !enableStopLossCheckbox.checked;

        enableStopLossCheckbox.addEventListener('change', function() {
            stopLossInput.disabled = !this.checked;
            if (!this.checked) {
                stopLossInput.value = ''; // Clear value when disabled
            }
        });
    }

    // Add form submit handler
    const tradeFormElement = document.getElementById('tradeForm'); // Use a new const for local scope
    tradeFormElement.addEventListener('submit', handleTradeSubmit);
    
    // Add order type change handler
    const orderTypeSelect = document.getElementById('order_type');
    const limitPriceContainer = document.getElementById('limit_price_container');
    const limitPriceInput = document.getElementById('limit_price');
    
    if (orderTypeSelect && limitPriceContainer) {
        orderTypeSelect.addEventListener('change', function() {
            if (this.value === 'limit') {
                limitPriceContainer.style.display = 'block';
                limitPriceInput.required = true;
                limitPriceContainer.style.opacity = '0';
                limitPriceContainer.style.transform = 'translateY(-10px)';
                setTimeout(() => {
                    limitPriceContainer.style.opacity = '1';
                    limitPriceContainer.style.transform = 'translateY(0)';
                }, 50);
            } else {
                limitPriceContainer.style.opacity = '0';
                limitPriceContainer.style.transform = 'translateY(-10px)';
                setTimeout(() => {
                    limitPriceContainer.style.display = 'none';
                    limitPriceInput.required = false;
                }, 300);
            }
        });
    }
    
    // Remove amount input restrictions
    const amountInput = document.getElementById('amount');
    if (amountInput) {
        amountInput.removeAttribute('step');
        amountInput.removeAttribute('min');
    }

    if (amountInput) {
        // Replace comma with dot as user types
        amountInput.addEventListener('input', function(e) {
            this.value = this.value.replace(',', '.');
        });
    }
    
    // Initial P/L update
    updateTotalPL();
    
    // Add visibility change handler
    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'visible' && (!ws || ws.readyState === WebSocket.CLOSED)) {
            connectWebSocket();
        }
    });
    
    // Add cleanup handler
    window.addEventListener('beforeunload', function() {
        if (ws) {
            ws.close();
        }
    });

    // Fetch active trades count from backend and update header
    fetchAndUpdateActiveTradesCount();
    setInterval(fetchAndUpdateActiveTradesCount, 10000); // update every 10s

    // Chart toggle button logic
    const toggleChartBtn = document.getElementById('toggleChartBtn');
    const tradeChartDiv = document.getElementById('tradeChart');
    if (toggleChartBtn && tradeChartDiv) {
        // Add smooth transition
        tradeChartDiv.style.transition = 'opacity 0.4s';
        // Set default to hidden if not set in localStorage
        let chartVisible = localStorage.getItem('chartVisible');
        if (chartVisible === null) {
            chartVisible = 'false';
            localStorage.setItem('chartVisible', 'false');
        }
        if (chartVisible === 'false') {
            tradeChartDiv.style.opacity = 0;
            tradeChartDiv.style.pointerEvents = 'none';
            setTimeout(() => { tradeChartDiv.style.display = 'none'; }, 400);
            toggleChartBtn.textContent = 'Show Chart';
        } else {
            tradeChartDiv.style.opacity = 1;
            tradeChartDiv.style.pointerEvents = '';
            tradeChartDiv.style.display = '';
            toggleChartBtn.textContent = 'Hide Chart';
        }
        toggleChartBtn.addEventListener('click', function() {
            if (tradeChartDiv.style.display === 'none' || tradeChartDiv.style.opacity === '0') {
                tradeChartDiv.style.display = '';
                setTimeout(() => {
                    tradeChartDiv.style.opacity = 1;
                    tradeChartDiv.style.pointerEvents = '';
                    if (window.Plotly && window.Plotly.Plots && typeof window.Plotly.Plots.resize === 'function') {
                        Plotly.Plots.resize(tradeChartDiv);
                    }
                }, 10);
                toggleChartBtn.textContent = 'Hide Chart';
                localStorage.setItem('chartVisible', 'true');
            } else {
                tradeChartDiv.style.opacity = 0;
                tradeChartDiv.style.pointerEvents = 'none';
                setTimeout(() => {
                    tradeChartDiv.style.display = 'none';
                }, 400);
                toggleChartBtn.textContent = 'Show Chart';
                localStorage.setItem('chartVisible', 'false');
            }
        });
    }

    // Chart range button group logic
    const chartRangeGroup = document.getElementById('chart-range-group');
    if (chartRangeGroup) {
        chartRangeGroup.addEventListener('click', function(e) {
            if (e.target.classList.contains('chart-range-btn')) {
                // Remove 'active' from all buttons
                document.querySelectorAll('.chart-range-btn').forEach(btn => btn.classList.remove('active'));
                // Add 'active' to clicked button
                e.target.classList.add('active');
                // Update chart
                plotTradeChart();
            }
        });
    }

    let logicLogs = [];

    function updateLogicLogArea() {
        const area = document.getElementById('logic-log-area');
        if (area) {
            area.innerHTML = logicLogs.map(msg => `<div>${msg}</div>`).join('');
            area.scrollTop = area.scrollHeight;
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        // ... existing code ...
        updateLogicLogArea(); // Show any logs already present on load
        // Logic Logs Toggle Button
        const toggleLogicLogBtn = document.getElementById('toggleLogicLogBtn');
        const logicLogArea = document.getElementById('logic-log-area');
        console.log('toggleLogicLogBtn:', toggleLogicLogBtn);
        console.log('logicLogArea:', logicLogArea);

        if (toggleLogicLogBtn && logicLogArea) {
            toggleLogicLogBtn.addEventListener('click', function() {
                console.log('Show Logic Logs button clicked!');
                if (logicLogArea.style.maxHeight === '0px' || logicLogArea.style.maxHeight === '0' || logicLogArea.style.display === 'none') {
                    logicLogArea.style.display = 'block';
                    setTimeout(() => {
                        logicLogArea.style.maxHeight = '200px';
                        logicLogArea.style.opacity = 1;
                    }, 10);
                    toggleLogicLogBtn.textContent = 'Hide Logic Logs';
                } else {
                    logicLogArea.style.maxHeight = '0';
                    logicLogArea.style.opacity = 0;
                    setTimeout(() => {
                        logicLogArea.style.display = 'none';
                    }, 400);
                    toggleLogicLogBtn.textContent = 'Show Logic Logs';
                }
            });
            // Ensure initial state is hidden
            logicLogArea.style.maxHeight = '0';
            logicLogArea.style.opacity = 0;
            logicLogArea.style.display = 'none';
        }
    });
});

// Add this function after the existing functions
async function placeAutoBuyOrder() {
    const form = document.getElementById('tradeForm');
    const submitButton = document.getElementById('autoBuyBtn');
    const errorContainer = document.getElementById('error-container');
    
    if (!form) {
        console.error('Trade form not found');
        return;
    }
    
    // Show loading state
    submitButton.disabled = true;
    submitButton.textContent = 'Checking Low Turn Point...';
    
    try {
        // Get form data
        const formData = new FormData(form);
        const tradeData = {
            coin: formData.get('coin'),
            amount: parseFloat(formData.get('amount')),
            order_type: formData.get('order_type'),
            limit_price: formData.get('order_type') === 'limit' ? parseFloat(formData.get('limit_price')) : null,
            take_profit: parseFloat(formData.get('take_profit')),
            take_profit_type: formData.get('take_profit_type'),
            stop_loss: enableStopLossCheckbox.checked ? parseFloat(formData.get('stop_loss')) : null // This is correct
        };
        
        // Validate amount
        if (isNaN(tradeData.amount) || tradeData.amount <= 0) {
            throw new Error('Please enter a valid amount greater than 0');
        }
        
        // Clear previous error messages
        if (errorContainer) {
            errorContainer.innerHTML = '';
        }
        
        // Send request to auto-buy endpoint
        const response = await fetch('/auto-buy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(tradeData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Auto-buy failed. Please try again.');
        }
        
        // Show success message
        if (errorContainer) {
            const successMessage = document.createElement('div');
            successMessage.className = 'success-message';
            successMessage.textContent = data.message || 'Trade created successfully!';
            errorContainer.appendChild(successMessage);
            
            // Remove success message after 3 seconds
            setTimeout(() => {
                successMessage.remove();
            }, 3000);
        }
        
    } catch (error) {
        console.error('Error in auto-buy:', error);
        if (errorContainer) {
            const errorMessage = document.createElement('div');
            errorMessage.className = 'error-message';
            errorMessage.textContent = error.message;
            errorContainer.appendChild(errorMessage);
        }
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.textContent = 'Auto Buy';
    }
}

// Fetch active trades count from backend and update header
async function fetchAndUpdateActiveTradesCount() {
    try {
        const response = await fetch('/active-trades');
        if (!response.ok) return;
        const trades = await response.json();
        // Only count trades with status 'Open' or 'Pending'
        const actionableCount = trades.filter(trade => trade.status === 'Open' || trade.status === 'Pending').length;
        const countElem = document.getElementById('active-trades-count');
        if (countElem) countElem.textContent = actionableCount;
    } catch (e) {
        // Optionally log error
    }
}

async function fetchActiveTrades() {
    const response = await fetch('/active-trades');
    const data = await response.json();
    return Array.isArray(data) ? data : [];
}

async function fetchTradeHistory() {
    const response = await fetch('/trade-history');
    const data = await response.json();
    return Array.isArray(data) ? data : [];
}

async function fetchPriceHistory(symbol, range) {
    try {
        console.log(`Fetching price history for ${symbol} with range ${range}...`);
        const response = await fetch(`/price-history?symbol=${symbol}&range=${range}`);
        console.log(`Response status for ${symbol}:`, response.status);
        
        if (!response.ok) {
            console.error(`Error fetching price history for ${symbol}:`, response.status, response.statusText);
            return null;
        }
        
        const data = await response.json();
        console.log(`Received price data for ${symbol}:`, data ? 'success' : 'failed');
        return data;
    } catch (error) {
        console.error(`Error fetching price history for ${symbol}:`, error);
        return null;
    }
}

// Test function to verify chart is working
function testChart() {
    console.log('Testing chart functionality...');
    const testData = [
        {
            x: ['2024-01-01', '2024-01-02', '2024-01-03'],
            y: [100, 110, 105],
            mode: 'lines',
            name: 'Test BTC',
            line: { color: '#26a69a', width: 2 }
        }
    ];
    
    const layout = {
        title: 'Test Chart',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Price' }
    };
    
    try {
        Plotly.newPlot('tradeChart', testData, layout);
        console.log('Test chart created successfully!');
        return true;
    } catch (error) {
        console.error('Error creating test chart:', error);
        return false;
    }
}

async function plotTradeChart() {
    console.log('plotTradeChart function called');
    
    // Always use BTC for the chart
    const symbol = 'BTCUSDT';
    const activeBtn = document.querySelector('.chart-range-btn.active');
    const range = activeBtn ? activeBtn.getAttribute('data-range') : '1'; // Default to 1 day if not found
    console.log('Using symbol:', symbol, 'with range:', range);

    // Color for BTC
    const color = '#26a69a'; // BTC - Teal

    // Fetch data
    const [activeTrades, tradeHistory] = await Promise.all([
        fetchActiveTrades(),
        fetchTradeHistory()
    ]);

    console.log('Fetched active trades:', activeTrades.length);
    console.log('Fetched trade history:', tradeHistory.length);

    try {
        // Fetch price data for BTC
        const priceData = await fetchPriceHistory(symbol, range);
        console.log('Price data for BTC:', priceData ? 'received' : 'failed');
        
        if (!priceData || !priceData.closes || priceData.closes.length === 0) {
            console.warn('No price data available for BTC');
            return;
        }

        const closes = priceData.closes;
        const times = priceData.times;
        const volumes = priceData.volumes || [];
        
        console.log('BTC data points:', closes.length);
        
        // Create price trace for BTC
        const priceTrace = {
            x: times,
            y: closes,
            mode: 'lines',
            line: { color: color, width: 2 },
            name: 'Bitcoin (BTC)',
            hovertemplate: 'Price: $%{y:.2f}<br>Time: %{x}<extra></extra>',
            connectgaps: true
        };

        // Create volume trace for BTC
        const volumeTrace = {
            x: times,
            y: volumes,
            type: 'bar',
            yaxis: 'y2',
            marker: { color: 'rgba(38, 166, 154, 0.3)', opacity: 0.2 },
            name: 'BTC Volume',
            opacity: 0.2,
            hovertemplate: 'Vol: %{y:.0f}<br>Time: %{x}<extra></extra>',
            showlegend: false
        };

        // Add buy/sell markers for BTC
        const coinBuys = [];
        const coinSells = [];

        function findClosestTime(targetTime, times) {
            let minDiff = Infinity;
            let closest = times[0];
            for (let t of times) {
                let diff = Math.abs(new Date(t) - new Date(targetTime));
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = t;
                }
            }
            // Debug: warn if the closest time is more than 10 minutes away
            if (minDiff > 10 * 60 * 1000) {
                console.warn('Trade time is far from any candle:', targetTime, 'Closest:', closest, 'Diff (min):', minDiff / 60000);
            }
            return closest;
        }

        function addMarkersForCoin(trades, isHistory, times, closes, coinSymbol) {
            if (!Array.isArray(trades)) {
                return;
            }
            trades.forEach(trade => {
                // Only process trades for BTC
                const tradeSymbol = SYMBOL_MAPPING[trade.coin?.toLowerCase()] || trade.coin;
                if (tradeSymbol !== coinSymbol) return;

                if (isHistory && trade.status === 'Closed') {
                    if (trade.entry_time && trade.entry_price) {
                        const closestTime = findClosestTime(trade.entry_time, times);
                        const idx = times.indexOf(closestTime);
                        if (idx >= 0) {
                            coinBuys.push({
                                time: closestTime,
                                price: closes[idx]
                            });
                        }
                    }
                    if (trade.exit_time && trade.exit_price) {
                        const closestTime = findClosestTime(trade.exit_time, times);
                        const idx = times.indexOf(closestTime);
                        if (idx >= 0) {
                            coinSells.push({
                                time: closestTime,
                                price: closes[idx]
                            });
                        }
                    }
                } else {
                    if (trade.type === 'BUY' || (!isHistory && trade.status === 'Open')) {
                        const closestTime = findClosestTime(trade.entry_time || trade.time, times);
                        const idx = times.indexOf(closestTime);
                        if (idx >= 0) {
                            coinBuys.push({
                                time: closestTime,
                                price: closes[idx]
                            });
                        }
                    } else if (trade.type === 'SELL' || (isHistory && trade.status === 'Closed')) {
                        const closestTime = findClosestTime(trade.exit_time || trade.time, times);
                        const idx = times.indexOf(closestTime);
                        if (idx >= 0) {
                            coinSells.push({
                                time: closestTime,
                                price: closes[idx]
                            });
                        }
                    }
                }
            });
        }

        addMarkersForCoin(Array.isArray(activeTrades) ? activeTrades : [], false, times, closes, symbol);
        addMarkersForCoin(Array.isArray(tradeHistory) ? tradeHistory : [], true, times, closes, symbol);

        console.log('BTC buy markers:', coinBuys.length);
        console.log('BTC sell markers:', coinSells.length);

        // Create buy markers for BTC
        const buyMarkers = {
            x: coinBuys.map(t => t.time),
            y: coinBuys.map(t => t.price),
            mode: 'markers+text',
            marker: { color: 'limegreen', size: 12, symbol: 'triangle-up' },
            name: 'BTC Buy',
            text: coinBuys.map(() => 'Buy'),
            textposition: 'top center',
            hovertemplate: 'BTC Buy<br>Price: <b>$%{y:,.2f}</b><br>Time: %{x}<extra></extra>',
            showlegend: false
        };

        // Create sell markers for BTC
        const sellMarkers = {
            x: coinSells.map(t => t.time),
            y: coinSells.map(t => t.price),
            mode: 'markers+text',
            marker: { color: 'red', size: 12, symbol: 'triangle-down' },
            name: 'BTC Sell',
            text: coinSells.map(() => 'Sell'),
            textposition: 'bottom center',
            hovertemplate: 'BTC Sell<br>Price: <b>$%{y:,.2f}</b><br>Time: %{x}<extra></extra>',
            showlegend: false
        };

        // Orange points (peaks where no sell)
        const orangePoints = (priceData.orange_points || []);
        const orangeMarkers = {
            x: orangePoints.map(pt => pt.time),
            y: orangePoints.map(pt => pt.price),
            mode: 'markers',
            marker: { color: 'orange', size: 14, symbol: 'circle' },
            name: 'Peak (No Sell)',
            hovertemplate: 'Peak Price: <b>$%{y:,.2f}</b><br>Time: %{x}<extra></extra>',
            showlegend: true
        };
        // Combine all traces
        const allData = [priceTrace, buyMarkers, sellMarkers, orangeMarkers, volumeTrace];
        console.log('Total traces to plot:', allData.length);

        // Check if chart container exists
        const chartContainer = document.getElementById('tradeChart');
        if (!chartContainer) {
            console.error('Chart container not found!');
            return;
        }

        console.log('Creating BTC chart...');

        // Create the chart
        Plotly.newPlot('tradeChart', allData, {
            title: {
                text: 'Bitcoin (BTC) Price Chart',
                font: { size: 16, color: '#2c3e50' }
            },
            plot_bgcolor: '#fff',
            paper_bgcolor: '#fff',
            font: { color: '#222' },
            xaxis: {
                title: '',
                type: 'date',
                gridcolor: '#eee',
                showgrid: true,
                zeroline: false,
                showline: false,
                tickformat: '%d %b %H:%M',
                rangeslider: { visible: false },
                showspikes: true,
                spikemode: 'across',
                spikecolor: '#aaa',
                spikethickness: 1
            },
            yaxis: {
                title: 'Price (USD)',
                gridcolor: '#eee',
                showgrid: true,
                zeroline: false,
                showline: false,
                tickprefix: '$',
                showspikes: true,
                spikemode: 'across',
                spikecolor: '#aaa',
                spikethickness: 1
            },
            yaxis2: {
                title: 'Volume',
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                rangemode: 'tozero',
                fixedrange: true
            },
            dragmode: 'pan',
            showlegend: true,
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#ddd',
                borderwidth: 1
            },
            hovermode: 'x unified',
            hoverlabel: {
                font: { size: 11 }
            },
            margin: { t: 40, b: 40, l: 60, r: 60 },
            responsive: true,
            scrollZoom: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['autoScale2d', 'resetScale2d', 'lasso2d', 'select2d']
        });
        console.log('BTC chart created successfully!');
    } catch (error) {
        console.error('Error creating BTC chart:', error);
    }
}

document.addEventListener('DOMContentLoaded', plotTradeChart); 