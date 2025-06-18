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
    console.log('Page loaded, fetching initial balance...');
    fetchAccountBalance();
});

// Global variables
let ws = null;
let isConnected = false;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 3000; // 3 seconds
const activeTradesMap = new Map(); // Keep track of active trades

// Global variables for Stop Loss elements
let enableStopLossCheckbox;
let stopLossInput;

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
        row.querySelector('.current-price').textContent = `$${parseFloat(trade.current_price).toFixed(2)}`;
        
        // Update P/L
        const plCell = row.querySelector('.profit-loss');
        plCell.textContent = `$${parseFloat(trade.profit_loss).toFixed(2)}`;
        plCell.className = `profit-loss ${parseFloat(trade.profit_loss) >= 0 ? 'positive' : 'negative'}`;
        
        // Update fees
        row.querySelector('.fees').textContent = parseFloat(trade.fees).toFixed(2);
        
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
            <td class="current-price">$${parseFloat(trade.current_price).toFixed(2)}</td>
            <td class="profit-loss ${parseFloat(trade.profit_loss) >= 0 ? 'positive' : 'negative'}">
                $${parseFloat(trade.profit_loss).toFixed(2)}
            </td>
            <td class="fees">$${parseFloat(trade.fees).toFixed(2)}</td>
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
                        priceCell.textContent = `$${parseFloat(trade.current_price).toFixed(2)}`;
                    }
                });
                
                // Update total P/L and counts
                updateActiveTradesCount();
                updateTotalPL();
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
    
    const tradeData = {
        coin: formData.get('coin'),
        amount: parseFloat(formData.get('amount')),
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
        // Add "No active trades" message
        const noTradesRow = document.createElement('tr');
        noTradesRow.innerHTML = `
            <td colspan="11" style="text-align: center; padding: 20px; color: #666; font-style: italic;">
                No active trades to display
            </td>
        `;
        tbody.appendChild(noTradesRow);
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

    row.innerHTML = `
        <td class="coin">${trade.coin}</td>
        <td class="amount">$${formatNumber(trade.amount_usdt)} USDT</td>
        <td class="entry-price">${trade.status === "Pending" ? 'N/A' : `$${formatNumber(trade.entry_price)}`}</td>
        <td class="current-price">$${formatNumber(trade.current_price)}</td>
        <td class="profit-loss ${parseFloat(trade.profit_loss || 0) >= 0 ? 'positive' : 'negative'}">
            $${formatNumber(trade.profit_loss)}
        </td>
        <td class="fees">$${formatNumber(trade.fees)}</td>
        <td class="take-profit">${takeProfitDisplay}</td>
        <td class="stop-loss">${stopLossDisplay}</td>
        <td class="roi ${parseFloat(trade.roi || 0) > 0 ? 'profit' : 'loss'}">${formatNumber(trade.roi)}%</td>
        <td class="status">${trade.status || 'Unknown'}</td>
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
    submitButton.textContent = 'Checking RSI...';
    
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
        submitButton.textContent = 'Auto Buy (RSI)';
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