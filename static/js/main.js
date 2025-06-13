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

// DOM Elements
const connectionStatus = document.getElementById('connection-status');
const activeTradesCount = document.getElementById('active-trades-count');
const totalPL = document.getElementById('total-pl');
const activePL = document.getElementById('active-pl');
const historyPL = document.getElementById('history-pl');
const tradeForm = document.getElementById('trade-form');
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
    const activeRows = document.querySelectorAll('#tradesTableBody tr');
    activeRows.forEach(row => {
        const plCell = row.querySelector('.profit-loss');
        if (plCell) {
            activeTotal += parseFloat(plCell.textContent) || 0;
        }
    });
    
    // Calculate history P/L
    const historyRows = document.querySelectorAll('#historyTableBody tr');
    historyRows.forEach(row => {
        const plCell = row.querySelector('.profit-loss');
        if (plCell) {
            historyTotal += parseFloat(plCell.textContent) || 0;
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
function updateTradeData(data) {
    const tradesTableBody = document.getElementById('tradesTableBody');
    if (!tradesTableBody) return;

    const existingRow = document.querySelector(`tr[data-trade-id="${data.id}"]`);
    
    if (existingRow) {
        // Update existing row with animations
        const cells = existingRow.cells;
        const oldPrice = parseFloat(cells[3].textContent);
        const newPrice = parseFloat(data.current_price);
        
        cells[0].textContent = data.coin;
        cells[1].textContent = data.amount_usdt;
        cells[2].textContent = data.entry_price;
        
        // Animate price change
        animatePriceChange(cells[3], newPrice, oldPrice);
        cells[3].textContent = data.current_price;
        
        const plCell = cells[4];
        const oldPL = parseFloat(plCell.textContent);
        const newPL = parseFloat(data.profit_loss);
        
        // Animate P/L change
        animatePriceChange(plCell, newPL, oldPL);
        plCell.textContent = data.profit_loss;
        plCell.className = `profit-loss ${data.profit_loss >= 0 ? 'positive' : 'negative'}`;
        
        cells[5].textContent = data.fees;
        cells[6].textContent = `${data.roi !== undefined && data.roi !== null ? data.roi.toFixed(2) : 'N/A'}%`;
        cells[7].textContent = data.status;
        
        // Update action button without animation
        const actionCell = cells[8];
        if (data.status === 'Open') {
            actionCell.innerHTML = `<button onclick="closeTrade('${data.id}')" class="close-btn">Close</button>`;
        } else if (data.status === 'Pending') {
            actionCell.innerHTML = `<button onclick="declineTrade('${data.id}')" class="decline-btn">Decline</button>`;
        }
    } else {
        // If there's a 'No active trades' message, remove it
        const noTradesRow = tradesTableBody.querySelector('tr td[colspan="9"]');
        if (noTradesRow) {
            noTradesRow.parentElement.remove(); // Remove the entire row
        }

        // Add new row with animation
        const row = document.createElement('tr');
        row.setAttribute('data-trade-id', data.id);
        row.style.opacity = '0';
        row.style.transform = 'translateY(20px)';
        
        row.innerHTML = `
            <td>${data.coin}</td>
            <td>${data.amount_usdt}</td>
            <td>${data.entry_price}</td>
            <td>${data.current_price}</td>
            <td class="profit-loss ${data.profit_loss >= 0 ? 'positive' : 'negative'}">${data.profit_loss}</td>
            <td>${data.fees}</td>
            <td>${(data.roi !== undefined && data.roi !== null) ? data.roi.toFixed(2) : 'N/A'}%</td>
            <td class="${data.status === 'Open' ? 'status-open' : 'status-pending'}">${data.status}</td>
            <td>
                ${data.status === 'Open' 
                    ? `<button onclick="closeTrade('${data.id}')" class="close-btn">Close</button>`
                    : data.status === 'Pending'
                    ? `<button onclick="declineTrade('${data.id}')" class="decline-btn">Decline</button>`
                    : ''}
            </td>
        `;
        
        tradesTableBody.appendChild(row);
        
        // Animate new row
        setTimeout(() => {
            row.style.opacity = '1';
            row.style.transform = 'translateY(0)';
        }, 50);
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
            updateTradeData(data);
            updateActiveTradesCount();
            updateTotalPL();
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
        amount: parseFloat(formData.get('amount_usdt')),
        order_type: formData.get('order_type'),
        limit_price: formData.get('order_type') === 'limit' ? parseFloat(formData.get('limit_price')) : null
    };
    
    fetch('/trade', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(tradeData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to create trade');
        }
        return response.json();
    })
    .then(data => {
        // Show success state
        submitButton.textContent = 'Success!';
        submitButton.style.backgroundColor = 'var(--success-color)';
        
        // Reset form
        form.reset();
        document.getElementById('limit_price_container').style.display = 'none';
        
        // Reset button after delay
        setTimeout(() => {
            submitButton.disabled = false;
            submitButton.textContent = originalText;
            submitButton.style.backgroundColor = '';
        }, 2000);
    })
    .catch(error => {
        console.error('Error creating trade:', error);
        
        // Show error state
        submitButton.textContent = 'Error';
        submitButton.style.backgroundColor = 'var(--danger-color)';
        
        // Reset button after delay
        setTimeout(() => {
            submitButton.disabled = false;
            submitButton.textContent = originalText;
            submitButton.style.backgroundColor = '';
        }, 2000);
        
        alert('Failed to create trade. Please try again.');
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

function addTradeToTable(trade) {
    const tableBody = document.querySelector('#tradesTableBody');
    const row = document.createElement('tr');
    row.setAttribute('data-trade-id', trade.id);

    // Format numbers with commas and 2 decimal places
    const formatNumber = (num) => {
        return parseFloat(num).toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    };

    row.innerHTML = `
        <td>${trade.coin}</td>
        <td>${formatNumber(trade.amount_usdt)}</td>
        <td>${formatNumber(trade.entry_price)}</td>
        <td class="current-price">${formatNumber(trade.current_price)}</td>
        <td class="profit-loss ${parseFloat(trade.profit_loss) >= 0 ? 'positive' : 'negative'}">
            ${formatNumber(trade.profit_loss)}
        </td>
        <td>${formatNumber(trade.fees)}</td>
        <td>${(trade.roi !== undefined && trade.roi !== null) ? trade.roi.toFixed(2) : 'N/A'}%</td>
        <td class="${trade.status === 'Open' ? 'status-open' : 'status-pending'}">${trade.status}</td>
        <td>
            ${trade.status === "Open" ? 
                `<button onclick="closeTrade('${trade.id}')" class="close-btn">Close</button>` : 
                `<button onclick="declineTrade('${trade.id}')" class="decline-btn">Decline</button>`
            }
        </td>
    `;

    tableBody.appendChild(row);
    updateTotalPL(); // Update total P/L when new trade is added
}

function closeTrade(tradeId) {
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
                    updateTotalPL(); // Update total P/L when trade is closed
                }, 500);
            }
        } else {
            alert('Error closing trade: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error closing trade. Please try again.');
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize WebSocket connection
    connectWebSocket();
    
    // Add form submit handler
    const tradeForm = document.getElementById('tradeForm');
    tradeForm.addEventListener('submit', handleTradeSubmit);
    
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
    const amountInput = document.getElementById('amount_usdt');
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
}); 