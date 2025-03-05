#paper_trading.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stock_data import fetch_stock_data

class PaperTradingSimulator:
    """
    A simulator for paper trading options strategies
    """
    
    def __init__(self, initial_balance=100000.0, trades=None):
        """
        Initialize the paper trading simulator
        
        Parameters:
        initial_balance (float): Initial account balance in USD
        trades (list): List of existing trades (optional)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = trades if trades is not None else []
        self.trade_history = []
    
    def sell_put(self, stock, strike, premium, expiry, current_price, delta=None):
        """
        Simulate selling a naked put option
        
        Parameters:
        stock (str): Stock ticker symbol
        strike (float): Strike price of the option
        premium (float): Premium received per share
        expiry (str): Expiration date (YYYY-MM-DD)
        current_price (float): Current stock price
        delta (float): Option delta (optional)
        
        Returns:
        bool: True if successful, False otherwise
        """
        # Calculate contract value (1 contract = 100 shares)
        premium_per_contract = premium * 100
        
        # Calculate required capital (cash-secured put)
        required_capital = strike * 100
        
        # Check if enough balance available
        if required_capital > self.balance:
            return False
        
        # Add trade to list
        trade = {
            'stock': stock,
            'type': 'naked_put',
            'strike': strike,
            'premium': premium,
            'contract_premium': premium_per_contract,
            'current_price': current_price,
            'expiry': expiry,
            'delta': delta,
            'required_capital': required_capital,
            'date_opened': datetime.now().strftime('%Y-%m-%d'),
            'status': 'open',
            'pnl': 0.0
        }
        
        self.trades.append(trade)
        
        # Add premium to balance
        self.balance += premium_per_contract
        
        # Reserve capital for the put
        self.balance -= required_capital
        
        return True
    
    def close_position(self, trade_index):
        """
        Close an open position
        
        Parameters:
        trade_index (int): Index of the trade in the trades list
        
        Returns:
        bool: True if successful, False otherwise
        """
        if trade_index < 0 or trade_index >= len(self.trades):
            return False
        
        trade = self.trades[trade_index]
        
        # Check if the position is already closed
        if trade['status'] != 'open':
            return False
        
        # Get current stock price
        stock_data = fetch_stock_data(trade['stock'])
        current_price = stock_data['Close'].iloc[-1]
        
        # Calculate P&L for the trade
        if trade['type'] == 'naked_put':
            pnl = self.calculate_pnl(trade, current_price)
            
            # Update trade information
            trade['status'] = 'closed'
            trade['date_closed'] = datetime.now().strftime('%Y-%m-%d')
            trade['closing_price'] = current_price
            trade['pnl'] = pnl
            
            # Update balance
            self.balance += trade['required_capital']  # Return the reserved capital
            self.balance += pnl  # Add or subtract P&L
            
            # Update the trade in the list
            self.trades[trade_index] = trade
            
            # Add to trade history
            self.trade_history.append(trade)
            
            return True
        
        return False
    
    def calculate_pnl(self, trade, current_price):
        """
        Calculate the P&L for a trade
        
        Parameters:
        trade (dict): Trade information
        current_price (float): Current stock price
        
        Returns:
        float: Profit or loss amount
        """
        if trade['type'] == 'naked_put':
            # For naked puts:
            # - If stock price > strike, profit is the premium
            # - If stock price < strike, loss is (strike - stock_price) * 100 - premium
            premium_per_contract = trade['premium'] * 100
            
            if current_price >= trade['strike']:
                # Put expires worthless, profit is the premium
                return premium_per_contract
            else:
                # Put is ITM, calculate loss
                loss_per_share = trade['strike'] - current_price
                total_loss = loss_per_share * 100
                net_pnl = premium_per_contract - total_loss
                return net_pnl
        
        return 0.0
    
    def update_open_positions(self):
        """
        Update all open positions with current market data
        
        Returns:
        list: Updated trades
        """
        for i, trade in enumerate(self.trades):
            if trade['status'] == 'open':
                # Get current stock price
                stock_data = fetch_stock_data(trade['stock'])
                current_price = stock_data['Close'].iloc[-1]
                
                # Check if option has expired
                expiry_date = datetime.strptime(trade['expiry'], '%Y-%m-%d')
                if datetime.now() > expiry_date:
                    # Automatically close the position
                    self.close_position(i)
                else:
                    # Update the unrealized P&L
                    trade['unrealized_pnl'] = self.calculate_pnl(trade, current_price)
                    trade['current_price'] = current_price
                    self.trades[i] = trade
        
        return self.trades
    
    def get_portfolio_summary(self):
        """
        Get a summary of the portfolio
        
        Returns:
        dict: Portfolio summary information
        """
        # Update positions first
        self.update_open_positions()
        
        # Calculate portfolio statistics
        total_open_positions = len([t for t in self.trades if t['status'] == 'open'])
        total_closed_positions = len([t for t in self.trades if t['status'] == 'closed'])
        
        # Calculate realized P&L
        realized_pnl = sum([t.get('pnl', 0) for t in self.trades if t['status'] == 'closed'])
        
        # Calculate unrealized P&L
        unrealized_pnl = sum([t.get('unrealized_pnl', 0) for t in self.trades if t['status'] == 'open'])
        
        # Calculate total capital at risk
        capital_at_risk = sum([t.get('required_capital', 0) for t in self.trades if t['status'] == 'open'])
        
        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_pnl': realized_pnl + unrealized_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'open_positions': total_open_positions,
            'closed_positions': total_closed_positions,
            'capital_at_risk': capital_at_risk,
            'available_capital': self.balance - capital_at_risk
        }


def calculate_put_option_greeks(stock_price, strike_price, days_to_expiry, risk_free_rate=0.05, implied_volatility=0.3):
    """
    Calculate approximate option greeks for a put option
    
    Parameters:
    stock_price (float): Current stock price
    strike_price (float): Option strike price
    days_to_expiry (int): Days to expiration
    risk_free_rate (float): Risk-free interest rate
    implied_volatility (float): Implied volatility
    
    Returns:
    dict: Option greeks (delta, gamma, theta, vega)
    """
    from scipy.stats import norm
    import numpy as np
    
    # Convert days to years
    t = days_to_expiry / 365.0
    
    # If time to expiry is very small, return approximate values
    if t <= 0.0001:
        return {
            'delta': -1.0 if stock_price < strike_price else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
    
    # Calculate d1 and d2
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate + (implied_volatility**2) / 2) * t) / (implied_volatility * np.sqrt(t))
    d2 = d1 - implied_volatility * np.sqrt(t)
    
    # Calculate greeks
    delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (stock_price * implied_volatility * np.sqrt(t))
    theta = -(stock_price * norm.pdf(d1) * implied_volatility) / (2 * np.sqrt(t)) + risk_free_rate * strike_price * np.exp(-risk_free_rate * t) * norm.cdf(-d2)
    vega = stock_price * np.sqrt(t) * norm.pdf(d1) / 100  # Dividing by 100 to get the change per 1% vol change
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365.0,  # Daily theta
        'vega': vega
    }