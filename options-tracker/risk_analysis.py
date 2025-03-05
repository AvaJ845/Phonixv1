#risk_analysis.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
import math

from stock_data import fetch_stock_data, calculate_historical_volatility
from option_analysis import get_option_chain, filter_low_delta_puts
from paper_trading import calculate_put_option_greeks

def run_monte_carlo_simulation(stock_price, days, volatility, num_simulations=1000, seed=None):
    """
    Run a Monte Carlo simulation for stock price movements
    
    Parameters:
    stock_price (float): Current stock price
    days (int): Number of days to simulate
    volatility (float): Annual volatility as a decimal
    num_simulations (int): Number of simulations to run
    seed (int): Random seed for reproducibility
    
    Returns:
    ndarray: Simulated price paths (num_simulations x days)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Daily volatility
    daily_vol = volatility / np.sqrt(252)
    
    # Daily returns are normally distributed with mean 0 and std = daily_vol
    daily_returns = np.random.normal(0, daily_vol, (num_simulations, days))
    
    # Calculate price paths
    price_paths = np.zeros((num_simulations, days + 1))
    price_paths[:, 0] = stock_price
    
    for t in range(1, days + 1):
        price_paths[:, t] = price_paths[:, t-1] * np.exp(daily_returns[:, t-1])
    
    return price_paths

def calculate_put_assignment_probability(current_price, strike_price, days_to_expiry, volatility):
    """
    Calculate the probability of a put option being assigned at expiration
    
    Parameters:
    current_price (float): Current stock price
    strike_price (float): Option strike price
    days_to_expiry (int): Days until option expiration
    volatility (float): Annual volatility as a decimal
    
    Returns:
    float: Probability of assignment (0-1)
    """
    # Convert days to years
    t = days_to_expiry / 365.0
    
    # If already ITM, higher probability of assignment
    if current_price < strike_price:
        return 0.95
    
    # Calculate probability using Black-Scholes
    if t <= 0:
        return 0.0
    
    # Calculate d2 from Black-Scholes
    d2 = (np.log(current_price / strike_price) - (volatility ** 2 / 2) * t) / (volatility * np.sqrt(t))
    
    # Calculate probability that stock price will be below strike at expiration
    probability = norm.cdf(-d2)
    
    return probability

def calculate_max_loss(put_options_df):
    """
    Calculate the maximum potential loss for a portfolio of put options
    
    Parameters:
    put_options_df (DataFrame): DataFrame containing put option positions
    
    Returns:
    float: Maximum potential loss
    """
    if put_options_df.empty:
        return 0.0
    
    # Sum of (strike_price * 100 - premium * 100) for each position
    max_loss = ((put_options_df['strike'] * 100) - (put_options_df['premium'] * 100)).sum()
    
    return max_loss

def calculate_value_at_risk(put_options_df, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) for a portfolio of put options
    
    Parameters:
    put_options_df (DataFrame): DataFrame containing put option positions
    confidence_level (float): Confidence level for VaR calculation (e.g., 0.95 for 95%)
    
    Returns:
    float: Value at Risk
    """
    if put_options_df.empty:
        return 0.0
    
    # Simulate potential losses
    losses = []
    
    for _, position in put_options_df.iterrows():
        ticker = position['ticker']
        strike = position['strike']
        premium = position['premium']
        days_to_expiry = position['days_to_expiry']
        
        try:
            # Get stock price data
            stock_data = fetch_stock_data(ticker)
            current_price = stock_data['Close'].iloc[-1]
            
            # Calculate volatility
            volatility = calculate_historical_volatility(ticker) / 100  # Convert from percentage
            
            # Run a Monte Carlo simulation
            price_paths = run_monte_carlo_simulation(
                current_price, 
                days_to_expiry, 
                volatility, 
                num_simulations=1000
            )
            
            # Calculate losses for each simulation at expiration
            final_prices = price_paths[:, -1]
            position_losses = np.maximum(0, strike - final_prices) * 100 - premium * 100
            
            losses.append(position_losses)
        except Exception as e:
            st.warning(f"Error calculating VaR for {ticker}: {str(e)}")
    
    if losses:
        # Combine losses across all positions
        total_losses = np.sum(losses, axis=0)
        
        # Calculate VaR
        var = np.percentile(total_losses, confidence_level * 100)
        
        return var
    
    return 0.0

def analyze_option_risk(ticker, option_expiry, option_strike, risk_free_rate=0.035):
    """
    Perform detailed risk analysis for a specific option
    
    Parameters:
    ticker (str): Stock ticker symbol
    option_expiry (str): Option expiration date (YYYY-MM-DD)
    option_strike (float): Option strike price
    risk_free_rate (float): Risk-free interest rate
    
    Returns:
    dict: Risk metrics
    DataFrame: Scenario analysis results
    """
    try:
        # Get stock data
        stock_data = fetch_stock_data(ticker)
        current_price = stock_data['Close'].iloc[-1]
        
        # Calculate historical volatility
        volatility = calculate_historical_volatility(ticker) / 100  # Convert from percentage
        
        # Calculate days to expiry
        days_to_expiry = (datetime.strptime(option_expiry, '%Y-%m-%d') - datetime.now()).days
        if days_to_expiry <= 0:
            days_to_expiry = 1  # Avoid division by zero
        
        # Get option chain to find premium
        option_chain = get_option_chain(ticker)
        if option_expiry in option_chain:
            put_options = option_chain[option_expiry]
            
            # Find the specific option
            target_option = put_options[put_options['strike'] == option_strike]
            
            if not target_option.empty:
                premium = target_option['lastPrice'].iloc[0]
            else:
                # Estimate premium using Black-Scholes if not found
                t = days_to_expiry / 365.0
                d1 = (np.log(current_price / option_strike) + (risk_free_rate + volatility**2/2) * t) / (volatility * np.sqrt(t))
                d2 = d1 - volatility * np.sqrt(t)
                premium = option_strike * np.exp(-risk_free_rate * t) * norm.cdf(-d2) - current_price * norm.cdf(-d1)
        else:
            # Estimate premium using Black-Scholes
            t = days_to_expiry / 365.0
            d1 = (np.log(current_price / option_strike) + (risk_free_rate + volatility**2/2) * t) / (volatility * np.sqrt(t))
            d2 = d1 - volatility * np.sqrt(t)
            premium = option_strike * np.exp(-risk_free_rate * t) * norm.cdf(-d2) - current_price * norm.cdf(-d1)
        
        # Calculate option Greeks
        greeks = calculate_put_option_greeks(
            current_price,
            option_strike,
            days_to_expiry,
            risk_free_rate,
            volatility
        )
        
        # Calculate probability of assignment
        assignment_prob = calculate_put_assignment_probability(
            current_price,
            option_strike,
            days_to_expiry,
            volatility
        )
        
        # Calculate break-even price
        break_even = option_strike - premium
        
        # Calculate distance to break-even
        distance_to_be = ((current_price - break_even) / current_price) * 100
        
        # Calculate maximum loss and gain
        max_loss = (option_strike - premium) * 100
        max_gain = premium * 100
        
        # Calculate expected value
        expected_value = max_gain * (1 - assignment_prob) - max_loss * assignment_prob
        
        # Calculate risk-reward ratio
        risk_reward = max_gain / max_loss if max_loss > 0 else float('inf')
        
        # Run scenario analysis
        price_changes = np.arange(-30, 31, 5)  # -30% to +30% in 5% increments
        days_to_analyze = [0, 7, 14, days_to_expiry // 2, days_to_expiry]
        
        scenarios = []
        
        for price_change in price_changes:
            new_price = current_price * (1 + price_change / 100)
            
            for days_passed in days_to_analyze:
                if days_passed > days_to_expiry:
                    continue
                
                remaining_days = days_to_expiry - days_passed
                
                # Calculate new option value
                if remaining_days <= 0:
                    # At expiration
                    option_value = max(0, option_strike - new_price)
                else:
                    # Before expiration, use Black-Scholes
                    t = remaining_days / 365.0
                    d1 = (np.log(new_price / option_strike) + (risk_free_rate + volatility**2/2) * t) / (volatility * np.sqrt(t))
                    d2 = d1 - volatility * np.sqrt(t)
                    option_value = option_strike * np.exp(-risk_free_rate * t) * norm.cdf(-d2) - new_price * norm.cdf(-d1)
                
                # Calculate P&L
                pnl = (premium - option_value) * 100
                pnl_pct = (pnl / (option_strike * 100)) * 100
                
                scenarios.append({
                    'price_change': price_change,
                    'days_passed': days_passed,
                    'stock_price': new_price,
                    'option_value': option_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
        
        # Create scenario DataFrame
        scenario_df = pd.DataFrame(scenarios)
        
        # Create risk metrics dictionary
        risk_metrics = {
            'current_price': current_price,
            'option_strike': option_strike,
            'premium': premium,
            'days_to_expiry': days_to_expiry,
            'volatility': volatility * 100,  # Convert to percentage
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'assignment_probability': assignment_prob * 100,  # Convert to percentage
            'break_even': break_even,
            'distance_to_break_even': distance_to_be,
            'max_loss': max_loss,
            'max_gain': max_gain,
            'expected_value': expected_value,
            'risk_reward_ratio': risk_reward,
            'return_on_capital': (premium / option_strike) * 100,
            'annualized_return': (premium / option_strike) * (365 / days_to_expiry) * 100
        }
        
        return risk_metrics, scenario_df