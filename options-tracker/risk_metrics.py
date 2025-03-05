#risk_metrics.py
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from stock_data import fetch_stock_data, calculate_historical_volatility
from risk_simulation import run_monte_carlo_simulation

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

def analyze_portfolio_risk(portfolio_positions):
    """
    Analyze the risk of a portfolio of put options
    
    Parameters:
    portfolio_positions (DataFrame): DataFrame containing put option positions
    
    Returns:
    dict: Portfolio risk metrics
    """
    if portfolio_positions.empty:
        return {}
    
    # Calculate maximum potential loss
    max_loss = calculate_max_loss(portfolio_positions)
    
    # Calculate Value at Risk
    var_95 = calculate_value_at_risk(portfolio_positions, confidence_level=0.95)
    var_99 = calculate_value_at_risk(portfolio_positions, confidence_level=0.99)
    
    # Calculate weighted average metrics
    weights = portfolio_positions['capital_required'] / portfolio_positions['capital_required'].sum()
    
    weighted_avg_delta = np.average(portfolio_positions['delta'], weights=weights)
    weighted_avg_days = np.average(portfolio_positions['days_to_expiry'], weights=weights)
    
    # Calculate diversification metrics
    stock_count = portfolio_positions['ticker'].nunique()
    expiry_count = portfolio_positions['expiry'].nunique()
    
    # Check if positions are highly correlated
    correlation_risk = "Low"
    if stock_count >= 2:
        try:
            stock_list = portfolio_positions['ticker'].unique()
            corr_matrix = calculate_correlation_matrix(stock_list)
            
            # Calculate average correlation
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
            
            if avg_corr > 0.7:
                correlation_risk = "High"
            elif avg_corr > 0.5:
                correlation_risk = "Medium"
        except:
            correlation_risk = "Unknown"
    
    return {
        'max_loss': max_loss,
        'var_95': var_95,
        'var_99': var_99,
        'weighted_avg_delta': weighted_avg_delta,
        'weighted_avg_days': weighted_avg_days,
        'stock_count': stock_count,
        'expiry_count': expiry_count,
        'correlation_risk': correlation_risk
    }

def calculate_correlation_matrix(stocks, period="1y"):
    """
    Calculate the correlation matrix for a list of stocks
    
    Parameters:
    stocks (list): List of stock tickers
    period (str): Time period for historical data
    
    Returns:
    DataFrame: Correlation matrix
    """
    # Fetch historical data for all stocks
    all_data = {}
    for ticker in stocks:
        try:
            df = fetch_stock_data(ticker, period=period)
            all_data[ticker] = df['Close']
        except Exception as e:
            pass
    
    # Create a dataframe with all closing prices
    if all_data:
        prices_df = pd.DataFrame(all_data)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    return pd.DataFrame()