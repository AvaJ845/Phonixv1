#option_analysis.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from stock_data import fetch_stock_data

@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_option_chain(ticker):
    """
    Fetch options chain for a given stock
    
    Parameters:
    ticker (str): Stock ticker symbol
    
    Returns:
    dict: Dictionary of dataframes containing options data by expiration date
    """
    try:
        stock = yf.Ticker(ticker)
        options_data = {}
        
        # Get all available expiration dates
        expirations = stock.options
        
        for expiry in expirations:
            # Get option chain for this expiration
            opts = stock.option_chain(expiry)
            
            # Store put options in our dictionary
            puts_df = opts.puts
            puts_df['expiration'] = expiry
            options_data[expiry] = puts_df
        
        return options_data
    except Exception as e:
        st.error(f"Error fetching options for {ticker}: {str(e)}")
        return {}


def filter_low_delta_puts(puts_df, max_delta=0.1):
    """
    Filter put options to find those with delta < max_delta
    
    Parameters:
    puts_df (DataFrame): DataFrame containing put options data
    max_delta (float): Maximum delta value to filter by
    
    Returns:
    DataFrame: Filtered DataFrame containing only puts with delta < max_delta
    """
    # Ensure delta is negative for puts and convert to absolute value
    if 'delta' in puts_df.columns:
        puts_df['delta'] = puts_df['delta'].abs()
        
        # Filter by delta threshold
        low_delta_puts = puts_df[puts_df['delta'] < max_delta].copy()
        
        # Add additional metrics if needed
        if not low_delta_puts.empty:
            # Calculate annualized return
            days_to_expiry = (datetime.strptime(low_delta_puts['expiration'].iloc[0], '%Y-%m-%d') - datetime.now()).days
            if days_to_expiry <= 0:
                days_to_expiry = 1  # Avoid division by zero
            
            low_delta_puts['annualized_return'] = (low_delta_puts['lastPrice'] / low_delta_puts['strike']) * (365 / days_to_expiry) * 100
            
            # Calculate risk-reward ratio (premium to distance from strike)
            try:
                # Try to get stock price from the contract symbol (first part before the date)
                ticker = low_delta_puts['contractSymbol'].iloc[0].split()[0]
                stock_data = fetch_stock_data(ticker)
                stock_price = stock_data['Close'].iloc[-1]
            except:
                # If that fails, use a heuristic approximation of current price
                stock_price = low_delta_puts['lastPrice'].iloc[0] + low_delta_puts['strike'].iloc[0]
            
            low_delta_puts['distance_pct'] = ((low_delta_puts['strike'] - stock_price) / stock_price) * 100
            low_delta_puts['premium_pct'] = (low_delta_puts['lastPrice'] / low_delta_puts['strike']) * 100
            
            # Calculate theta (time decay) to delta ratio
            if 'theta' in low_delta_puts.columns:
                low_delta_puts['theta_delta_ratio'] = low_delta_puts['theta'].abs() / low_delta_puts['delta']
            
            # Sort by delta (ascending)
            low_delta_puts = low_delta_puts.sort_values('delta')
        
        return low_delta_puts
    else:
        st.warning("Delta information not available in the options data")
        return pd.DataFrame()


def get_all_low_delta_puts(ticker, max_delta=0.1):
    """
    Get all put options with delta < max_delta across all available expirations
    
    Parameters:
    ticker (str): Stock ticker symbol
    max_delta (float): Maximum delta value to filter by
    
    Returns:
    DataFrame: Combined DataFrame of all low delta puts
    """
    try:
        # Get current stock price
        stock_data = fetch_stock_data(ticker)
        current_price = stock_data['Close'].iloc[-1]
        
        # Get all option chains
        option_chains = get_option_chain(ticker)
        
        # Collect all low delta puts
        all_low_delta_puts = []
        
        for expiry, puts_df in option_chains.items():
            if 'delta' in puts_df.columns:
                # Convert delta to absolute value
                puts_df['delta'] = puts_df['delta'].abs()
                
                # Filter by delta
                low_delta_puts = puts_df[puts_df['delta'] < max_delta].copy()
                
                if not low_delta_puts.empty:
                    # Add expiration and days to expiry
                    days_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
                    if days_to_expiry <= 0:
                        days_to_expiry = 1  # Avoid division by zero
                        
                    low_delta_puts['days_to_expiry'] = days_to_expiry
                    
                    # Calculate additional metrics
                    low_delta_puts['annualized_return'] = (low_delta_puts['lastPrice'] / low_delta_puts['strike']) * (365 / days_to_expiry) * 100
                    low_delta_puts['distance_pct'] = ((low_delta_puts['strike'] - current_price) / current_price) * 100
                    
                    # Add current price and ticker
                    low_delta_puts['current_price'] = current_price
                    low_delta_puts['ticker'] = ticker
                    
                    all_low_delta_puts.append(low_delta_puts)
        
        # Combine all low delta puts
        if all_low_delta_puts:
            combined_df = pd.concat(all_low_delta_puts)
            return combined_df
        else:
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error getting low delta puts for {ticker}: {str(e)}")
        return pd.DataFrame()


def recommend_put_strategies(ticker, options_data, risk_tolerance='medium'):
    """
    Recommend put selling strategies based on risk tolerance
    
    Parameters:
    ticker (str): Stock ticker symbol
    options_data (dict): Dictionary of options data by expiration
    risk_tolerance (str): Risk tolerance level ('low', 'medium', 'high')
    
    Returns:
    DataFrame: Recommended put options to sell
    """
    try:
        # Get stock price and volatility data
        stock_data = fetch_stock_data(ticker)
        current_price = stock_data['Close'].iloc[-1]
        
        recommendations = []
        
        # Define delta thresholds based on risk tolerance
        if risk_tolerance == 'low':
            max_delta = 0.05
            desired_dte = [30, 45]  # Days to expiration range
        elif risk_tolerance == 'medium':
            max_delta = 0.10
            desired_dte = [45, 60]
        else:  # high
            max_delta = 0.20
            desired_dte = [60, 90]
        
        for expiry, puts_df in options_data.items():
            # Calculate days to expiration
            days_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
            
            # Check if this expiration fits our desired DTE range
            if days_to_expiry < desired_dte[0] or days_to_expiry > desired_dte[1]:
                continue
            
            # Filter puts by delta
            if 'delta' in puts_df.columns:
                # Convert delta to absolute value since puts have negative delta
                puts_df['delta_abs'] = puts_df['delta'].abs()
                
                # Filter by delta threshold
                filtered_puts = puts_df[puts_df['delta_abs'] < max_delta].copy()
                
                for _, row in filtered_puts.iterrows():
                    # Calculate metrics
                    annualized_return = (row['lastPrice'] / row['strike']) * (365 / days_to_expiry) * 100
                    distance_pct = ((row['strike'] - current_price) / current_price) * 100
                    
                    # Add to recommendations
                    recommendations.append({
                        'ticker': ticker,
                        'expiry': expiry,
                        'strike': row['strike'],
                        'premium': row['lastPrice'],
                        'delta': row['delta_abs'],
                        'days_to_expiry': days_to_expiry,
                        'annualized_return': annualized_return,
                        'distance_pct': distance_pct,
                        'current_price': current_price
                    })
        
        # Convert to DataFrame and sort by annualized return
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            
            # Calculate risk-adjusted return (annualized return / delta)
            rec_df['risk_adjusted_return'] = rec_df['annualized_return'] / rec_df['delta']
            
            # Sort by risk-adjusted return (descending)
            rec_df = rec_df.sort_values('risk_adjusted_return', ascending=False)
            
            return rec_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error generating recommendations for {ticker}: {str(e)}")
        return pd.DataFrame()