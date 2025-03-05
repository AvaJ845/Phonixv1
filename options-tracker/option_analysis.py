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
    Fetch options chain for a given stock with complete greeks
    
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
            try:
                # Get option chain for this expiration with all greeks
                opts = stock.option_chain(expiry)
                
                # Store put options in our dictionary
                puts_df = opts.puts
                
                # Ensure we have delta information
                if 'delta' not in puts_df.columns:
                    # Calculate approximate delta if not available
                    current_price = stock.history(period='1d')['Close'].iloc[-1]
                    puts_df['delta'] = puts_df.apply(
                        lambda row: calculate_approximate_delta(
                            current_price, 
                            row['strike'], 
                            (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days,
                            row['impliedVolatility'] if 'impliedVolatility' in puts_df.columns else 0.3,
                            is_put=True
                        ),
                        axis=1
                    )
                
                puts_df['expiration'] = expiry
                options_data[expiry] = puts_df
                
            except Exception as e:
                st.warning(f"Could not fetch complete data for expiration {expiry}: {str(e)}")
                continue
        
        if not options_data:
            st.error("No valid options data could be fetched")
            return {}
            
        return options_data
    
    except Exception as e:
        st.error(f"Error fetching options for {ticker}: {str(e)}")
        return {}

def calculate_approximate_delta(S, K, T, sigma, is_put=True):
    """
    Calculate approximate delta using a simplified Black-Scholes formula
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (int): Days to expiration
    sigma (float): Implied volatility
    is_put (bool): True if calculating for put option
    
    Returns:
    float: Approximate delta value
    """
    try:
        from scipy.stats import norm
        
        # Convert days to years
        T = max(T, 1) / 365.0
        
        # Risk-free rate (simplified)
        r = 0.05
        
        # Calculate d1 from Black-Scholes
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        
        # Calculate delta
        if is_put:
            delta = -norm.cdf(-d1)
        else:
            delta = norm.cdf(d1)
            
        return delta
    
    except ImportError:
        # Fallback to very simple delta approximation if scipy not available
        moneyness = S/K
        if is_put:
            if moneyness < 0.95:
                return -0.8
            elif moneyness > 1.05:
                return -0.2
            else:
                return -0.5
        else:
            if moneyness < 0.95:
                return 0.2
            elif moneyness > 1.05:
                return 0.8
            else:
                return 0.5

def filter_low_delta_puts(puts_df, max_delta=0.1):
    """
    Filter put options to find those with delta < 0.1
    
    Parameters:
    puts_df (DataFrame): DataFrame containing put options data
    max_delta (float): Maximum delta value to filter by (fixed at 0.1)
    
    Returns:
    DataFrame: Filtered DataFrame containing only puts with delta < 0.1
    """
    if puts_df.empty:
        st.warning("No put options data available")
        return pd.DataFrame()
        
    # Ensure delta is available and properly formatted
    if 'delta' not in puts_df.columns:
        st.warning("Delta information not available in the options data")
        return pd.DataFrame()
        
    # Convert delta to absolute value since puts have negative delta
    puts_df['delta'] = puts_df['delta'].abs()
    
    # Filter by fixed delta threshold of 0.1
    low_delta_puts = puts_df[puts_df['delta'] < max_delta].copy()
    
    if low_delta_puts.empty:
        st.info(f"No puts found with delta < {max_delta}")
        return pd.DataFrame()
        
    # Add additional metrics
    try:
        # Calculate days to expiry
        days_to_expiry = (datetime.strptime(low_delta_puts['expiration'].iloc[0], '%Y-%m-%d') - datetime.now()).days
        days_to_expiry = max(days_to_expiry, 1)  # Avoid division by zero
        
        # Calculate annualized return
        low_delta_puts['annualized_return'] = (low_delta_puts['lastPrice'] / low_delta_puts['strike']) * (365 / days_to_expiry) * 100
        
        # Calculate risk-reward metrics
        stock_price = low_delta_puts['lastPrice'].iloc[0] + low_delta_puts['strike'].iloc[0]
        low_delta_puts['distance_pct'] = ((low_delta_puts['strike'] - stock_price) / stock_price) * 100
        low_delta_puts['premium_pct'] = (low_delta_puts['lastPrice'] / low_delta_puts['strike']) * 100
        
        # Calculate theta-delta ratio if theta is available
        if 'theta' in low_delta_puts.columns:
            low_delta_puts['theta_delta_ratio'] = low_delta_puts['theta'].abs() / low_delta_puts['delta']
        
        # Sort by delta for better analysis
        low_delta_puts = low_delta_puts.sort_values('delta')
        
    except Exception as e:
        st.warning(f"Error calculating additional metrics: {str(e)}")
    
    return low_delta_puts

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
            # Convert delta to absolute value
            puts_df['delta'] = puts_df['delta'].abs()
            
            # Filter by delta
            low_delta_puts = puts_df[puts_df['delta'] < max_delta].copy()
            
            if not low_delta_puts.empty:
                # Add expiration and days to expiry
                days_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
                days_to_expiry = max(days_to_expiry, 1)  # Avoid division by zero
                    
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

def recommend_put_strategies(ticker, options_data, risk_tolerance='low'):
    """
    Recommend put selling strategies focusing on deltas under 0.1
    
    Parameters:
    ticker (str): Stock ticker symbol
    options_data (dict): Dictionary of options data by expiration
    risk_tolerance (str): Risk tolerance level ('low', 'medium' - both capped at 0.1 delta)
    
    Returns:
    DataFrame: Recommended put options to sell with delta < 0.1
    """
    try:
        # Get stock price and volatility data
        stock_data = fetch_stock_data(ticker)
        current_price = stock_data['Close'].iloc[-1]
        
        recommendations = []
        
        # Define delta thresholds - now capped at 0.1 for all risk levels
        if risk_tolerance == 'low':
            max_delta = 0.05
            desired_dte = [30, 45]  # Days to expiration range
        else:  # medium risk - still capped at 0.1
            max_delta = 0.1
            desired_dte = [45, 60]
        
        for expiry, puts_df in options_data.items():
            # Calculate days to expiration
            days_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
            days_to_expiry = max(days_to_expiry, 1)  # Avoid division by zero
            
            # Check if this expiration fits our desired DTE range
            if days_to_expiry < desired_dte[0] or days_to_expiry > desired_dte[1]:
                continue
            
            # Filter puts by delta
            if 'delta' in puts_df.columns:
                # Convert delta to absolute value since puts have negative delta
                puts_df['delta_abs'] = puts_df['delta'].abs()
                
                # Filter by delta threshold - ensure it's under 0.1
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
            
            # Calculate risk-adjusted return
            rec_df['risk_adjusted_return'] = rec_df['annualized_return'] / rec_df['delta']
            
            # Sort by risk-adjusted return
            rec_df = rec_df.sort_values('risk_adjusted_return', ascending=False)
            return rec_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error generating recommendations for {ticker}: {str(e)}")
        return pd.DataFrame()