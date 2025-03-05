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
        st.write(f"Fetching options for {ticker}...")
        stock = yf.Ticker(ticker)
        options_data = {}
        
        # Get all available expiration dates
        expirations = stock.options
        st.write(f"Found {len(expirations)} expiration dates")
        
        for expiry in expirations:
            # Get option chain for this expiration
            opts = stock.option_chain(expiry)
            
            # Store put options in our dictionary
            puts_df = opts.puts
            puts_df['expiration'] = expiry
            
            # Check if 'delta' is in the columns, if not, calculate approximate delta
            if 'delta' not in puts_df.columns:
                st.write(f"Delta information not available for {ticker} - {expiry}, approximating...")
                # Get current stock price
                stock_data = fetch_stock_data(ticker)
                current_price = stock_data['Close'].iloc[-1]
                
                # Calculate days to expiry
                days_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
                if days_to_expiry <= 0:
                    days_to_expiry = 1
                
                # Approximate delta based on moneyness and DTE
                # This is a very rough approximation
                puts_df['delta'] = puts_df.apply(
                    lambda row: approximate_delta(current_price, row['strike'], days_to_expiry, 'put'),
                    axis=1
                )
            
            options_data[expiry] = puts_df
        
        return options_data
    except Exception as e:
        st.error(f"Error fetching options for {ticker}: {str(e)}")
        return {}

def approximate_delta(stock_price, strike_price, days_to_expiry, option_type, implied_vol=0.3):
    """
    Approximate option delta when not available from API
    
    Parameters:
    stock_price (float): Current stock price
    strike_price (float): Option strike price
    days_to_expiry (int): Days to expiration
    option_type (str): 'call' or 'put'
    implied_vol (float): Implied volatility estimate
    
    Returns:
    float: Approximate delta value
    """
    # For puts:
    # ITM puts (S < K) have delta approaching -1 as S decreases
    # OTM puts (S > K) have delta approaching 0 as S increases
    
    # Simple approximation based on moneyness and time
    moneyness = stock_price / strike_price
    time_factor = min(1, days_to_expiry / 365)  # Normalize time to max of 1
    
    if option_type.lower() == 'put':
        if moneyness < 0.85:  # Deep ITM
            delta = -0.9 * time_factor
        elif moneyness < 0.95:  # ITM
            delta = -0.7 * time_factor
        elif moneyness < 1.0:  # Near the money
            delta = -0.5 * time_factor
        elif moneyness < 1.05:  # Slightly OTM
            delta = -0.3 * time_factor
        elif moneyness < 1.15:  # OTM
            delta = -0.15 * time_factor
        else:  # Deep OTM
            delta = -0.05 * time_factor
    else:  # Call option
        if moneyness > 1.15:  # Deep ITM
            delta = 0.9 * time_factor
        elif moneyness > 1.05:  # ITM
            delta = 0.7 * time_factor
        elif moneyness > 1.0:  # Near the money
            delta = 0.5 * time_factor
        elif moneyness > 0.95:  # Slightly OTM
            delta = 0.3 * time_factor
        elif moneyness > 0.85:  # OTM
            delta = 0.15 * time_factor
        else:  # Deep OTM
            delta = 0.05 * time_factor
    
    return delta

def filter_low_delta_puts(puts_df, max_delta=0.1):
    """
    Filter put options to find those with delta < max_delta
    
    Parameters:
    puts_df (DataFrame): DataFrame containing put options data
    max_delta (float): Maximum delta value to filter by
    
    Returns:
    DataFrame: Filtered DataFrame containing only puts with delta < max_delta
    """
    # Make a copy to avoid modifying the original
    df = puts_df.copy()
    
    # Ensure delta is present
    if 'delta' not in df.columns:
        st.warning("Delta information not available in the options data. Using approximation.")
        return pd.DataFrame()
    
    # Ensure delta is negative for puts and convert to absolute value
    df['delta'] = df['delta'].abs()
    
    # Display before filtering
    st.write(f"Before filtering: {len(df)} options")
    st.write(f"Delta range: {df['delta'].min()} to {df['delta'].max()}")
    
    # Filter by delta threshold
    low_delta_puts = df[df['delta'] < max_delta].copy()
    st.write(f"After filtering: {len(low_delta_puts)} options with delta < {max_delta}")
    
    # Add additional metrics if needed
    if not low_delta_puts.empty:
        # Calculate days to expiry
        expiry_date = low_delta_puts['expiration'].iloc[0]
        days_to_expiry = (datetime.strptime(expiry_date, '%Y-%m-%d') - datetime.now()).days
        if days_to_expiry <= 0:
            days_to_expiry = 1  # Avoid division by zero
        
        # Calculate annualized return
        low_delta_puts['annualized_return'] = (low_delta_puts['lastPrice'] / low_delta_puts['strike']) * (365 / days_to_expiry) * 100
        
        # Get stock price for distance calculation
        try:
            ticker = puts_df['contractSymbol'].iloc[0].split()[0]
            stock_data = fetch_stock_data(ticker)
            stock_price = stock_data['Close'].iloc[-1]
        except:
            # Fallback: Use the ticker from the first few characters of contractSymbol
            try:
                # Common pattern is like "AAPL230915P00150000"
                ticker_symbol = puts_df['contractSymbol'].iloc[0]
                # Extract the alphabetic prefix (ticker symbol)
                ticker = ''.join([c for c in ticker_symbol if c.isalpha()])
                stock_data = fetch_stock_data(ticker)
                stock_price = stock_data['Close'].iloc[-1]
            except:
                # Last resort: approximate current price
                st.warning("Could not determine stock price, using approximation")
                atm_options = puts_df.iloc[(puts_df['inTheMoney'] == False).argmin():].head(3)
                stock_price = atm_options['strike'].mean()
        
        # Calculate distance and other metrics
        low_delta_puts['distance_pct'] = ((low_delta_puts['strike'] - stock_price) / stock_price) * 100
        low_delta_puts['premium_pct'] = (low_delta_puts['lastPrice'] / low_delta_puts['strike']) * 100
        
        # Calculate theta (time decay) to delta ratio if available
        if 'theta' in low_delta_puts.columns:
            low_delta_puts['theta_delta_ratio'] = low_delta_puts['theta'].abs() / low_delta_puts['delta']
        
        # Sort by delta (ascending)
        low_delta_puts = low_delta_puts.sort_values('delta')
    
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
        st.write(f"Retrieved {len(option_chains)} expiration dates for {ticker}")
        
        # Collect all low delta puts
        all_low_delta_puts = []
        
        for expiry, puts_df in option_chains.items():
            st.write(f"Processing {expiry}: {len(puts_df)} put options")
            
            # Make a copy to avoid modifying the original
            df = puts_df.copy()
            
            # Ensure delta is present and handle negative values
            if 'delta' not in df.columns:
                st.warning(f"Delta information not available for {ticker} - {expiry}. Skipping.")
                continue
            
            # Convert delta to absolute value
            df['delta'] = df['delta'].abs()
            
            # Filter by delta
            low_delta_puts = df[df['delta'] < max_delta].copy()
            st.write(f"Found {len(low_delta_puts)} options with delta < {max_delta}")
            
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
            st.write(f"Total low delta puts found: {len(combined_df)}")
            return combined_df
        else:
            st.warning(f"No low delta puts found for {ticker}")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error getting low delta puts for {ticker}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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