#stock_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(ticker, period="6mo"):
    """
    Fetch stock data for the given ticker
    
    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Time period to fetch data for (default: 6 months)
    
    Returns:
    pandas.DataFrame: DataFrame containing stock price data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()


def get_stock_info(ticker):
    """
    Get comprehensive stock information
    
    Parameters:
    ticker (str): Stock ticker symbol
    
    Returns:
    dict: Dictionary containing stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key metrics
        metrics = {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'beta': info.get('beta', 0),
            'avg_volume': info.get('averageVolume', 0),
            '52wk_high': info.get('fiftyTwoWeekHigh', 0),
            '52wk_low': info.get('fiftyTwoWeekLow', 0),
            'analyst_target': info.get('targetMeanPrice', 0)
        }
        
        return metrics
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {str(e)}")
        return {}


def calculate_historical_volatility(ticker, days=30):
    """
    Calculate historical volatility for a given stock
    
    Parameters:
    ticker (str): Stock ticker symbol
    days (int): Number of days to calculate volatility for
    
    Returns:
    float: Annualized volatility as a percentage
    """
    try:
        # Get data for a bit longer than the requested period to ensure we have enough
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days*2)  
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        # Calculate daily returns
        df['daily_return'] = df['Close'].pct_change()
        
        # Calculate volatility (standard deviation of returns)
        daily_volatility = df['daily_return'].tail(days).std()
        
        # Annualize volatility (assuming 252 trading days in a year)
        annualized_volatility = daily_volatility * (252 ** 0.5)
        
        return annualized_volatility * 100  # Convert to percentage
    except Exception as e:
        st.error(f"Error calculating volatility for {ticker}: {str(e)}")
        return 0.0