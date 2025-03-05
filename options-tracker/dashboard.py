#dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from stock_data import fetch_stock_data, get_stock_info, calculate_historical_volatility
from option_analysis import get_option_chain, recommend_put_strategies

def create_dashboard(stocks):
    """
    Create the main dashboard view
    
    Parameters:
    stocks (list): List of stock tickers to display
    """
    st.header("Options Dashboard")
    
    # Create columns for header metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Stocks Tracked", len(stocks))
    
    with col2:
        # Count total available options
        total_options = 0
        for ticker in stocks: