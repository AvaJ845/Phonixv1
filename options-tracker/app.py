#app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

from stock_data import fetch_stock_data
from option_analysis import get_option_chain, filter_low_delta_puts
from paper_trading import PaperTradingSimulator
from dashboard import create_dashboard
from portfolio_optimization import display_portfolio_optimization
from risk_analysis import display_risk_analysis

# Set page configuration
st.set_page_config(
    page_title="Options Tracker - Naked Puts Scanner",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Options Tracker - Naked Puts Scanner")
st.markdown("""
This application tracks and analyzes naked put options for selected stocks, 
focusing on contracts with delta values under 0.1. It also provides recommendations 
and allows for paper trading simulation.
""")

# Define list of stocks to track
STOCKS = ["MSFT", "CMG", "NVDA", "TSLA", "V"]

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", [
    "Dashboard", 
    "Stock Analysis", 
    "Options Chain", 
    "Portfolio Optimization",
    "Advanced Risk Analysis",
    "Paper Trading"
])

# Initialize session state for paper trading
if 'paper_trades' not in st.session_state:
    st.session_state.paper_trades = []
if 'account_balance' not in st.session_state:
    st.session_state.account_balance = 100000.0  # Default starting balance

# Dashboard page
if page == "Dashboard":
    create_dashboard(STOCKS)

# Stock Analysis page
elif page == "Stock Analysis":
    st.header("Stock Analysis")
    
    # Stock selection
    selected_stock = st.selectbox("Select Stock", STOCKS)
    
    # Fetch stock data
    stock_data = fetch_stock_data(selected_stock)
    
    # Display stock info
    st.subheader(f"{selected_stock} Overview")
    
    # Display current price and basic metrics
    current_price = stock_data['Close'].iloc[-1]
    st.metric("Current Price", f"${current_price:.2f}", 
              f"{(stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]:.2%}")
    
    # Stock price chart
    st.subheader("Stock Price History")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Candlestick"
    ))
    fig.update_layout(
        title=f"{selected_stock} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# Options Chain page
elif page == "Options Chain":
    st.header("Options Chain Analysis")
    
    # Stock selection
    selected_stock = st.selectbox("Select Stock", STOCKS)
    
    # Fetch stock data to get current price
    stock_data = fetch_stock_data(selected_stock)
    current_price = stock_data['Close'].iloc[-1]
    st.metric("Current Price", f"${current_price:.2f}")
    
    # Options expiration dates selection
    option_chain = get_option_chain(selected_stock)
    expiration_dates = option_chain.keys()
    
    selected_expiry = st.selectbox("Select Expiration Date", list(expiration_dates))
    
    # Filter for low delta puts (< 0.1)
    low_delta_puts = filter_low_delta_puts(option_chain[selected_expiry])
    
    # Display filtered puts
    if not low_delta_puts.empty:
        st.subheader(f"Low Delta Puts (Delta < 0.1) for {selected_stock} expiring {selected_expiry}")
        st.dataframe(low_delta_puts)
        
        # Recommendations
        st.subheader("Recommendations")
        
        for idx, row in low_delta_puts.iterrows():
            strike = row['strike']
            premium = row['lastPrice']
            delta = row['delta']
            days_to_expiry = (datetime.strptime(selected_expiry, '%Y-%m-%d') - datetime.now()).days
            annualized_return = (premium / strike) * (365 / days_to_expiry) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Strike: ${strike:.2f}**")
                st.write(f"Premium: ${premium:.2f}")
                st.write(f"Delta: {delta:.4f}")
            
            with col2:
                st.write(f"Days to Expiry: {days_to_expiry}")
                st.write(f"Annualized Return: {annualized_return:.2f}%")
                
                # Add to paper trading button
                if st.button(f"Simulate Selling Put (Strike: ${strike:.2f})", key=f"add_{selected_stock}_{strike}"):
                    new_trade = {
                        'stock': selected_stock,
                        'type': 'naked_put',
                        'strike': strike,
                        'premium': premium,
                        'current_price': current_price,
                        'expiry': selected_expiry,
                        'delta': delta,
                        'date_opened': datetime.now().strftime('%Y-%m-%d'),
                        'status': 'open'
                    }
                    st.session_state.paper_trades.append(new_trade)
                    st.success(f"Added {selected_stock} ${strike} put to paper trading simulation!")
    else:
        st.info(f"No put options with delta < 0.1 found for {selected_stock} expiring {selected_expiry}")

# Portfolio Optimization page
elif page == "Portfolio Optimization":
    # Get risk tolerance from sidebar
    risk_tolerance = st.sidebar.radio(
        "Risk Tolerance",
        ["Low", "Medium", "High"],
        index=1  # Default to medium
    )
    
    # Display portfolio optimization
    display_portfolio_optimization(STOCKS, risk_tolerance=risk_tolerance.lower())

# Advanced Risk Analysis page
elif page == "Advanced Risk Analysis":
    display_risk_analysis(STOCKS)

# Paper Trading page
elif page == "Paper Trading":
    st.header("Paper Trading Simulation")
    
    # Initialize or get paper trading simulator
    simulator = PaperTradingSimulator(
        initial_balance=st.session_state.account_balance,
        trades=st.session_state.paper_trades
    )
    
    # Display account summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Account Balance", f"${simulator.balance:.2f}")
    with col2:
        st.metric("Open Positions", len([t for t in simulator.trades if t['status'] == 'open']))
    with col3:
        st.metric("Closed Positions", len([t for t in simulator.trades if t['status'] != 'open']))
    
    # Tabs for open and closed positions
    tab1, tab2 = st.tabs(["Open Positions", "Closed Positions"])
    
    # Open positions
    with tab1:
        open_trades = [t for t in simulator.trades if t['status'] == 'open']
        if open_trades:
            open_df = pd.DataFrame(open_trades)
            st.dataframe(open_df)
            
            # Action buttons for each open trade
            st.subheader("Position Actions")
            for i, trade in enumerate(open_trades):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{trade['stock']} ${trade['strike']} Put**")
                with col2:
                    if st.button("Close Position", key=f"close_{i}"):
                        simulator.close_position(i)
                        st.success(f"Closed position on {trade['stock']} ${trade['strike']} put")
                with col3:
                    current_price = fetch_stock_data(trade['stock'])['Close'].iloc[-1]
                    trade_pnl = simulator.calculate_pnl(trade, current_price)
                    st.metric("Current P&L", f"${trade_pnl:.2f}")
        else:
            st.info("No open positions in your paper trading account.")
    
    # Closed positions
    with tab2:
        closed_trades = [t for t in simulator.trades if t['status'] != 'open']
        if closed_trades:
            closed_df = pd.DataFrame(closed_trades)
            st.dataframe(closed_df)
        else:
            st.info("No closed positions in your paper trading history.")
    
    # Update session state
    st.session_state.paper_trades = simulator.trades
    st.session_state.account_balance = simulator.balance