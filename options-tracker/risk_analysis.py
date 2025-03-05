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
from risk_simulation import run_monte_carlo_simulation
from risk_metrics import (
    calculate_put_assignment_probability,
    calculate_max_loss,
    calculate_value_at_risk,
    analyze_portfolio_risk,
    calculate_correlation_matrix
)

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
    
    except Exception as e:
        st.error(f"Error analyzing option risk: {str(e)}")
        return {}, pd.DataFrame()

def display_risk_analysis(stocks):
    """
    Display risk analysis in Streamlit
    
    Parameters:
    stocks (list): List of stock tickers to analyze
    """
    st.header("Advanced Risk Analysis")
    
    # Stock selection
    selected_stock = st.selectbox("Select Stock", stocks)
    
    # Get stock data
    stock_data = fetch_stock_data(selected_stock)
    current_price = stock_data['Close'].iloc[-1]
    
    # Historical volatility
    volatility = calculate_historical_volatility(selected_stock)
    
    # Display basic stock risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Historical Volatility", f"{volatility:.2f}%")
    
    with col3:
        # Calculate beta
        market_data = fetch_stock_data("SPY")
        if not market_data.empty and not stock_data.empty:
            stock_returns = stock_data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align the data
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned_data) > 0:
                # Calculate beta using covariance / variance
                beta = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])[0, 1] / np.var(aligned_data.iloc[:, 1])
                st.metric("Beta", f"{beta:.2f}")
            else:
                st.metric("Beta", "N/A")
        else:
            st.metric("Beta", "N/A")
    
    with col4:
        if not stock_data.empty:
            # Calculate drawdown
            rolling_max = stock_data['Close'].cummax()
            drawdown = (stock_data['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        else:
            st.metric("Max Drawdown", "N/A")
    
    # Option Chain for the selected stock
    option_chain = get_option_chain(selected_stock)
    
    if option_chain:
        # Get expiration dates
        expiry_dates = list(option_chain.keys())
        
        # Expiration date selection
        selected_expiry = st.selectbox("Select Expiration Date", expiry_dates)
        
        # Get put options for selected expiry
        put_options = option_chain[selected_expiry]
        
        # Filter to show options around current price (±30%)
        price_range = (current_price * 0.7, current_price * 1.3)
        filtered_puts = put_options[(put_options['strike'] >= price_range[0]) & 
                                    (put_options['strike'] <= price_range[1])]
        
        # Display available put options
        if not filtered_puts.empty:
            st.subheader(f"Put Options for {selected_stock} expiring on {selected_expiry}")
            
            # Format the display
            display_puts = filtered_puts.copy()
            if 'lastPrice' in display_puts.columns:
                display_puts['lastPrice'] = display_puts['lastPrice'].map('${:,.2f}'.format)
            if 'strike' in display_puts.columns:
                display_puts['strike'] = display_puts['strike'].map('${:,.2f}'.format)
            
            st.dataframe(display_puts)
            
            # Select a specific option for detailed analysis
            strike_options = put_options['strike'].unique()
            selected_strike = st.select_slider(
                "Select Strike Price for Analysis",
                options=sorted(strike_options),
                value=min(strike_options, key=lambda x: abs(x - current_price))
            )
            
            # Perform detailed risk analysis
            st.subheader(f"Detailed Risk Analysis for ${selected_strike} Put")
            
            risk_metrics, scenario_df = analyze_option_risk(
                selected_stock,
                selected_expiry,
                selected_strike
            )
            
            if risk_metrics and not scenario_df.empty:
                from risk_visualization import (
                    display_risk_metrics,
                    display_pnl_chart,
                    display_scenario_analysis,
                    display_monte_carlo_simulation,
                    display_risk_recommendations
                )
                
                # Display risk metrics
                display_risk_metrics(risk_metrics)
                
                # Display P&L chart
                display_pnl_chart(current_price, selected_strike, risk_metrics)
                
                # Display scenario analysis
                display_scenario_analysis(scenario_df)
                
                # Display Monte Carlo simulation
                display_monte_carlo_simulation(
                    current_price, 
                    selected_strike,
                    risk_metrics
                )
                
                # Display risk recommendations
                display_risk_recommendations(risk_metrics)
            else:
                st.error("Unable to perform risk analysis for the selected option.")
        else:
            st.info(f"No put options available for {selected_stock} in the selected price range.")
    else:
        st.warning(f"No option data available for {selected_stock}. Please try another stock.")