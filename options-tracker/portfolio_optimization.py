#portfolio_optimization.py
import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

from stock_data import fetch_stock_data, calculate_historical_volatility
from option_analysis import get_option_chain, filter_low_delta_puts
from paper_trading import calculate_put_option_greeks

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
            st.warning(f"Could not fetch data for {ticker}: {str(e)}")
    
    # Create a dataframe with all closing prices
    if all_data:
        prices_df = pd.DataFrame(all_data)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    return pd.DataFrame()

def calculate_sharpe_ratio(returns, risk_free_rate=0.035):
    """
    Calculate the Sharpe ratio for a series of returns
    
    Parameters:
    returns (Series): Series of percentage returns
    risk_free_rate (float): Annual risk-free rate (default: 3.5%)
    
    Returns:
    float: Annualized Sharpe ratio
    """
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate Sharpe ratio
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    return sharpe

def optimize_put_portfolio(stocks, max_positions=5, max_per_stock=2, total_capital=100000, risk_tolerance='medium'):
    """
    Optimize a portfolio of naked put positions
    
    Parameters:
    stocks (list): List of stock tickers
    max_positions (int): Maximum number of positions to recommend
    max_per_stock (int): Maximum positions per individual stock
    total_capital (float): Total capital available for the portfolio
    risk_tolerance (str): Risk tolerance ('low', 'medium', 'high')
    
    Returns:
    DataFrame: Optimized portfolio of put options
    dict: Portfolio metrics
    """
    # Define risk parameters based on tolerance
    if risk_tolerance.lower() == 'low':
        max_delta = 0.05
        max_capital_per_position = 0.15  # Max 15% of capital per position
        min_days_to_expiry = 30
        max_days_to_expiry = 45
    elif risk_tolerance.lower() == 'medium':
        max_delta = 0.10
        max_capital_per_position = 0.20  # Max 20% of capital per position
        min_days_to_expiry = 45
        max_days_to_expiry = 60
    else:  # high
        max_delta = 0.15
        max_capital_per_position = 0.25  # Max 25% of capital per position
        min_days_to_expiry = 15
        max_days_to_expiry = 90
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(stocks)
    
    # Get all available put options for each stock
    all_puts = []
    for ticker in stocks:
        try:
            # Get current stock data
            stock_data = fetch_stock_data(ticker)
            current_price = stock_data['Close'].iloc[-1]
            
            # Get option chains
            option_chain = get_option_chain(ticker)
            
            # Process each expiration date
            for expiry, puts_df in option_chain.items():
                # Calculate days to expiry
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                days_to_expiry = (expiry_date - datetime.now()).days
                
                # Check if within desired expiration range
                if days_to_expiry < min_days_to_expiry or days_to_expiry > max_days_to_expiry:
                    continue
                
                # Filter for low delta puts
                if 'delta' in puts_df.columns:
                    # Filter by delta
                    filtered_puts = puts_df[puts_df['delta'].abs() < max_delta].copy()
                    
                    for _, row in filtered_puts.iterrows():
                        strike = row['strike']
                        premium = row['lastPrice']
                        delta = row['delta'].abs() if 'delta' in row else None
                        
                        # If delta is not available, estimate it
                        if delta is None:
                            try:
                                greeks = calculate_put_option_greeks(
                                    current_price, 
                                    strike, 
                                    days_to_expiry
                                )
                                delta = greeks['delta'].abs()
                            except:
                                delta = 0.5  # Default if calculation fails
                        
                        # Calculate capital requirement (100 shares per contract)
                        capital_required = strike * 100
                        
                        # Check if position fits within capital constraints
                        if capital_required > max_capital_per_position * total_capital:
                            continue
                        
                        # Calculate annualized return
                        annualized_return = (premium / strike) * (365 / days_to_expiry) * 100
                        
                        # Calculate risk metrics
                        distance_pct = ((strike - current_price) / current_price) * 100
                        
                        # Add to list of available puts
                        all_puts.append({
                            'ticker': ticker,
                            'expiry': expiry,
                            'days_to_expiry': days_to_expiry,
                            'strike': strike,
                            'premium': premium,
                            'delta': delta,
                            'capital_required': capital_required,
                            'annualized_return': annualized_return,
                            'distance_pct': distance_pct,
                            'current_price': current_price
                        })
        except Exception as e:
            st.warning(f"Error processing options for {ticker}: {str(e)}")
    
    # Convert to DataFrame
    if not all_puts:
        return pd.DataFrame(), {}
    
    puts_df = pd.DataFrame(all_puts)
    
    # Portfolio optimization
    # Sort by risk-adjusted return (return / delta)
    puts_df['risk_adjusted_return'] = puts_df['annualized_return'] / puts_df['delta']
    
    # Start with the highest risk-adjusted returns
    sorted_puts = puts_df.sort_values('risk_adjusted_return', ascending=False)
    
    # Initialize portfolio
    portfolio = []
    total_capital_used = 0
    stock_position_count = {}
    
    # Greedy algorithm with constraints
    for _, position in sorted_puts.iterrows():
        ticker = position['ticker']
        
        # Check if we already have max positions
        if len(portfolio) >= max_positions:
            break
        
        # Check if we already have max positions for this stock
        if ticker in stock_position_count and stock_position_count[ticker] >= max_per_stock:
            continue
        
        # Check if we have enough capital
        if total_capital_used + position['capital_required'] > total_capital:
            continue
        
        # Check correlation with existing positions
        if portfolio:
            # Get existing tickers in portfolio
            existing_tickers = [p['ticker'] for p in portfolio]
            
            # Check average correlation
            if ticker in corr_matrix.index:
                avg_corr = np.mean([corr_matrix.loc[ticker, t] for t in existing_tickers if t in corr_matrix.columns])
                
                # If highly correlated (>0.7), skip unless high risk tolerance
                if avg_corr > 0.7 and risk_tolerance.lower() != 'high':
                    continue
        
        # Add position to portfolio
        portfolio.append(position.to_dict())
        
        # Update running totals
        total_capital_used += position['capital_required']
        stock_position_count[ticker] = stock_position_count.get(ticker, 0) + 1
    
    # Convert portfolio to DataFrame
    if portfolio:
        portfolio_df = pd.DataFrame(portfolio)
        
        # Calculate portfolio metrics
        portfolio_metrics = {
            'total_positions': len(portfolio_df),
            'total_capital_used': portfolio_df['capital_required'].sum(),
            'capital_utilization': portfolio_df['capital_required'].sum() / total_capital * 100,
            'average_annualized_return': portfolio_df['annualized_return'].mean(),
            'average_delta': portfolio_df['delta'].mean(),
            'weighted_average_days_to_expiry': np.average(
                portfolio_df['days_to_expiry'], 
                weights=portfolio_df['capital_required']
            ),
            'stocks_used': portfolio_df['ticker'].nunique(),
            'max_drawdown_risk': portfolio_df['capital_required'].sum() - portfolio_df['premium'].sum() * 100
        }
        
        return portfolio_df, portfolio_metrics
    
    return pd.DataFrame(), {}

def display_portfolio_optimization(stocks, total_capital=100000, risk_tolerance='medium'):
    """
    Display portfolio optimization results in Streamlit
    
    Parameters:
    stocks (list): List of stock tickers
    total_capital (float): Total capital available
    risk_tolerance (str): Risk tolerance level
    """
    st.header("Portfolio Optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_capital = st.number_input(
            "Total Capital ($)",
            min_value=10000.0,
            max_value=1000000.0,
            value=float(total_capital),
            step=10000.0
        )
    
    with col2:
        max_positions = st.slider(
            "Maximum Positions",
            min_value=1,
            max_value=15,
            value=5
        )
    
    with col3:
        max_per_stock = st.slider(
            "Max Positions Per Stock",
            min_value=1,
            max_value=5,
            value=2
        )
    
    # Run optimization
    with st.spinner("Optimizing portfolio..."):
        optimized_portfolio, portfolio_metrics = optimize_put_portfolio(
            stocks=stocks,
            max_positions=max_positions,
            max_per_stock=max_per_stock,
            total_capital=selected_capital,
            risk_tolerance=risk_tolerance
        )
    
    # Display results
    if not optimized_portfolio.empty and portfolio_metrics:
        # Display portfolio metrics
        st.subheader("Portfolio Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Positions", portfolio_metrics['total_positions'])
            st.metric("Capital Utilization", f"{portfolio_metrics['capital_utilization']:.2f}%")
        
        with metric_col2:
            st.metric("Avg. Annual Return", f"{portfolio_metrics['average_annualized_return']:.2f}%")
            st.metric("Avg. Delta", f"{portfolio_metrics['average_delta']:.4f}")
        
        with metric_col3:
            st.metric("Stocks Used", portfolio_metrics['stocks_used'])
            st.metric("Avg. Days to Expiry", f"{portfolio_metrics['weighted_average_days_to_expiry']:.0f}")
        
        with metric_col4:
            capital_used = portfolio_metrics['total_capital_used']
            st.metric("Capital Used", f"${capital_used:,.2f}")
            st.metric("Capital Remaining", f"${selected_capital - capital_used:,.2f}")
        
        # Display optimized portfolio
        st.subheader("Recommended Portfolio")
        
        # Format for display
        display_df = optimized_portfolio.copy()
        display_df['strike'] = display_df['strike'].map('${:,.2f}'.format)
        display_df['premium'] = display_df['premium'].map('${:,.2f}'.format)
        display_df['current_price'] = display_df['current_price'].map('${:,.2f}'.format)
        display_df['capital_required'] = display_df['capital_required'].map('${:,.2f}'.format)
        display_df['annualized_return'] = display_df['annualized_return'].map('{:.2f}%'.format)
        display_df['distance_pct'] = display_df['distance_pct'].map('{:.2f}%'.format)
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            'ticker': 'Stock',
            'expiry': 'Expiration',
            'days_to_expiry': 'Days',
            'strike': 'Strike',
            'premium': 'Premium',
            'delta': 'Delta',
            'capital_required': 'Capital Req.',
            'annualized_return': 'Ann. Return',
            'distance_pct': 'Distance',
            'current_price': 'Stock Price'
        })
        
        st.dataframe(display_df)
        
        # Create allocation visualization
        st.subheader("Capital Allocation")
        
        # Pie chart of capital allocation by stock
        fig = px.pie(
            optimized_portfolio, 
            values='capital_required', 
            names='ticker',
            title='Capital Allocation by Stock',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Stock Correlation Matrix")
        
        corr_matrix = calculate_correlation_matrix(optimized_portfolio['ticker'].unique())
        if not corr_matrix.empty:
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title='Correlation Between Selected Stocks'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation insight
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
            if avg_corr > 0.7:
                st.warning(f"⚠️ High average correlation ({avg_corr:.2f}) between stocks. Consider diversifying across different sectors.")
            elif avg_corr < 0.3:
                st.success(f"✅ Low average correlation ({avg_corr:.2f}) between stocks. Good diversification.")
            else:
                st.info(f"ℹ️ Moderate average correlation ({avg_corr:.2f}) between stocks.")
        
        # Explain optimization strategy
        with st.expander("About Portfolio Optimization Strategy"):
            st.markdown("""
            ### Optimization Strategy
            
            This portfolio optimizer uses a constrained optimization approach with the following objectives:
            
            1. **Maximize risk-adjusted returns** (annualized return divided by delta)
            2. **Diversify across different stocks** to reduce correlation risk
            3. **Limit capital allocation** to comply with position size constraints
            4. **Balance expiration dates** to manage timing risk
            
            The algorithm applies the following constraints:
            
            - Maximum number of total positions
            - Maximum positions per individual stock
            - Capital limits per position based on risk tolerance
            - Correlation thresholds between positions
            - Expiration date ranges based on risk tolerance
            
            For a more conservative portfolio, lower the risk tolerance or reduce the maximum positions per stock.
            """)
    else:
        st.warning("Could not generate an optimized portfolio. Try adjusting your parameters or check data availability.")
        
        # Suggestions if optimization fails
        st.markdown("""
        ### Suggestions:
        
        - Increase your capital allocation
        - Increase the maximum number of positions
        - Try a higher risk tolerance setting
        - Check that the selected stocks have available options data
        """)

def calculate_put_portfolio_metrics(portfolio_df):
    """
    Calculate comprehensive metrics for a portfolio of put options
    
    Parameters:
    portfolio_df (DataFrame): Portfolio of put options
    
    Returns:
    dict: Portfolio metrics
    """
    if portfolio_df.empty:
        return {}
    
    # Basic metrics
    total_capital = portfolio_df['capital_required'].sum()
    total_premium = (portfolio_df['premium'] * 100).sum()  # Premium per contract
    
    # Calculate weighted average metrics
    weights = portfolio_df['capital_required'] / total_capital
    
    weighted_avg_delta = np.average(portfolio_df['delta'], weights=weights)
    weighted_avg_days = np.average(portfolio_df['days_to_expiry'], weights=weights)
    weighted_avg_return = np.average(portfolio_df['annualized_return'], weights=weights)
    
    # Calculate expected return
    expected_return = total_premium / total_capital * 100
    
    # Calculate maximum drawdown risk (if all puts are assigned)
    max_drawdown = total_capital - total_premium
    
    # Calculate diversification metrics
    stock_diversification = 1 / (portfolio_df['ticker'].value_counts() / len(portfolio_df)).sum()
    expiry_diversification = 1 / (portfolio_df['expiry'].value_counts() / len(portfolio_df)).sum()
    
    return {
        'total_capital': total_capital,
        'total_premium': total_premium,
        'premium_yield': total_premium / total_capital * 100,
        'expected_return': expected_return,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown / total_capital * 100,
        'weighted_avg_delta': weighted_avg_delta,
        'weighted_avg_days': weighted_avg_days,
        'weighted_avg_return': weighted_avg_return,
        'stock_diversification': stock_diversification,
        'expiry_diversification': expiry_diversification
    }