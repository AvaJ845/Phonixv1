#dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from stock_data import fetch_stock_data, get_stock_info, calculate_historical_volatility
from option_analysis import get_option_chain, recommend_put_strategies, get_all_low_delta_puts

def create_dashboard(stocks, max_delta=0.1):
    """
    Create the main dashboard view
    
    Parameters:
    stocks (list): List of stock tickers to display
    max_delta (float): Maximum delta value for filtering options
    """
    st.header("Options Dashboard")
    
    # Help and Documentation Section
    with st.expander("📚 How to Use This Application"):
        st.markdown("""
        ## Welcome to the Options Tracker Application
        
        This application helps you find and analyze low delta put options for selling (also known as writing puts). 
        Here's how to use the different sections:
        
        ### 1. Dashboard
        - Overview of all tracked stocks
        - Summary of available low delta options
        - Top opportunities based on risk-adjusted returns
        
        ### 2. Low Delta Opportunities
        - Comprehensive list of all available put options with delta < 0.1
        - Visual comparison of different options
        - Add promising options directly to your paper trading account
        
        ### 3. Stock Analysis
        - Detailed stock price charts and metrics
        - Historical volatility and performance data
        
        ### 4. Options Chain
        - Explore specific expiration dates for each stock
        - Filter for low delta options
        - View detailed metrics on each option contract
        
        ### 5. Portfolio Optimization
        - Get recommendations for a balanced portfolio of put options
        - Adjust for your risk tolerance and capital constraints
        - Visualize portfolio allocation and correlation
        
        ### 6. Advanced Risk Analysis
        - Detailed risk metrics for specific options
        - Monte Carlo simulations to project outcomes
        - Value at Risk (VaR) calculations
        
        ### 7. Paper Trading
        - Simulate selling put options without risking real money
        - Track performance over time
        - Close positions and see P&L
        """)
    
    # Options Education Section
    with st.expander("🎓 Options Trading Terminology"):
        st.markdown("""
        ## Options Trading Key Concepts
        
        ### Naked Put
        A naked put (or short put) is an options strategy where you sell a put option without having a corresponding short position in the underlying stock. When you sell a put option, you receive a premium upfront, and you're obligated to buy the stock at the strike price if the option is exercised. This strategy is used when you're neutral to bullish on a stock and want to generate income.
        
        ### Option Contract
        An options contract gives the holder the right (but not the obligation) to buy or sell an underlying asset at a specified price (strike price) before or at expiration. Each standard equity option contract represents 100 shares of the underlying stock.
        
        ### Strike Price
        The price at which the option holder can buy (for call options) or sell (for put options) the underlying asset. For put options, if the stock price falls below the strike price, the option may be exercised.
        
        ### Premium
        The price paid by the option buyer to the option seller. When selling puts, this is the income you receive upfront.
        
        ### Expiration Date
        The date when the option contract expires. After this date, the contract is no longer valid.
        
        ### Assignment
        When the option holder exercises their option, the seller (you, when selling puts) is "assigned" and must fulfill the obligation. For put options, this means buying the stock at the strike price, regardless of its current market value.
        
        ### In The Money (ITM)
        A put option is in the money when the current stock price is below the strike price. This means the option has intrinsic value, and there's a higher probability of assignment.
        
        ### Out of The Money (OTM)
        A put option is out of the money when the current stock price is above the strike price. These options have no intrinsic value, only extrinsic (time) value.
        
        ### Delta
        Delta measures the rate of change in an option's price relative to a $1 change in the underlying stock price. For put options:
        - Delta ranges from 0 to -1 (we use absolute value 0 to 1 in this app)
        - A delta of -0.50 means the option price will change by approximately $0.50 for every $1 move in the stock
        
        ### Low Delta (< 0.1) Significance
        Put options with delta < 0.1 have:
        - Lower probability (typically <10%) of ending in the money
        - Lower risk of assignment
        - Usually further out of the money (strike price well below current stock price)
        - Lower premium income, but potentially better risk-adjusted returns
        - Good for conservative income strategies
        
        ### Annualized Return
        The projected return on capital if the option expires worthless, expressed as an annual percentage rate to allow comparison between options with different expiration dates.
        
        ### Value at Risk (VaR)
        A statistical measure that quantifies the level of financial risk within a portfolio over a specific time frame. For example, a 95% VaR of $500 means there's a 5% chance of losing $500 or more.
        """)
    
    # Paper Trading Guide Section
    with st.expander("💵 Paper Trading Guide"):
        st.markdown("""
        ## How to Use the Paper Trading Feature
        
        Paper trading allows you to practice options trading strategies without risking real money. It's a great way to test your strategies and gain experience.
        
        ### Getting Started
        1. Navigate to the "Low Delta Opportunities" or "Options Chain" page
        2. Find an option you'd like to sell
        3. Click the "Add to Paper Trading" or "Simulate Selling Put" button
        4. The position will be added to your paper trading account
        
        ### Managing Positions
        1. Go to the "Paper Trading" page to see all your positions
        2. For open positions, you'll see:
           - The stock and strike price
           - Current stock price and P&L
           - Option to close the position
        
        ### Understanding P&L Calculation
        - When selling a put, your maximum profit is the premium received
        - P&L is calculated as: Premium - Max(0, Strike Price - Current Stock Price)
        - If the stock is above the strike at expiration, you keep the full premium
        - If the stock is below the strike, your loss is offset by the premium
        
        ### Account Balance
        - You start with a default balance of $100,000
        - When selling a put, the required capital (strike price × 100) is reserved
        - The premium is added to your available balance
        - When closing a position, the reserved capital plus any P&L is returned to your balance
        
        ### Assignment Simulation
        - If you hold a position to expiration and the stock price is below the strike price, the position is automatically assigned
        - This means you would buy 100 shares of the stock at the strike price
        - In the simulation, this is reflected as a realized loss
        
        ### Tips for Paper Trading
        - Start with a small number of positions to learn the mechanics
        - Try different expiration dates and strike prices to see how they affect risk and return
        - Keep track of your win rate and average P&L
        - Use the Advanced Risk Analysis to understand the risk of each position before opening it
        """)
    
    # Risk tolerance selector
    risk_tolerance = st.radio(
        "Risk Tolerance",
        ["Low", "Medium", "High"],
        horizontal=True,
        index=1  # Default to medium
    )
    
    # Create columns for header metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Stocks Tracked", len(stocks))
    
    with col2:
        # Count total available options
        total_options = 0
        low_delta_count = 0
        for ticker in stocks:
            try:
                options = get_option_chain(ticker)
                # Count total options
                total_puts = sum(len(df) for df in options.values())
                total_options += total_puts
                
                # Count low delta options
                for expiry, df in options.items():
                    if 'delta' in df.columns:
                        low_delta_count += len(df[df['delta'].abs() < max_delta])
            except:
                pass
        st.metric("Available Put Options", total_options)
    
    with col3:
        # Count low delta options (delta < max_delta)
        st.metric(f"Low Delta Puts (< {max_delta})", low_delta_count)
    
    # Stock overview section
    st.subheader("Stock Overview")
    
    # Create a table with stock information
    stock_data = []
    for ticker in stocks:
        try:
            # Get basic stock information
            info = get_stock_info(ticker)
            price_data = fetch_stock_data(ticker)
            current_price = price_data['Close'].iloc[-1]
            prev_price = price_data['Close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Calculate volatility
            volatility = calculate_historical_volatility(ticker)
            
            # Add to stock data list
            stock_data.append({
                'Ticker': ticker,
                'Name': info.get('name', ticker),
                'Price': current_price,
                'Change %': price_change,
                'Volatility %': volatility,
                'Sector': info.get('sector', 'N/A'),
                'Market Cap': info.get('market_cap', 0)
            })
        except Exception as e:
            st.warning(f"Error retrieving data for {ticker}: {str(e)}")
    
    # Create DataFrame from stock data
    if stock_data:
        stock_df = pd.DataFrame(stock_data)
        
        # Format the dataframe
        stock_df['Price'] = stock_df['Price'].map('${:,.2f}'.format)
        stock_df['Change %'] = stock_df['Change %'].map('{:+.2f}%'.format)
        stock_df['Volatility %'] = stock_df['Volatility %'].map('{:.2f}%'.format)
        stock_df['Market Cap'] = stock_df['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.2f}M")
        
        # Display the dataframe
        st.dataframe(stock_df)
    else:
        st.error("No stock data available. Please check your internet connection.")
    
    # Top Low Delta Put Opportunities Section
    st.subheader(f"Top Low Delta Put Opportunities (Delta < {max_delta})")
    
    # Get all low delta puts for all stocks
    all_low_delta_puts = []
    for ticker in stocks:
        try:
            low_delta_options = get_all_low_delta_puts(ticker, max_delta=max_delta)
            if not low_delta_options.empty:
                all_low_delta_puts.append(low_delta_options)
        except Exception as e:
            st.warning(f"Error getting low delta puts for {ticker}: {str(e)}")
    
    # Combine and display recommendations
    if all_low_delta_puts:
        combined_options = pd.concat(all_low_delta_puts)
        
        # Calculate risk-adjusted return (return / delta)
        combined_options['risk_adjusted_return'] = combined_options['annualized_return'] / combined_options['delta']
        
        # Sort by risk-adjusted return and take top 10
        top_opportunities = combined_options.sort_values('risk_adjusted_return', ascending=False).head(10)
        
        # Format for display
        display_opportunities = top_opportunities.copy()
        display_opportunities['strike'] = display_opportunities['strike'].map('${:,.2f}'.format)
        display_opportunities['lastPrice'] = display_opportunities['lastPrice'].map('${:,.2f}'.format)
        display_opportunities['current_price'] = display_opportunities['current_price'].map('${:,.2f}'.format)
        display_opportunities['annualized_return'] = display_opportunities['annualized_return'].map('{:.2f}%'.format)
        display_opportunities['distance_pct'] = display_opportunities['distance_pct'].map('{:.2f}%'.format)
        display_opportunities['risk_adjusted_return'] = display_opportunities['risk_adjusted_return'].map('{:.2f}'.format)
        
        # Select columns to display
        columns_to_display = [
            'ticker', 'expiration', 'strike', 'delta', 'lastPrice', 
            'days_to_expiry', 'annualized_return', 'distance_pct', 'risk_adjusted_return'
        ]
        
        # Rename columns for display
        display_opportunities = display_opportunities[columns_to_display].rename(columns={
            'ticker': 'Stock',
            'expiration': 'Expiration',
            'strike': 'Strike',
            'lastPrice': 'Premium',
            'days_to_expiry': 'Days',
            'annualized_return': 'Ann. Return',
            'distance_pct': 'Distance',
            'risk_adjusted_return': 'Risk-Adjusted'
        })
        
        st.dataframe(display_opportunities)
        
        # Create a bar chart for the risk-adjusted returns
        fig = px.bar(
            top_opportunities,
            x='ticker',
            y='risk_adjusted_return',
            color='delta',
            labels={
                'ticker': 'Stock',
                'risk_adjusted_return': 'Risk-Adjusted Return',
                'delta': 'Delta (Lower is Better)'
            },
            title='Top Opportunities by Risk-Adjusted Return',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No low delta put opportunities (delta < {max_delta}) found. Try adjusting your risk tolerance or check back later.")
    
    # Risk Analysis Section
    st.subheader("Risk vs. Return Analysis")
    
    # Create a scatter plot of delta vs. annualized return if we have data
    if all_low_delta_puts and len(all_low_delta_puts) > 0:
        scatter_data = pd.concat(all_low_delta_puts)
        fig = px.scatter(
            scatter_data,
            x='delta',
            y='annualized_return',
            color='ticker',
            size='lastPrice',
            hover_data=['strike', 'days_to_expiry', 'distance_pct'],
            labels={
                'delta': 'Delta (Lower = Less Risk)',
                'annualized_return': 'Annualized Return (%)',
                'ticker': 'Stock',
                'lastPrice': 'Premium ($)'
            },
            title='Risk-Reward Analysis for Low Delta Puts',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add some risk analysis insights
        st.markdown("""
        ### Risk Insights
        
        - **Lower delta** options have less risk of being assigned but offer lower premiums
        - **Higher annualized returns** often come with higher risk
        - **Longer expiration dates** typically have higher total premium but lower annualized returns
        - **Higher implied volatility** stocks offer higher premiums but with more price movement risk
        """)
    else:
        st.info("Insufficient data for risk analysis visualization.")
        
    # Trading Strategy Guide
    with st.expander("📝 Low Delta Put Selling Strategy Guide"):
        st.markdown("""
        ## Low Delta Put Selling Strategy Guide
        
        ### Strategy Overview
        Selling low delta puts (delta < 0.1) is a conservative options strategy focusing on:
        - High probability of profit (typically >90%)
        - Lower risk of assignment
        - Consistent income generation
        - Lower margin requirements
        
        ### When to Use This Strategy
        - In sideways or slightly bullish markets
        - On stocks you wouldn't mind owning at the strike price
        - As a way to generate extra income with lower risk than higher delta strategies
        - When implied volatility is higher than historical volatility
        
        ### Risk Management Best Practices
        1. **Position Sizing**: Limit each position to 3-5% of your portfolio
        2. **Diversification**: Spread positions across different stocks and sectors
        3. **Strike Selection**: Choose strikes with at least 15-20% distance from current price
        4. **Duration**: 30-45 days is optimal for many traders (balances time decay with risk)
        5. **Take Profit Early**: Consider closing positions at 50-75% of maximum profit
        6. **Manage Losers**: Have an exit plan if the stock drops toward your strike price
        
        ### Performance Expectations
        - **Win Rate**: Typically 85-95% with disciplined delta selection
        - **Return on Capital**: Usually 0.5-2% per month (6-24% annualized)
        - **Standard Deviation**: Lower than with higher delta strategies
        
        ### Common Mistakes to Avoid
        - Selling puts on high-volatility stocks without proper position sizing
        - Not accounting for upcoming earnings or major announcements
        - Ignoring overall market conditions and correlation with your positions
        - Over-leveraging your account
        - Holding losing positions too long
        
        ### Measuring Success
        Track these metrics to evaluate your strategy:
        - Win rate (% of trades that are profitable)
        - Average return on capital
        - Standard deviation of returns
        - Maximum drawdown
        """)

def get_help_content(section):
    """
    Returns help content for different sections of the app
    
    Parameters:
    section (str): Section name to get help for
    
    Returns:
    str: Markdown formatted help content
    """
    help_content = {
        "basics": """
        ## Options Basics
        
        - **Put Option**: A contract giving the holder the right to sell 100 shares at the strike price
        - **Naked Put**: Selling a put option without holding a corresponding short position
        - **Premium**: The price paid for an option, representing your maximum profit when selling puts
        - **Strike Price**: The price at which the option holder can sell the stock if exercised
        - **Expiration Date**: The date when the option contract expires
        """,
        
        "delta": """
        ## Understanding Delta
        
        Delta measures the rate of change in an option's price relative to a change in the underlying stock price.
        
        For put options:
        - Delta ranges from 0 to -1 (shown as absolute values 0 to 1 in this app)
        - **Delta < 0.1** indicates:
          - Low probability (<10%) of ending in-the-money
          - Lower risk of assignment
          - Further out-of-the-money (OTM)
          - Lower premium but potentially better risk-adjusted return
        """,
        
        "paper_trading": """
        ## Paper Trading Guide
        
        1. **Find Opportunities**: Browse the "Low Delta Opportunities" or "Options Chain" page
        2. **Add to Paper Trading**: Click the button to simulate selling a put
        3. **Monitor Positions**: Go to the "Paper Trading" page to see your positions
        4. **Close Positions**: Close positions anytime or let them expire
        5. **Track Performance**: Monitor your P&L and success rate
        """
    }
    
    return help_content.get(section, "Help content not found for this section.")