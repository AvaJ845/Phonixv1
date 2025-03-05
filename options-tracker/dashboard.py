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
        for ticker in stocks:
            try:
                options = get_option_chain(ticker)
                total_options += sum(len(df) for df in options.values())
            except:
                pass
        st.metric("Available Options", total_options)
    
    with col3:
        # Count recommended options (delta < 0.1)
        recommended_count = 0
        for ticker in stocks:
            try:
                options = get_option_chain(ticker)
                for expiry, df in options.items():
                    if 'delta' in df.columns:
                        recommended_count += len(df[df['delta'].abs() < 0.1])
            except:
                pass
        st.metric("Recommended Options", recommended_count)
    
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
    
    # Top Put Opportunities Section
    st.subheader("Top Put Selling Opportunities")
    
    # Get recommendations for all stocks
    all_recommendations = []
    for ticker in stocks:
        try:
            options = get_option_chain(ticker)
            if options:
                recommendations = recommend_put_strategies(ticker, options, risk_tolerance.lower())
                if not recommendations.empty:
                    # Take top 2 recommendations per stock
                    all_recommendations.append(recommendations.head(2))
        except Exception as e:
            st.warning(f"Error generating recommendations for {ticker}: {str(e)}")
    
    # Combine and display recommendations
    if all_recommendations:
        combined_recommendations = pd.concat(all_recommendations)
        combined_recommendations = combined_recommendations.sort_values('annualized_return', ascending=False).head(10)
        
        # Format the dataframe
        display_recommendations = combined_recommendations.copy()
        display_recommendations['strike'] = display_recommendations['strike'].map('${:,.2f}'.format)
        display_recommendations['premium'] = display_recommendations['premium'].map('${:,.2f}'.format)
        display_recommendations['current_price'] = display_recommendations['current_price'].map('${:,.2f}'.format)
        display_recommendations['annualized_return'] = display_recommendations['annualized_return'].map('{:.2f}%'.format)
        display_recommendations['distance_pct'] = display_recommendations['distance_pct'].map('{:.2f}%'.format)
        
        # Rename columns for display
        display_recommendations = display_recommendations.rename(columns={
            'ticker': 'Stock',
            'expiry': 'Expiration',
            'strike': 'Strike',
            'premium': 'Premium',
            'delta': 'Delta',
            'days_to_expiry': 'Days',
            'annualized_return': 'Ann. Return',
            'distance_pct': 'Distance',
            'current_price': 'Stock Price'
        })
        
        st.dataframe(display_recommendations)
        
        # Create a bar chart for the annualized returns
        fig = px.bar(
            combined_recommendations,
            x='ticker',
            y='annualized_return',
            color='delta',
            labels={
                'ticker': 'Stock',
                'annualized_return': 'Annualized Return (%)',
                'delta': 'Delta'
            },
            title='Annualized Returns by Stock',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recommendations available at this time. Try adjusting your risk tolerance or check back later.")
    
    # Risk Analysis Section
    st.subheader("Risk Analysis")
    
    # Create a scatter plot of delta vs. annualized return
    if all_recommendations and len(pd.concat(all_recommendations)) > 0:
        scatter_data = pd.concat(all_recommendations)
        fig = px.scatter(
            scatter_data,
            x='delta',
            y='annualized_return',
            color='ticker',
            size='premium',
            hover_data=['strike', 'days_to_expiry', 'distance_pct'],
            labels={
                'delta': 'Delta (Lower = Less Risk)',
                'annualized_return': 'Annualized Return (%)',
                'ticker': 'Stock',
                'premium': 'Premium ($)'
            },
            title='Risk-Reward Analysis',
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