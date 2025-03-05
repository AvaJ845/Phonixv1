#risk_visualization.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from risk_simulation import run_monte_carlo_simulation

def display_risk_metrics(risk_metrics):
    """
    Display the risk metrics for an option
    
    Parameters:
    risk_metrics (dict): Dictionary of risk metrics
    """
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Premium", f"${risk_metrics['premium']:.2f}")
        st.metric("Max Loss", f"${risk_metrics['max_loss']:.2f}")
    
    with metric_col2:
        st.metric("Break Even", f"${risk_metrics['break_even']:.2f}")
        st.metric("Probability of Assignment", f"{risk_metrics['assignment_probability']:.2f}%")
    
    with metric_col3:
        st.metric("Delta", f"{risk_metrics['delta']:.4f}")
        st.metric("Theta", f"{risk_metrics['theta']:.4f}")
    
    with metric_col4:
        st.metric("Annualized Return", f"{risk_metrics['annualized_return']:.2f}%")
        st.metric("Expected Value", f"${risk_metrics['expected_value']:.2f}")

def display_pnl_chart(current_price, selected_strike, risk_metrics):
    """
    Display the P&L chart for an option
    
    Parameters:
    current_price (float): Current stock price
    selected_strike (float): Strike price of the option
    risk_metrics (dict): Dictionary of risk metrics
    """
    st.subheader("Option P&L Analysis")
    
    # Create price range for x-axis
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
    
    # Calculate P&L at expiration for each price point
    pnl_at_expiry = [(risk_metrics['premium'] - max(0, selected_strike - price)) * 100 for price in price_range]
    
    # Calculate profit zones
    profit_threshold = selected_strike - risk_metrics['premium']
    
    # Create P&L chart
    fig = go.Figure()
    
    # Add P&L line
    fig.add_trace(go.Scatter(
        x=price_range,
        y=pnl_at_expiry,
        mode='lines',
        name='P&L at Expiration',
        line=dict(color='blue', width=2)
    ))
    
    # Add break-even line
    fig.add_trace(go.Scatter(
        x=[profit_threshold, profit_threshold],
        y=[min(pnl_at_expiry) * 1.1, max(pnl_at_expiry) * 1.1],
        mode='lines',
        name='Break-Even',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    # Add current price line
    fig.add_trace(go.Scatter(
        x=[current_price, current_price],
        y=[min(pnl_at_expiry) * 1.1, max(pnl_at_expiry) * 1.1],
        mode='lines',
        name='Current Price',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    # Add horizontal line at P&L = 0
    fig.add_trace(go.Scatter(
        x=[price_range[0], price_range[-1]],
        y=[0, 0],
        mode='lines',
        name='Break-Even P&L',
        line=dict(color='gray', width=1)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Put Option - P&L at Expiration",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss ($)",
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_scenario_analysis(scenario_df):
    """
    Display scenario analysis for an option
    
    Parameters:
    scenario_df (DataFrame): DataFrame with scenario analysis data
    """
    st.subheader("Scenario Analysis")
    
    # Create pivot table for scenario analysis
    pivot_table = scenario_df.pivot_table(
        values='pnl',
        index='price_change',
        columns='days_passed',
        aggfunc='first'
    )
    
    # Add some formatting to the table
    pivot_styled = pivot_table.style.format("${:,.2f}")
    
    st.write("Profit/Loss ($) by Price Change and Days Passed")
    st.dataframe(pivot_styled)
    
    # Heatmap visualization of scenario analysis
    st.subheader("P&L Heatmap by Scenario")
    
    # Reshape data for heatmap
    heatmap_data = scenario_df.pivot_table(
        values='pnl_pct',
        index='price_change',
        columns='days_passed',
        aggfunc='first'
    )
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(
            x="Days Passed",
            y="Price Change (%)",
            color="P&L (%)"
        ),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='RdYlGn',
        aspect="auto",
        title="P&L (%) by Market Scenario"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_monte_carlo_simulation(current_price, selected_strike, risk_metrics):
    """
    Display Monte Carlo simulation results
    
    Parameters:
    current_price (float): Current stock price
    selected_strike (float): Strike price of the option
    risk_metrics (dict): Dictionary of risk metrics
    """
    st.subheader("Monte Carlo Simulation")
    
    # Run simulation
    num_simulations = 1000
    price_paths = run_monte_carlo_simulation(
        current_price,
        risk_metrics['days_to_expiry'],
        risk_metrics['volatility'] / 100,
        num_simulations
    )
    
    # Calculate final prices and outcomes
    final_prices = price_paths[:, -1]
    
    # Calculate outcomes
    expired_worthless = sum(final_prices >= selected_strike) / num_simulations * 100
    assigned = sum(final_prices < selected_strike) / num_simulations * 100
    
    # Calculate P&L for each path
    pnl_results = np.zeros(num_simulations)
    for i in range(num_simulations):
        if final_prices[i] >= selected_strike:
            # Option expires worthless, full premium is profit
            pnl_results[i] = risk_metrics['premium'] * 100
        else:
            # Option is assigned
            pnl_results[i] = (risk_metrics['premium'] - (selected_strike - final_prices[i])) * 100
    
    # Calculate VaR
    var_95 = np.percentile(pnl_results, 5)  # 95% VaR
    var_99 = np.percentile(pnl_results, 1)  # 99% VaR
    
    # Display simulation results
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.metric("Probability Option Expires Worthless", f"{expired_worthless:.2f}%")
        st.metric("Probability of Assignment", f"{assigned:.2f}%")
    
    with sim_col2:
        st.metric("95% Value at Risk (VaR)", f"${-var_95:.2f}")
        st.metric("99% Value at Risk (VaR)", f"${-var_99:.2f}")
    
    # Plot price paths
    st.subheader("Stock Price Path Simulation")
    
    # Sample paths for visualization (too many makes the chart unreadable)
    sample_indices = np.random.choice(num_simulations, 100, replace=False)
    
    # Create time points for x-axis
    time_points = np.arange(0, risk_metrics['days_to_expiry'] + 1)
    
    # Create the figure
    fig = go.Figure()
    
    # Add sample paths
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=time_points,
            y=price_paths[idx, :],
            mode='lines',
            line=dict(width=0.5, color='rgba(70, 130, 180, 0.2)'),
            showlegend=False
        ))
    
    # Add strike price line
    fig.add_trace(go.Scatter(
        x=[0, risk_metrics['days_to_expiry']],
        y=[selected_strike, selected_strike],
        mode='lines',
        name='Strike Price',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add current price line
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[current_price, current_price],
        mode='lines',
        name='Current Price',
        line=dict(color='green', width=2),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Monte Carlo Simulation ({num_simulations} paths, 100 shown)",
        xaxis_title="Days to Expiration",
        yaxis_title="Stock Price ($)",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # P&L Distribution
    st.subheader("P&L Distribution at Expiration")
    
    # Create histogram of P&L results
    fig = px.histogram(
        pnl_results,
        nbins=50,
        labels={'value': 'P&L ($)'},
        title="P&L Distribution from Monte Carlo Simulation",
        color_discrete_sequence=['blue']
    )
    
    # Add vertical line at mean P&L
    fig.add_vline(
        x=np.mean(pnl_results),
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean: ${np.mean(pnl_results):.2f}",
        annotation_position="top right"
    )
    
    # Add vertical line at VaR
    fig.add_vline(
        x=var_95,
        line_dash="dash",
        line_color="red",
        annotation_text=f"95% VaR: ${-var_95:.2f}",
        annotation_position="top left"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_risk_recommendations(risk_metrics):
    """
    Display risk management recommendations
    
    Parameters:
    risk_metrics (dict): Dictionary of risk metrics
    """
    st.subheader("Risk Management Recommendations")
    
    # Generate recommendations based on the risk analysis
    recommendations = []
    
    # High probability of assignment
    if risk_metrics['assignment_probability'] > 30:
        recommendations.append("⚠️ This option has a high probability of assignment. Consider a further OTM strike or a shorter expiration.")
    
    # Poor risk-reward ratio
    if risk_metrics['risk_reward_ratio'] < 0.2:
        recommendations.append("⚠️ This option has a poor risk-reward ratio. Consider a strike with better premium relative to risk.")
    
    # High volatility
    if risk_metrics['volatility'] > 50:
        recommendations.append("⚠️ The underlying stock has high volatility. Consider using a credit spread to limit downside risk.")
    
    # Close to earnings or dividends
    recommendations.append("ℹ️ Check for upcoming earnings or dividend dates that might affect option pricing.")
    
    # Position sizing
    max_position_size = 5  # Example: Max 5% of portfolio in one position
    recommendations.append(f"ℹ️ Suggested position sizing: Limit this position to {max_position_size}% of your portfolio value.")
    
    # Stop-loss strategy
    stop_loss_pct = 200  # Example: 2x premium
    recommendations.append(f"ℹ️ Consider a stop-loss at {stop_loss_pct}% of premium received to manage risk.")
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(rec)