#risk_simulation.py
import numpy as np

def run_monte_carlo_simulation(stock_price, days, volatility, num_simulations=1000, seed=None):
    """
    Run a Monte Carlo simulation for stock price movements
    
    Parameters:
    stock_price (float): Current stock price
    days (int): Number of days to simulate
    volatility (float): Annual volatility as a decimal
    num_simulations (int): Number of simulations to run
    seed (int): Random seed for reproducibility
    
    Returns:
    ndarray: Simulated price paths (num_simulations x days)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Daily volatility
    daily_vol = volatility / np.sqrt(252)
    
    # Daily returns are normally distributed with mean 0 and std = daily_vol
    daily_returns = np.random.normal(0, daily_vol, (num_simulations, days))
    
    # Calculate price paths
    price_paths = np.zeros((num_simulations, days + 1))
    price_paths[:, 0] = stock_price
    
    for t in range(1, days + 1):
        price_paths[:, t] = price_paths[:, t-1] * np.exp(daily_returns[:, t-1])
    
    return price_paths