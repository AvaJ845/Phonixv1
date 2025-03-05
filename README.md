# Phonixv1
# Options Tracker - Naked Puts Scanner

A Streamlit application for tracking and analyzing naked put options with a focus on low delta contracts (< 0.1) for MSFT, CMG, NVDA, Tesla, and V. The app includes paper trading simulation capabilities.

## Features

- **Stock Analysis**: Track and visualize price data for selected stocks
- **Options Chain Analysis**: Filter and identify put options with delta < 0.1
- **Recommendations**: Get suggestions for potential put selling opportunities
- **Paper Trading Simulation**: Test strategies without risking real money
- **Dashboard**: Overview of all available opportunities and market data
- **Portfolio Optimization**: Get optimized portfolio suggestions based on risk tolerance
- **Advanced Risk Analysis**: Monte Carlo simulations, scenario analysis, and Value at Risk calculations

## Installation

1. Clone the repository or download the code files
2. Install the required packages:

```bash
pip install streamlit pandas yfinance numpy plotly scipy
```

Alternatively, install all requirements using the requirements.txt file:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Application Structure

- `app.py`: Main Streamlit application
- `stock_data.py`: Functions for fetching and analyzing stock data
- `option_analysis.py`: Options chain analysis and recommendation engine
- `paper_trading.py`: Paper trading simulation functionality
- `dashboard.py`: Dashboard visualization components
- `portfolio_optimization.py`: Portfolio optimization algorithms and visualizations
- `risk_analysis.py`: Advanced risk analysis tools and simulations

## Usage

1. **Dashboard**: View an overview of all stocks and top opportunities
2. **Stock Analysis**: Analyze individual stock performance and trends
3. **Options Chain**: Explore available put options and filter by delta
4. **Portfolio Optimization**: Get optimized portfolio recommendations based on your risk tolerance and capital
5. **Advanced Risk Analysis**: Perform detailed risk analysis with Monte Carlo simulations and scenario testing
6. **Paper Trading**: Simulate put selling strategies with virtual money

## Paper Trading

The paper trading simulation allows you to:

- Sell put options virtually using real market data
- Track your simulated positions
- Calculate profits and losses
- Test different strategies

## Requirements

- Python 3.7+
- Internet connection for real-time stock and options data

## Notes

- Market data is provided by Yahoo Finance through the `yfinance` package
- Options data may have slight delays depending on the Yahoo Finance API
- The application is designed for educational purposes and strategy testing

## Future Enhancements

- Additional options strategies (covered calls, spreads, etc.)
- Portfolio optimization recommendations
- Advanced risk analysis based on historical performance
- Export/import capabilities for simulation data
