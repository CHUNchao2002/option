# Stock Options Analysis Tool

A comprehensive Python-based toolkit for analyzing stock options and visualizing potential profit/loss scenarios. This project includes both a command-line script and an interactive web application built with Streamlit.

## Overview

This toolkit provides investors with powerful options analysis capabilities:

- **Single Option Analysis**: Analyze individual call and put options with customizable parameters
- **Strategy Comparison**: Compare different options strategies (Long Call, Covered Call, Bull Call Spread, etc.)
- **Interactive Visualizations**: View payoff diagrams, ROI charts, and profit/loss projections
- **Real-time Data**: Fetch current stock prices and options chains using yfinance

## Components

### 1. Command-Line Script (`ibit_option_analysis.py`)

A Python script for quick options analysis from the terminal.

**Features:**
- Fetches current stock price using yfinance
- Identifies option expiration dates closest to 90 and 180 days from today
- Filters call options with strike prices within ±5% of current price
- Calculates potential profit/loss assuming a 10% price increase by expiry
- Displays results including strike price, option premium, implied volatility, and ROI

### 2. Streamlit Web Application (`options_streamlit_app.py`)

An interactive web application with a user-friendly interface for in-depth options analysis.

**Features:**
- Interactive UI with adjustable parameters
- Real-time data fetching for any stock ticker
- Multiple visualization types (payoff diagrams, ROI charts, etc.)
- Advanced options strategy comparison
- Customizable price scenarios and volatility assumptions

### 3. Strategy Comparison Module (`options_strategy_comparison.py`)

A module for comparing different options strategies.

**Supported Strategies:**
- Long Call
- Covered Call
- Bull Call Spread
- Protective Put
- Iron Condor

## Requirements

- Python 3.6+
- Dependencies listed in requirements.txt:
  - yfinance
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - streamlit

## Installation

1. Clone this repository or download the scripts
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Script

Run the script with a ticker symbol (defaults to IBIT if not specified):

```bash
python3 ibit_option_analysis.py [TICKER]
```

Examples:
```bash
python3 ibit_option_analysis.py         # Analyzes IBIT options
python3 ibit_option_analysis.py SPY     # Analyzes SPY options
python3 ibit_option_analysis.py QQQ     # Analyzes QQQ options
```

### Streamlit Web Application

Launch the interactive web application:

```bash
streamlit run options_streamlit_app.py
```

Then open your browser to the URL displayed in the terminal (typically http://localhost:8501).

## Notes and Limitations

- The application uses the "ask" price as an approximation for the option premium
- Calculations are simplified and don't account for factors like theta decay or changes in volatility
- No transaction costs or fees are included in the calculations
- Relies on yfinance for data (potential API changes may affect functionality)
- If a specified ticker doesn't have options data available, try a more common stock like SPY

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## Future Enhancements

- Additional options strategies (Calendar Spreads, Butterfly Spreads, etc.)
- Historical volatility analysis
- Options Greeks calculations and visualization
- Portfolio-level options analysis
- Backtesting capabilities for strategy performance
