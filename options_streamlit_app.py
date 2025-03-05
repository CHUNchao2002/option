#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Options Analysis App
A Streamlit web application for analyzing Stock options with adjustable parameters.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from options_strategy_comparison import (
    get_option_strategy_payoff,
    create_strategy_comparison_chart,
    create_strategy_metrics_table,
    find_optimal_strategy_for_outlook,
    get_strategy_descriptions
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Options Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-text {
        font-size: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>Stock Options Analysis Tool</h1>", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("<h2 class='sub-header'>Parameters</h2>", unsafe_allow_html=True)

# Ticker selection
default_ticker = "IBIT"
ticker = st.sidebar.text_input("Stock Ticker Symbol", value=default_ticker).upper()

# Load data button
load_data = st.sidebar.button("Load Stock Data")

# Add iframe under the Load Stock Data button
st.sidebar.markdown("""
<iframe
 src="https://udify.app/chatbot/9bIwPze39OWUkTkp"
 style="width: 100%; height: 100%; min-height: 700px"
 frameborder="0"
 allow="microphone">
</iframe>

## Stock Options Analysis Tool

A comprehensive Python-based toolkit for analyzing stock options and visualizing potential profit/loss scenarios. This project includes both a command-line script and an interactive web application built with Streamlit.

### Overview

This toolkit provides investors with powerful options analysis capabilities:

- **Single Option Analysis**: Analyze individual call and put options with customizable parameters
- **Strategy Comparison**: Compare different options strategies (Long Call, Covered Call, Bull Call Spread, etc.)
- **Interactive Visualizations**: View payoff diagrams, ROI charts, and profit/loss projections
- **Real-time Data**: Fetch current stock prices and options chains using yfinance

### Components

#### 1. Command-Line Script (`ibit_option_analysis.py`)

A Python script for quick options analysis from the terminal.

**Features:**
- Fetches current stock price using yfinance
- Identifies option expiration dates closest to 90 and 180 days from today
- Filters call options with strike prices within ±5% of current price
- Calculates potential profit/loss assuming a 10% price increase by expiry

#### 2. Streamlit Web Application (`options_streamlit_app.py`)

An interactive web application with a user-friendly interface for in-depth options analysis.

**Features:**
- Interactive UI with adjustable parameters
- Real-time data fetching for any stock ticker
- Multiple visualization types (payoff diagrams, ROI charts, etc.)
- Advanced options strategy comparison

#### 3. Strategy Comparison Module

Supported Strategies:
- Long Call
- Covered Call
- Bull Call Spread
- Protective Put
- Iron Condor

### Usage Examples

```bash
python3 ibit_option_analysis.py         # Analyzes IBIT options
python3 ibit_option_analysis.py SPY     # Analyzes SPY options
streamlit run options_streamlit_app.py  # Launch web app
```
""", unsafe_allow_html=True)

# Global variables for data storage
current_price = None
expiry_dates = None
options_data = {}

# Function to get current price
@st.cache_data(ttl=3600)
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period='1d')
        if len(todays_data) == 0:
            st.error(f"No price data found for {ticker}. Please check if the ticker symbol is correct.")
            return None
        return todays_data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error getting price for {ticker}: {e}")
        return None

# Function to get expiration dates
@st.cache_data(ttl=3600)
def get_expiration_dates(ticker):
    try:
        stock = yf.Ticker(ticker)
        options = stock.options
        if not options:
            st.error(f"No options data available for {ticker}.")
        return options
    except Exception as e:
        st.error(f"Error getting option dates for {ticker}: {e}")
        return []

# Function to get options chain
@st.cache_data(ttl=3600)
def get_options_chain(ticker, expiry):
    try:
        stock = yf.Ticker(ticker)
        options = stock.option_chain(expiry)
        return options.calls, options.puts
    except Exception as e:
        st.error(f"Error getting options chain for {ticker} expiry {expiry}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Function to filter options by strike price
def filter_options_by_strike(options_chain, price, pct_range=0.05):
    lower_bound = price * (1 - pct_range)
    upper_bound = price * (1 + pct_range)
    
    filtered = options_chain[(options_chain['strike'] >= lower_bound) & 
                            (options_chain['strike'] <= upper_bound)]
    return filtered

# Function to calculate profit/loss
def calculate_profit_loss(options, price, price_increase_pct=0.10):
    future_price = price * (1 + price_increase_pct)
    
    # Calculate P/L for each option
    options = options.copy()
    options['future_price'] = future_price
    options['intrinsic_value_at_expiry'] = np.maximum(0, future_price - options['strike'])
    options['profit_loss'] = options['intrinsic_value_at_expiry'] - options['ask']
    options['roi_pct'] = (options['profit_loss'] / options['ask']) * 100
    
    return options

# Function to create plotly payoff diagram
def create_payoff_diagram(results, price, price_range_pct=0.3):
    if results.empty:
        return None
    
    # Create price points from -30% to +30% of current price
    price_points = np.linspace(price * (1 - price_range_pct), 
                              price * (1 + price_range_pct), 
                              100)
    
    # Select a subset of strikes to avoid overcrowding the plot
    if len(results) > 5:
        step = len(results) // 5
        data_subset = results.iloc[::step]
        if len(data_subset) < 5:  # Ensure we have at least a few lines
            data_subset = results.iloc[::max(1, len(results) // 5)]
    else:
        data_subset = results
    
    # Create figure
    fig = go.Figure()
    
    # Add payoff lines for each option
    for idx, option in data_subset.iterrows():
        strike = option['strike']
        premium = option['ask']
        
        # Calculate payoff at each price point
        payoffs = [max(0, price_point - strike) - premium for price_point in price_points]
        
        # Add line to plot
        fig.add_trace(go.Scatter(
            x=price_points,
            y=payoffs,
            mode='lines',
            name=f'Strike: ${strike:.2f}',
            hovertemplate='Price: $%{x:.2f}<br>Profit/Loss: $%{y:.2f}'
        ))
    
    # Add reference lines
    fig.add_shape(
        type="line",
        x0=price_points[0],
        y0=0,
        x1=price_points[-1],
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=price,
        y0=min([min([max(0, price_point - option['strike']) - option['ask'] 
                    for price_point in price_points]) for _, option in data_subset.iterrows()]) - 1,
        x1=price,
        y1=max([max([max(0, price_point - option['strike']) - option['ask'] 
                    for price_point in price_points]) for _, option in data_subset.iterrows()]) + 1,
        line=dict(color="green", width=2, dash="dash"),
    )
    
    # Update layout
    fig.update_layout(
        title=f'Option Payoff Diagram at Expiry',
        xaxis_title='Stock Price at Expiry ($)',
        yaxis_title='Profit/Loss ($)',
        legend_title='Options',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Function to create ROI vs Strike chart
def create_roi_strike_chart(results):
    if results.empty:
        return None
    
    fig = px.scatter(
        results,
        x='strike',
        y='roi_pct',
        size='ask',
        color='impliedVolatility',
        hover_data=['ask', 'profit_loss', 'intrinsic_value_at_expiry'],
        labels={
            'strike': 'Strike Price ($)',
            'roi_pct': 'ROI (%)',
            'ask': 'Option Premium ($)',
            'impliedVolatility': 'Implied Volatility',
            'profit_loss': 'Profit/Loss ($)',
            'intrinsic_value_at_expiry': 'Value at Expiry ($)'
        },
        title='Return on Investment (ROI) by Strike Price'
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=results['strike'].min() - 1,
        y0=0,
        x1=results['strike'].max() + 1,
        y1=0,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.update_layout(
        height=500,
        yaxis=dict(ticksuffix="%")
    )
    
    return fig

# Main app logic
if load_data or 'data_loaded' in st.session_state:
    # Store that data has been loaded
    if load_data:
        st.session_state['data_loaded'] = True
    
    # Get current price
    with st.spinner(f"Fetching current price for {ticker}..."):
        current_price = get_current_price(ticker)
    
    if current_price is not None:
        st.markdown(f"<h2 class='sub-header'>Analysis for {ticker}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='info-text'>Current price: <b>${current_price:.2f}</b></p>", unsafe_allow_html=True)
        
        # Get expiration dates
        with st.spinner("Fetching option expiration dates..."):
            expiry_dates = get_expiration_dates(ticker)
        
        if expiry_dates:
            # Allow user to adjust the current price
            adjusted_price = st.slider(
                "Adjust underlying Stock price for analysis:",
                min_value=current_price * 0.7,
                max_value=current_price * 1.3,
                value=current_price,
                step=0.01,
                format="$%.2f"
            )
            
            # Convert dates to datetime objects and calculate days to expiry
            today = datetime.now().date()
            expiry_datetime = [datetime.strptime(date, '%Y-%m-%d').date() for date in expiry_dates]
            days_to_expiry = [(date - today).days for date in expiry_datetime]
            
            # Create a dataframe with expiry dates and days to expiry
            expiry_df = pd.DataFrame({
                'Expiry Date': expiry_dates,
                'Days to Expiry': days_to_expiry
            })
            
            # Allow user to select expiry dates
            st.markdown("<h3 class='sub-header'>Select Expiration Date</h3>", unsafe_allow_html=True)
            
            # Display expiry dates in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<p class='info-text'>Available expiration dates:</p>", unsafe_allow_html=True)
                st.dataframe(expiry_df, height=200)
            
            with col2:
                selected_expiry = st.selectbox(
                    "Choose an expiration date:",
                    options=expiry_dates,
                    index=min(2, len(expiry_dates)-1),  # Default to the third option or last if fewer
                    format_func=lambda x: f"{x} ({(datetime.strptime(x, '%Y-%m-%d').date() - today).days} days)"
                )
            
            # Get options chain for selected expiry
            with st.spinner(f"Fetching options data for {selected_expiry}..."):
                calls, puts = get_options_chain(ticker, selected_expiry)
            
            if not calls.empty:
                # Parameters for analysis
                st.markdown("<h3 class='sub-header'>Analysis Parameters</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    strike_range_pct = st.slider(
                        "Strike price range (±%):",
                        min_value=1,
                        max_value=20,
                        value=5,
                        step=1,
                        format="%d%%"
                    ) / 100
                    
                    option_type = st.radio(
                        "Option type:",
                        options=["Calls", "Puts"],
                        index=0
                    )
                
                with col2:
                    price_change_pct = st.slider(
                        "Assumed price change at expiry:",
                        min_value=-30,
                        max_value=30,
                        value=10 if option_type == "Calls" else -10,
                        step=1,
                        format="%d%%"
                    ) / 100
                
                # Calculate strike price range
                lower_bound = adjusted_price * (1 - strike_range_pct)
                upper_bound = adjusted_price * (1 + strike_range_pct)
                
                st.markdown(f"<p class='info-text highlight'>Filtering options with strike prices between <b>${lower_bound:.2f}</b> and <b>${upper_bound:.2f}</b></p>", unsafe_allow_html=True)
                
                # Filter options by strike price
                options_chain = calls if option_type == "Calls" else puts
                filtered_options = filter_options_by_strike(options_chain, adjusted_price, strike_range_pct)
                
                if filtered_options.empty:
                    st.warning(f"No {option_type.lower()} found within the strike price range for {selected_expiry}")
                else:
                    # Calculate profit/loss
                    results = calculate_profit_loss(filtered_options, adjusted_price, price_change_pct)
                    
                    # Display results
                    st.markdown("<h3 class='sub-header'>Analysis Results</h3>", unsafe_allow_html=True)
                    
                    future_price = adjusted_price * (1 + price_change_pct)
                    st.markdown(f"<p class='info-text'>Assuming {ticker} price {'increases' if price_change_pct > 0 else 'decreases'} by {abs(price_change_pct)*100:.0f}% to <b>${future_price:.2f}</b> at expiry:</p>", unsafe_allow_html=True)
                    
                    # Format the output
                    display_cols = ['strike', 'ask', 'impliedVolatility', 'intrinsic_value_at_expiry', 
                                    'profit_loss', 'roi_pct']
                    
                    formatted_result = results[display_cols].copy()
                    formatted_result['impliedVolatility'] = formatted_result['impliedVolatility'] * 100
                    
                    # Rename columns for better readability
                    formatted_result.columns = ['Strike Price', 'Option Premium', 'Implied Volatility (%)', 
                                               'Value at Expiry', 'Profit/Loss', 'ROI (%)']
                    
                    # Display the table
                    st.dataframe(formatted_result.style.format({
                        'Strike Price': '${:.2f}',
                        'Option Premium': '${:.2f}',
                        'Implied Volatility (%)': '{:.2f}%',
                        'Value at Expiry': '${:.2f}',
                        'Profit/Loss': '${:.2f}',
                        'ROI (%)': '{:.2f}%'
                    }), height=400)
                    
                    # Create visualizations
                    st.markdown("<h3 class='sub-header'>Visualizations</h3>", unsafe_allow_html=True)
                    
                    tab1, tab2 = st.tabs(["Payoff Diagram", "ROI Analysis"])
                    
                    with tab1:
                        payoff_fig = create_payoff_diagram(results, adjusted_price)
                        if payoff_fig:
                            st.plotly_chart(payoff_fig, use_container_width=True)
                    
                    with tab2:
                        roi_fig = create_roi_strike_chart(results)
                        if roi_fig:
                            st.plotly_chart(roi_fig, use_container_width=True)
            else:
                st.error(f"No options data available for {ticker} on {selected_expiry}")
        else:
            st.error(f"No options data available for {ticker}")
else:
    st.info("Enter a ticker symbol and click 'Load Stock Data' to begin analysis.")
    
    # Example section
    st.markdown("<h3 class='sub-header'>Example Tickers with Options</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>Try these popular Stocks:</p>
    <ul>
        <li>IBIT - BlackRock Bitcoin Stock</li>
        <li>SPY - SPDR S&P 500 Stock</li>
        <li>QQQ - Invesco QQQ Trust (Nasdaq-100)</li>
        <li>IWM - iShares Russell 2000 Stock</li>
        <li>EEM - iShares MSCI Emerging Markets Stock</li>
    </ul>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
---
<p style="text-align: center; color: #888;">
    Options Analysis Tool | Data provided by Yahoo Finance | Not financial advice
</p>
""", unsafe_allow_html=True)
