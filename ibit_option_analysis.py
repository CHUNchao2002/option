#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Analysis Script with Visualization
This script analyzes ETF options, focusing on call options within ±5% of current price
and calculates potential profit/loss for a 10% price increase scenario.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Suppress warnings
warnings.filterwarnings('ignore')

def get_current_price(ticker):
    """Get the current market price for a ticker symbol"""
    try:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period='1d')
        if len(todays_data) == 0:
            print(f"No price data found for {ticker}. Please check if the ticker symbol is correct.")
            return None
        return todays_data['Close'].iloc[-1]
    except Exception as e:
        print(f"Error getting price for {ticker}: {e}")
        return None

def get_expiration_dates(ticker):
    """Get all available option expiration dates for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        options = stock.options
        if not options:
            print(f"No options data available for {ticker}.")
        return options
    except Exception as e:
        print(f"Error getting option dates for {ticker}: {e}")
        return []

def find_target_expiry_dates(expiry_dates, target_days=[90, 180]):
    """Find expiration dates closest to the target days from now"""
    if not expiry_dates:
        return []
        
    today = datetime.now().date()
    
    # Convert string dates to datetime objects
    expiry_datetime = [datetime.strptime(date, '%Y-%m-%d').date() for date in expiry_dates]
    
    # Calculate days to expiry for each date
    days_to_expiry = [(date - today).days for date in expiry_datetime]
    
    # Find closest dates to targets
    target_expiry = []
    for target in target_days:
        closest_idx = np.argmin(np.abs(np.array(days_to_expiry) - target))
        target_expiry.append(expiry_dates[closest_idx])
        print(f"Target: {target} days, Selected: {expiry_dates[closest_idx]} "
              f"({days_to_expiry[closest_idx]} days from now)")
    
    return target_expiry

def filter_options_by_strike(options_chain, current_price, pct_range=0.05):
    """Filter options by strike price within a percentage range of current price"""
    lower_bound = current_price * (1 - pct_range)
    upper_bound = current_price * (1 + pct_range)
    
    filtered = options_chain[(options_chain['strike'] >= lower_bound) & 
                            (options_chain['strike'] <= upper_bound)]
    return filtered

def calculate_profit_loss(options, current_price, price_increase_pct=0.10):
    """Calculate potential profit/loss for options at expiry with given price increase"""
    future_price = current_price * (1 + price_increase_pct)
    
    # Calculate P/L for each option
    options['future_price'] = future_price
    options['intrinsic_value_at_expiry'] = np.maximum(0, future_price - options['strike'])
    options['profit_loss'] = options['intrinsic_value_at_expiry'] - options['ask']
    options['roi_pct'] = (options['profit_loss'] / options['ask']) * 100
    
    return options

def plot_profit_loss(results_dict, ticker, current_price, price_increase_pct=0.10):
    """Create visualizations for the options analysis results"""
    if not results_dict:
        print("No data available for visualization.")
        return
    
    future_price = current_price * (1 + price_increase_pct)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f'{ticker} Options Analysis - Current Price: ${current_price:.2f}, Projected Price: ${future_price:.2f}', 
                 fontsize=16, y=0.98)
    
    # Define grid layout
    gs = fig.add_gridspec(3, 2)
    
    # Plot 1: ROI vs Strike Price for all expiry dates
    ax1 = fig.add_subplot(gs[0, :])
    for expiry, data in results_dict.items():
        if not data.empty:
            ax1.plot(data['strike'], data['roi_pct'], 'o-', label=f'Expiry: {expiry}')
    
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax1.set_title('Return on Investment (ROI) by Strike Price')
    ax1.set_xlabel('Strike Price ($)')
    ax1.set_ylabel('ROI (%)')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Profit/Loss vs Strike Price
    ax2 = fig.add_subplot(gs[1, 0])
    for expiry, data in results_dict.items():
        if not data.empty:
            ax2.plot(data['strike'], data['profit_loss'], 'o-', label=f'Expiry: {expiry}')
    
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_title('Profit/Loss by Strike Price')
    ax2.set_xlabel('Strike Price ($)')
    ax2.set_ylabel('Profit/Loss ($)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Option Premium vs Strike Price
    ax3 = fig.add_subplot(gs[1, 1])
    for expiry, data in results_dict.items():
        if not data.empty:
            ax3.plot(data['strike'], data['ask'], 'o-', label=f'Expiry: {expiry}')
    
    ax3.set_title('Option Premium by Strike Price')
    ax3.set_xlabel('Strike Price ($)')
    ax3.set_ylabel('Option Premium ($)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Implied Volatility vs Strike Price
    ax4 = fig.add_subplot(gs[2, 0])
    for expiry, data in results_dict.items():
        if not data.empty:
            ax4.plot(data['strike'], data['impliedVolatility'] * 100, 'o-', label=f'Expiry: {expiry}')
    
    ax4.set_title('Implied Volatility by Strike Price')
    ax4.set_xlabel('Strike Price ($)')
    ax4.set_ylabel('Implied Volatility (%)')
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Profit/Loss vs ROI (Scatter)
    ax5 = fig.add_subplot(gs[2, 1])
    for expiry, data in results_dict.items():
        if not data.empty:
            scatter = ax5.scatter(data['roi_pct'], data['profit_loss'], 
                         label=f'Expiry: {expiry}', 
                         alpha=0.7, s=50)
            
            # Add strike price annotations to a few points
            for i, (x, y, strike) in enumerate(zip(data['roi_pct'], data['profit_loss'], data['strike'])):
                if i % 3 == 0:  # Annotate every 3rd point to avoid clutter
                    ax5.annotate(f'${strike:.0f}', (x, y), xytext=(5, 5), 
                                textcoords='offset points', fontsize=8)
    
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax5.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax5.set_title('Profit/Loss vs ROI')
    ax5.set_xlabel('ROI (%)')
    ax5.set_ylabel('Profit/Loss ($)')
    ax5.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    plt.tight_layout()
    
    # Save the figure
    filename = f"{ticker}_options_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as {filename}")
    
    # Show the plot
    plt.show()

def plot_payoff_diagram(results_dict, ticker, current_price, price_range_pct=0.2):
    """Create payoff diagrams for the options at different price points"""
    if not results_dict:
        print("No data available for visualization.")
        return
    
    # Create price points from -20% to +20% of current price
    price_points = np.linspace(current_price * (1 - price_range_pct), 
                              current_price * (1 + price_range_pct), 
                              100)
    
    # Create a figure with subplots (one for each expiry date)
    num_expiry = len(results_dict)
    fig, axes = plt.subplots(num_expiry, 1, figsize=(12, 5 * num_expiry))
    
    # Handle the case where there's only one expiry date
    if num_expiry == 1:
        axes = [axes]
    
    fig.suptitle(f'{ticker} Option Payoff Diagrams at Expiry', fontsize=16, y=0.98)
    
    # For each expiry date
    for i, (expiry, data) in enumerate(results_dict.items()):
        ax = axes[i]
        
        if data.empty:
            ax.text(0.5, 0.5, f'No data available for {expiry}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            continue
        
        # Select a subset of strikes to avoid overcrowding the plot
        if len(data) > 5:
            # Take evenly spaced samples
            step = len(data) // 5
            data_subset = data.iloc[::step]
            if len(data_subset) < 5:  # Ensure we have at least a few lines
                data_subset = data.iloc[::max(1, len(data) // 5)]
        else:
            data_subset = data
        
        # Calculate payoff for each option at different price points
        for idx, option in data_subset.iterrows():
            strike = option['strike']
            premium = option['ask']
            
            # Calculate payoff at each price point
            payoffs = [max(0, price - strike) - premium for price in price_points]
            
            # Plot the payoff line
            ax.plot(price_points, payoffs, label=f'Strike: ${strike:.2f}, Premium: ${premium:.2f}')
        
        # Add reference lines
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=current_price, color='g', linestyle='--', alpha=0.5, 
                  label=f'Current Price: ${current_price:.2f}')
        
        # Set labels and title
        ax.set_title(f'Expiry Date: {expiry}')
        ax.set_xlabel('Stock Price at Expiry ($)')
        ax.set_ylabel('Profit/Loss ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save the figure
    filename = f"{ticker}_payoff_diagrams.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Payoff diagrams saved as {filename}")
    
    # Show the plot
    plt.show()

def main():
    # Set default ticker symbol and allow command line override
    default_ticker = "IBIT"
    ticker = sys.argv[1] if len(sys.argv) > 1 else default_ticker
    
    print(f"\n{'='*60}")
    print(f"{ticker} OPTIONS ANALYSIS")
    print(f"{'='*60}")
    
    # Get current price
    print(f"\nFetching current price for {ticker}...")
    current_price = get_current_price(ticker)
    
    if current_price is None:
        print(f"\nUnable to proceed without valid price data for {ticker}.")
        print(f"Try running with a different ticker: python3 {sys.argv[0]} SPY")
        return
    
    print(f"Current price of {ticker}: ${current_price:.2f}")
    
    # Get expiration dates
    print("\nFetching option expiration dates...")
    expiry_dates = get_expiration_dates(ticker)
    
    if not expiry_dates:
        print(f"No options data available for {ticker}. This could be because:")
        print(f"1. {ticker} may not have options trading available")
        print("2. There might be an issue with the data provider")
        print("3. The ticker symbol might be incorrect")
        print(f"\nTry running with a different ticker: python3 {sys.argv[0]} SPY")
        return
    
    # Find target expiry dates (closest to 90 and 180 days)
    print("\nSelecting target expiration dates:")
    target_expiry = find_target_expiry_dates(expiry_dates)
    
    if not target_expiry:
        print(f"Could not find suitable expiration dates for {ticker}.")
        return
    
    # Calculate strike price range (±5%)
    lower_bound = current_price * 0.95
    upper_bound = current_price * 1.05
    print(f"\nFiltering options with strike prices between ${lower_bound:.2f} and ${upper_bound:.2f}")
    
    # Store results for visualization
    results_dict = {}
    
    # Process each target expiration date
    for expiry in target_expiry:
        print(f"\n{'-'*60}")
        print(f"ANALYSIS FOR EXPIRATION DATE: {expiry}")
        print(f"{'-'*60}")
        
        try:
            # Get options chain for calls
            options = yf.Ticker(ticker).option_chain(expiry)
            calls = options.calls
            
            # Filter by strike price
            filtered_calls = filter_options_by_strike(calls, current_price)
            
            if filtered_calls.empty:
                print(f"No call options found within the strike price range for {expiry}")
                results_dict[expiry] = pd.DataFrame()
                continue
            
            # Calculate profit/loss assuming 10% price increase
            result = calculate_profit_loss(filtered_calls, current_price)
            
            # Store results for visualization
            results_dict[expiry] = result
            
            # Display results
            print(f"\nAssuming {ticker} price increases by 10% to ${current_price * 1.1:.2f} at expiry:")
            
            # Format the output
            display_cols = ['strike', 'ask', 'impliedVolatility', 'intrinsic_value_at_expiry', 
                            'profit_loss', 'roi_pct']
            
            formatted_result = result[display_cols].copy()
            formatted_result['impliedVolatility'] = formatted_result['impliedVolatility'] * 100
            
            # Rename columns for better readability
            formatted_result.columns = ['Strike Price', 'Option Premium', 'Implied Volatility (%)', 
                                       'Value at Expiry', 'Profit/Loss', 'ROI (%)']
            
            print(formatted_result.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        except Exception as e:
            print(f"Error processing options for expiry date {expiry}: {e}")
            results_dict[expiry] = pd.DataFrame()
    
    # Create visualizations
    try:
        print("\nGenerating visualizations...")
        plot_profit_loss(results_dict, ticker, current_price)
        plot_payoff_diagram(results_dict, ticker, current_price)
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main()
