#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Strategy Comparison Module
This module provides functionality to compare different options strategies
for ETF analysis in the Streamlit application.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

def get_option_strategy_payoff(strategy_type, options_data, current_price, price_points):
    """
    Calculate the payoff for different option strategies at various price points.
    
    Parameters:
    -----------
    strategy_type : str
        The type of option strategy ('long_call', 'covered_call', 'bull_call_spread', etc.)
    options_data : dict
        Dictionary containing the selected options data for the strategy
    current_price : float
        Current price of the underlying ETF
    price_points : array
        Array of price points to calculate the payoff at
    
    Returns:
    --------
    dict
        Dictionary containing the payoff data and strategy details
    """
    payoffs = []
    strategy_cost = 0
    strategy_description = ""
    
    if strategy_type == "long_call":
        # Long Call: Buy a call option
        call_option = options_data.get('call', None)
        if call_option is None:
            return None
        
        strike = call_option['strike']
        premium = call_option['ask']
        strategy_cost = premium
        
        # Calculate payoff at each price point
        for price in price_points:
            payoff = max(0, price - strike) - premium
            payoffs.append(payoff)
            
        strategy_description = f"Long Call: Buy {call_option['contractSymbol']} at strike ${strike:.2f} for ${premium:.2f}"
    
    elif strategy_type == "covered_call":
        # Covered Call: Own the stock and sell a call option
        call_option = options_data.get('call', None)
        if call_option is None:
            return None
        
        strike = call_option['strike']
        premium = call_option['bid']  # Use bid price when selling
        strategy_cost = current_price - premium
        
        # Calculate payoff at each price point
        for price in price_points:
            payoff = min(price - current_price + premium, strike - current_price + premium)
            payoffs.append(payoff)
            
        strategy_description = f"Covered Call: Buy ETF at ${current_price:.2f} and sell {call_option['contractSymbol']} at strike ${strike:.2f} for ${premium:.2f}"
    
    elif strategy_type == "bull_call_spread":
        # Bull Call Spread: Buy a call option and sell a higher strike call option
        lower_call = options_data.get('lower_call', None)
        higher_call = options_data.get('higher_call', None)
        
        if lower_call is None or higher_call is None:
            return None
        
        lower_strike = lower_call['strike']
        higher_strike = higher_call['strike']
        lower_premium = lower_call['ask']
        higher_premium = higher_call['bid']  # Use bid price when selling
        strategy_cost = lower_premium - higher_premium
        
        # Calculate payoff at each price point
        for price in price_points:
            lower_call_payoff = max(0, price - lower_strike) - lower_premium
            higher_call_payoff = -(max(0, price - higher_strike) - higher_premium)
            payoff = lower_call_payoff + higher_call_payoff
            payoffs.append(payoff)
            
        strategy_description = f"Bull Call Spread: Buy {lower_call['contractSymbol']} at strike ${lower_strike:.2f} for ${lower_premium:.2f} and sell {higher_call['contractSymbol']} at strike ${higher_strike:.2f} for ${higher_premium:.2f}"
    
    elif strategy_type == "protective_put":
        # Protective Put: Own the stock and buy a put option
        put_option = options_data.get('put', None)
        if put_option is None:
            return None
        
        strike = put_option['strike']
        premium = put_option['ask']
        strategy_cost = current_price + premium
        
        # Calculate payoff at each price point
        for price in price_points:
            stock_payoff = price - current_price
            put_payoff = max(0, strike - price) - premium
            payoff = stock_payoff + put_payoff
            payoffs.append(payoff)
            
        strategy_description = f"Protective Put: Buy ETF at ${current_price:.2f} and buy {put_option['contractSymbol']} at strike ${strike:.2f} for ${premium:.2f}"
    
    elif strategy_type == "iron_condor":
        # Iron Condor: Sell a put spread and a call spread
        lower_put = options_data.get('lower_put', None)
        higher_put = options_data.get('higher_put', None)
        lower_call = options_data.get('lower_call', None)
        higher_call = options_data.get('higher_call', None)
        
        if lower_put is None or higher_put is None or lower_call is None or higher_call is None:
            return None
        
        # Put spread
        lower_put_strike = lower_put['strike']
        higher_put_strike = higher_put['strike']
        lower_put_premium = lower_put['bid']  # Selling the lower put
        higher_put_premium = higher_put['ask']  # Buying the higher put
        
        # Call spread
        lower_call_strike = lower_call['strike']
        higher_call_strike = higher_call['strike']
        lower_call_premium = lower_call['bid']  # Selling the lower call
        higher_call_premium = higher_call['ask']  # Buying the higher call
        
        strategy_cost = -(lower_put_premium - higher_put_premium + lower_call_premium - higher_call_premium)
        
        # Calculate payoff at each price point
        for price in price_points:
            # Put spread payoff
            short_put_payoff = -(max(0, higher_put_strike - price) - higher_put_premium)
            long_put_payoff = max(0, lower_put_strike - price) - lower_put_premium
            
            # Call spread payoff
            short_call_payoff = -(max(0, price - lower_call_strike) - lower_call_premium)
            long_call_payoff = max(0, price - higher_call_strike) - higher_call_premium
            
            payoff = short_put_payoff + long_put_payoff + short_call_payoff + long_call_payoff
            payoffs.append(payoff)
            
        strategy_description = f"Iron Condor: Complex strategy with 4 options at different strikes"
    
    return {
        'payoffs': payoffs,
        'strategy_cost': strategy_cost,
        'description': strategy_description
    }

def create_strategy_comparison_chart(strategies_data, price_points, current_price):
    """
    Create a Plotly chart comparing different options strategies.
    
    Parameters:
    -----------
    strategies_data : dict
        Dictionary containing the payoff data for different strategies
    price_points : array
        Array of price points used for the x-axis
    current_price : float
        Current price of the underlying ETF
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the comparison chart
    """
    fig = go.Figure()
    
    # Add a trace for each strategy
    for strategy_name, strategy_data in strategies_data.items():
        if strategy_data is None:
            continue
            
        fig.add_trace(go.Scatter(
            x=price_points,
            y=strategy_data['payoffs'],
            mode='lines',
            name=strategy_name,
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
        x0=current_price,
        y0=min([min(data['payoffs']) for name, data in strategies_data.items() if data is not None]) - 1,
        x1=current_price,
        y1=max([max(data['payoffs']) for name, data in strategies_data.items() if data is not None]) + 1,
        line=dict(color="green", width=2, dash="dash"),
    )
    
    # Update layout
    fig.update_layout(
        title='Options Strategy Comparison',
        xaxis_title='ETF Price at Expiry ($)',
        yaxis_title='Profit/Loss ($)',
        legend_title='Strategies',
        hovermode='x unified',
        height=600
    )
    
    return fig

def create_strategy_metrics_table(strategies_data, price_points, current_price):
    """
    Create a DataFrame with key metrics for each strategy.
    
    Parameters:
    -----------
    strategies_data : dict
        Dictionary containing the payoff data for different strategies
    price_points : array
        Array of price points used for the x-axis
    current_price : float
        Current price of the underlying ETF
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metrics for each strategy
    """
    metrics = []
    
    for strategy_name, strategy_data in strategies_data.items():
        if strategy_data is None:
            continue
            
        payoffs = strategy_data['payoffs']
        cost = strategy_data['strategy_cost']
        
        # Calculate metrics
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        breakeven_indices = np.where(np.diff(np.signbit(payoffs)))[0]
        
        # Find breakeven points
        breakeven_points = []
        for idx in breakeven_indices:
            # Linear interpolation to find the exact breakeven point
            x1, x2 = price_points[idx], price_points[idx + 1]
            y1, y2 = payoffs[idx], payoffs[idx + 1]
            
            if y1 != y2:  # Avoid division by zero
                x_breakeven = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                breakeven_points.append(x_breakeven)
        
        breakeven_str = ", ".join([f"${bp:.2f}" for bp in breakeven_points]) if breakeven_points else "N/A"
        
        # Calculate max ROI
        max_roi = (max_profit / cost) * 100 if cost > 0 else float('inf')
        
        # Calculate probability metrics (simplified)
        profit_probability = sum(1 for p in payoffs if p > 0) / len(payoffs) * 100
        
        metrics.append({
            'Strategy': strategy_name,
            'Initial Cost': f"${cost:.2f}",
            'Max Profit': f"${max_profit:.2f}",
            'Max Loss': f"${max_loss:.2f}",
            'Max ROI': f"{max_roi:.2f}%" if max_roi != float('inf') else "∞",
            'Breakeven Points': breakeven_str,
            'Profit Probability': f"{profit_probability:.1f}%"
        })
    
    return pd.DataFrame(metrics)

def find_optimal_strategy_for_outlook(strategies_data, price_points, outlook):
    """
    Find the optimal strategy based on the market outlook.
    
    Parameters:
    -----------
    strategies_data : dict
        Dictionary containing the payoff data for different strategies
    price_points : array
        Array of price points used for the x-axis
    outlook : str
        Market outlook ('bullish', 'bearish', 'neutral', 'volatile')
    
    Returns:
    --------
    str
        Name of the optimal strategy
    """
    if not strategies_data:
        return None
        
    # Define price ranges based on outlook
    mid_idx = len(price_points) // 2
    
    if outlook == 'bullish':
        # Focus on upper half of price range
        relevant_indices = range(mid_idx, len(price_points))
    elif outlook == 'bearish':
        # Focus on lower half of price range
        relevant_indices = range(0, mid_idx)
    elif outlook == 'neutral':
        # Focus on middle third of price range
        lower_third = len(price_points) // 3
        upper_third = 2 * len(price_points) // 3
        relevant_indices = range(lower_third, upper_third)
    elif outlook == 'volatile':
        # Consider all price points but weight extremes more
        relevant_indices = range(len(price_points))
    else:
        # Default to all price points
        relevant_indices = range(len(price_points))
    
    # Calculate expected value for each strategy based on outlook
    expected_values = {}
    
    for strategy_name, strategy_data in strategies_data.items():
        if strategy_data is None:
            continue
            
        payoffs = [strategy_data['payoffs'][i] for i in relevant_indices]
        
        if outlook == 'volatile':
            # For volatile outlook, weight the extremes more
            weights = [1 + abs(i - mid_idx) / mid_idx for i in relevant_indices]
            weighted_payoffs = [p * w for p, w in zip(payoffs, weights)]
            expected_values[strategy_name] = sum(weighted_payoffs) / sum(weights)
        else:
            expected_values[strategy_name] = sum(payoffs) / len(payoffs)
    
    # Find strategy with highest expected value
    if expected_values:
        return max(expected_values.items(), key=lambda x: x[1])[0]
    
    return None

def get_strategy_descriptions():
    """
    Get descriptions for different options strategies.
    
    Returns:
    --------
    dict
        Dictionary with strategy descriptions
    """
    return {
        'long_call': {
            'name': 'Long Call',
            'description': 'Buying a call option gives you the right to purchase the ETF at the strike price. This strategy is used when you expect the ETF price to rise significantly.',
            'risk_profile': 'Limited risk (premium paid), unlimited potential profit',
            'best_for': 'Bullish outlook with significant upside potential',
            'example': 'Buy a call option with strike price $100 for a premium of $5. Break-even at $105. Profit increases as the ETF price rises above $105.'
        },
        'covered_call': {
            'name': 'Covered Call',
            'description': 'Owning the ETF and selling a call option against it. This strategy generates income from the premium while limiting potential upside.',
            'risk_profile': 'Downside risk of owning the ETF, limited upside potential',
            'best_for': 'Slightly bullish or neutral outlook, income generation',
            'example': 'Own the ETF at $100 and sell a call with strike price $105 for $3. Keep the premium regardless of outcome, but must sell at $105 if the ETF price exceeds that level.'
        },
        'bull_call_spread': {
            'name': 'Bull Call Spread',
            'description': 'Buying a call option at a lower strike price and selling a call at a higher strike price. This reduces the cost but caps the maximum profit.',
            'risk_profile': 'Limited risk (net premium paid), limited potential profit',
            'best_for': 'Moderately bullish outlook with defined risk/reward',
            'example': 'Buy a call with strike $100 for $5 and sell a call with strike $110 for $2. Net cost is $3, max profit is $7 if ETF price exceeds $110.'
        },
        'protective_put': {
            'name': 'Protective Put',
            'description': 'Owning the ETF and buying a put option as insurance. This strategy protects against significant downside while maintaining upside potential.',
            'risk_profile': 'Limited downside risk, unlimited upside potential, but higher cost',
            'best_for': 'Bullish with concern about potential downside risks',
            'example': 'Own the ETF at $100 and buy a put with strike $95 for $3. Maximum loss limited to $8 ($5 from ETF decline + $3 premium), unlimited upside.'
        },
        'iron_condor': {
            'name': 'Iron Condor',
            'description': 'Selling an out-of-the-money put spread and an out-of-the-money call spread. This strategy profits when the ETF price stays within a range.',
            'risk_profile': 'Limited risk, limited potential profit',
            'best_for': 'Neutral outlook, expecting low volatility',
            'example': 'Sell a put spread (buy $90 put, sell $95 put) and sell a call spread (sell $105 call, buy $110 call). Profit when ETF price stays between $95 and $105.'
        }
    }
