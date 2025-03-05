# ETF Options Analysis with Visualization

A Python script to analyze ETF options and calculate potential profit/loss scenarios with visual representations.

## Features

- Fetches current ETF price using yfinance
- Identifies option expiration dates closest to 90 and 180 days from today
- Filters call options with strike prices within ±5% of current price
- Calculates potential profit/loss assuming a 10% price increase by expiry
- Displays results including strike price, option premium, implied volatility, and ROI
- **Generates visualizations** to help understand the options data:
  - ROI vs Strike Price chart
  - Profit/Loss vs Strike Price chart
  - Option Premium vs Strike Price chart
  - Implied Volatility vs Strike Price chart
  - Profit/Loss vs ROI scatter plot
  - Payoff diagrams for different expiration dates

## Requirements

- Python 3.6+
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository or download the script
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

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

## Output

The script will display:
- Current price of the specified ETF
- Selected expiration dates (closest to 90 and 180 days)
- Strike price range (±5% of current price)
- For each expiration date:
  - List of filtered call options
  - Potential profit/loss for each option assuming 10% price increase
  - ROI percentage

## Visualizations

The script generates two types of visualization files:
1. `[TICKER]_options_analysis.png` - Contains multiple charts showing relationships between strike price, ROI, profit/loss, option premium, and implied volatility
2. `[TICKER]_payoff_diagrams.png` - Shows payoff diagrams for selected options at different expiration dates

These visualizations help in understanding:
- Which options offer the best potential return on investment
- How profit/loss changes with different strike prices
- The relationship between option premiums and strike prices
- How implied volatility varies across different strike prices
- The potential payoff at different price points at expiry

## Notes

- The script uses the "ask" price as an approximation for the option premium
- Calculations are simplified and don't account for factors like theta decay or changes in volatility
- No transaction costs or fees are included in the calculations
- If the specified ticker doesn't have options data available, try a more common ETF like SPY
