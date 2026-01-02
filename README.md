# S&P 500 Momentum Stock Screener

A Python-based stock screening and backtesting engine that identifies high-performance momentum stocks within the S&P 500 universe.

## Features

- **Momentum Screening**: Identifies stocks that outperformed the S&P 500 in each of the past 5 years
- **Weighted Ranking**: Scores stocks using a weighted momentum formula (40% most recent year, 30% each for prior two years)
- **Survivorship Bias Handling**: Uses historical S&P 500 constituent data to avoid look-ahead bias
- **Efficient Data Fetching**: Multithreaded batch fetching with parquet-based local caching
- **Portfolio Movement Tracking**: Daily portfolio values with peak and max drawdown annotations
- **Backtesting Engine**: Single-year and multi-year backtests with performance comparison charts

## Installation

```bash
cd /path/to/SP
pip install -r requirements.txt
```

## Usage

### Screen for Current Year Portfolio

```bash
python main.py --mode screen
```

Identifies the top 10 momentum stocks for the upcoming year based on the past 5 years of performance.

### Run Historical Backtest

```bash
python main.py --mode backtest --start-date 2024-01-01
```

Tests the strategy starting from a specific date, using 5-year lookback for stock selection.

### Multi-Year Backtest

```bash
python main.py --mode multi-backtest --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

Runs backtests across multiple years and generates a comparison chart.

### Portfolio Movement Analysis

```bash
python main.py --mode movement-analysis --years 2020 2021 2022 2023 2024 2025
```

Generates daily portfolio movement charts with peak return and max drawdown markers for each year.

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Operation mode: screen, backtest, multi-backtest, movement-analysis | Required |
| `--start-date` | Strategy start date for single backtest (YYYY-MM-DD) | - |
| `--years` | Years for multi-year analysis | - |
| `--lookback` | Number of years to look back for stock selection | 5 |
| `--top-n` | Number of top stocks to select | 10 |
| `--workers` | Number of parallel workers for data fetching | 10 |
| `--no-show` | Save charts without displaying | False |

## Project Structure

```
SP/
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── sp500_momentum/
│   ├── __init__.py
│   ├── constituents.py        # S&P 500 historical constituents
│   ├── data_fetcher.py        # Multithreaded data fetching with caching
│   ├── strategy.py            # Momentum filter and scoring logic
│   ├── backtester.py          # Backtesting engine
│   └── visualization.py       # Chart generation
├── cache/                     # Cached stock price data (parquet files)
├── data/                      # S&P 500 constituents cache
└── output/                    # Generated charts and CSV files
```

## Strategy Logic

1. **Universe Selection**: Get S&P 500 constituents at the start of the lookback period
2. **Filter**: Identify stocks that beat the S&P 500 index in ALL 5 years
3. **Score**: Calculate weighted momentum score:
   - Year T-1: 40% weight
   - Year T-2: 30% weight
   - Year T-3: 30% weight
4. **Rank**: Select top 10 stocks by momentum score
5. **Hold**: Equal-weighted portfolio for 12 months

## Performance (10-Year Backtest: 2016-2025)

| Metric | Value |
|--------|-------|
| Average Portfolio Return | 20.00% |
| Average Benchmark Return | 14.09% |
| Average Alpha | +5.91% |
| Win Rate | 70% (7/10 years) |

## Dependencies

- yfinance: Yahoo Finance API for stock data
- pandas: Data manipulation
- numpy: Numerical operations
- matplotlib: Visualization
- pyarrow: Parquet file support for caching
- requests: HTTP requests
- tqdm: Progress bars
- lxml: HTML parsing for Wikipedia data

## License

MIT License
