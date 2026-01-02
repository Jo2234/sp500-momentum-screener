"""
Momentum Strategy Logic

Implements the stock screening and momentum scoring strategy:
1. Filter stocks that outperformed S&P 500 in all specified years
2. Calculate weighted momentum scores
3. Rank and select top stocks
"""

from datetime import date
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from .data_fetcher import fetch_batch_data, get_sp500_index_data
from .constituents import get_constituents_at_date


def calculate_annual_return(prices: pd.DataFrame, year: int) -> Optional[float]:
    """
    Calculate total return for a specific year.
    
    Args:
        prices: DataFrame with price data (must have 'Close' column or be a Series)
        year: The year to calculate return for
        
    Returns:
        Annual return as a decimal (e.g., 0.25 for 25%), or None if insufficient data
    """
    if prices is None or (hasattr(prices, 'empty') and prices.empty):
        return None
    
    # Handle both DataFrame and Series
    if isinstance(prices, pd.DataFrame):
        if 'Close' in prices.columns:
            close_prices = prices['Close']
        else:
            close_prices = prices.iloc[:, 0]
    else:
        close_prices = prices
    
    # Filter to the specific year
    mask = close_prices.index.year == year
    year_prices = close_prices[mask]
    
    if len(year_prices) < 10:  # Need at least some trading days
        return None
    
    # Calculate return from first to last price of the year
    first_price = year_prices.iloc[0]
    last_price = year_prices.iloc[-1]
    
    if first_price <= 0:
        return None
    
    return (last_price - first_price) / first_price


def calculate_returns_for_years(
    prices: pd.DataFrame,
    years: List[int]
) -> Dict[int, Optional[float]]:
    """
    Calculate returns for multiple years.
    
    Args:
        prices: DataFrame with price data
        years: List of years to calculate returns for
        
    Returns:
        Dictionary mapping year -> return
    """
    return {year: calculate_annual_return(prices, year) for year in years}


def filter_consistent_outperformers(
    stock_data: Dict[str, pd.DataFrame],
    benchmark_returns: Dict[int, float],
    years: List[int]
) -> Dict[str, Dict[int, float]]:
    """
    Filter stocks that outperformed the benchmark in ALL specified years.
    
    Args:
        stock_data: Dictionary of ticker -> price DataFrame
        benchmark_returns: Dictionary of year -> S&P 500 return
        years: List of years to check
        
    Returns:
        Dictionary of ticker -> {year: return} for stocks that passed the filter
    """
    outperformers = {}
    
    for ticker, prices in stock_data.items():
        stock_returns = calculate_returns_for_years(prices, years)
        
        # Check if stock beat benchmark in ALL years
        all_years_beat = True
        valid_returns = {}
        
        for year in years:
            stock_ret = stock_returns.get(year)
            bench_ret = benchmark_returns.get(year)
            
            if stock_ret is None or bench_ret is None:
                all_years_beat = False
                break
            
            if stock_ret <= bench_ret:
                all_years_beat = False
                break
            
            valid_returns[year] = stock_ret
        
        if all_years_beat:
            outperformers[ticker] = valid_returns
    
    return outperformers


def calculate_momentum_score(
    returns: Dict[int, float],
    weights: Optional[Dict[int, float]] = None
) -> float:
    """
    Calculate weighted momentum score.
    
    Default weights:
    - Most recent year: 40%
    - Second most recent: 30%
    - Third most recent: 30%
    
    Args:
        returns: Dictionary of year -> return
        weights: Optional custom weights (year -> weight)
        
    Returns:
        Weighted momentum score
    """
    if weights is None:
        # Default: use most recent 3 years
        years = sorted(returns.keys(), reverse=True)[:3]
        if len(years) < 3:
            # Use available years with default weights spread
            weights = {y: 1.0 / len(years) for y in years}
        else:
            weights = {
                years[0]: 0.40,  # Most recent (2025)
                years[1]: 0.30,  # Second most recent (2024)
                years[2]: 0.30,  # Third most recent (2023)
            }
    
    score = 0.0
    for year, weight in weights.items():
        if year in returns:
            score += returns[year] * weight
    
    return score


def select_top_stocks(
    filtered_stocks: Dict[str, Dict[int, float]],
    n: int = 10,
    weights: Optional[Dict[int, float]] = None
) -> List[Tuple[str, float, Dict[int, float]]]:
    """
    Rank stocks by momentum score and select top N.
    
    Args:
        filtered_stocks: Dictionary of ticker -> {year: return} for filtered stocks
        n: Number of top stocks to select
        weights: Optional custom weights for momentum calculation
        
    Returns:
        List of (ticker, momentum_score, returns_dict) sorted by score descending
    """
    scored = []
    for ticker, returns in filtered_stocks.items():
        score = calculate_momentum_score(returns, weights)
        scored.append((ticker, score, returns))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return scored[:n]


def run_momentum_screen(
    analysis_date: date,
    lookback_years: int = 5,
    top_n: int = 10,
    workers: int = 10,
    show_progress: bool = True
) -> Tuple[List[Tuple[str, float, Dict[int, float]]], Dict[int, float], pd.DataFrame]:
    """
    Run the complete momentum screening process.
    
    Args:
        analysis_date: Date to run analysis from (typically Jan 1 of target year)
        lookback_years: Number of years to look back for filtering
        top_n: Number of top stocks to return
        workers: Number of parallel workers for data fetching
        show_progress: Whether to show progress bars
        
    Returns:
        Tuple of:
        - List of (ticker, score, returns) for top stocks
        - Dictionary of benchmark returns by year
        - DataFrame summary of results
    """
    # Determine the years to analyze
    target_year = analysis_date.year
    years = list(range(target_year - lookback_years, target_year))
    
    if show_progress:
        print(f"\n{'='*60}")
        print(f"Momentum Screen for {target_year}")
        print(f"Lookback years: {years}")
        print(f"{'='*60}")
    
    # Get S&P 500 constituents at the start of the lookback period
    constituents_date = date(years[0], 1, 1)
    universe = get_constituents_at_date(constituents_date)
    
    if show_progress:
        print(f"\nUniverse: {len(universe)} stocks in S&P 500 as of {constituents_date}")
    
    # Fetch all price data
    start_date = date(years[0], 1, 1)
    end_date = date(years[-1], 12, 31)
    
    if show_progress:
        print(f"\nFetching data from {start_date} to {end_date}...")
    
    stock_data = fetch_batch_data(
        universe, start_date, end_date,
        workers=workers, show_progress=show_progress
    )
    
    if show_progress:
        print(f"Successfully fetched data for {len(stock_data)} stocks")
    
    # Get benchmark (S&P 500) returns
    sp500_data = get_sp500_index_data(start_date, end_date)
    benchmark_returns = calculate_returns_for_years(sp500_data, years)
    
    if show_progress:
        print(f"\nS&P 500 Benchmark Returns:")
        for year, ret in sorted(benchmark_returns.items()):
            print(f"  {year}: {ret*100:.2f}%")
    
    # Filter for consistent outperformers
    outperformers = filter_consistent_outperformers(stock_data, benchmark_returns, years)
    
    if show_progress:
        print(f"\nStocks that beat S&P 500 in ALL {len(years)} years: {len(outperformers)}")
    
    if not outperformers:
        print("No stocks passed the filter!")
        return [], benchmark_returns, pd.DataFrame()
    
    # Calculate momentum scores and rank
    top_stocks = select_top_stocks(outperformers, n=top_n)
    
    # Create summary DataFrame
    summary_data = []
    for ticker, score, returns in top_stocks:
        row = {'Ticker': ticker, 'Momentum_Score': score}
        for year, ret in sorted(returns.items()):
            row[f'Return_{year}'] = ret
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    if show_progress:
        print(f"\n{'='*60}")
        print(f"TOP {top_n} MOMENTUM STOCKS")
        print(f"{'='*60}")
        
        for i, (ticker, score, returns) in enumerate(top_stocks, 1):
            returns_str = ", ".join([f"{y}: {r*100:.1f}%" for y, r in sorted(returns.items())])
            print(f"{i:2}. {ticker:6} | Score: {score*100:.2f}% | {returns_str}")
    
    return top_stocks, benchmark_returns, summary_df


if __name__ == "__main__":
    # Test the strategy
    print("Testing momentum strategy...")
    
    # Run screen for 2026 (looking back at 2021-2025)
    top_stocks, benchmark_rets, summary = run_momentum_screen(
        analysis_date=date(2026, 1, 1),
        lookback_years=5,
        top_n=10
    )
    
    if not summary.empty:
        print("\n\nSummary DataFrame:")
        print(summary.to_string())
