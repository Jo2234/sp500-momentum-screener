"""
Backtesting Engine

Run historical backtests to evaluate the momentum strategy's performance
against the S&P 500 benchmark.
"""

from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from .data_fetcher import fetch_batch_data, get_sp500_index_data
from .strategy import run_momentum_screen, calculate_annual_return


def calculate_portfolio_return(
    tickers: List[str],
    start_date: date,
    end_date: date,
    equal_weight: bool = True,
    workers: int = 10
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the return of an equal-weighted portfolio.
    
    Args:
        tickers: List of ticker symbols in the portfolio
        start_date: Portfolio start date
        end_date: Portfolio end date
        equal_weight: If True, use equal weights (default)
        workers: Number of parallel workers
        
    Returns:
        Tuple of (portfolio_return, individual_returns_dict)
    """
    # Fetch data for all tickers
    stock_data = fetch_batch_data(
        tickers, start_date, end_date,
        workers=workers, show_progress=False
    )
    
    individual_returns = {}
    for ticker in tickers:
        if ticker in stock_data:
            df = stock_data[ticker]
            if len(df) >= 2:
                first_price = df['Close'].iloc[0]
                last_price = df['Close'].iloc[-1]
                if first_price > 0:
                    individual_returns[ticker] = (last_price - first_price) / first_price
    
    if not individual_returns:
        return 0.0, {}
    
    # Calculate equal-weighted portfolio return
    if equal_weight:
        portfolio_return = sum(individual_returns.values()) / len(individual_returns)
    else:
        portfolio_return = sum(individual_returns.values()) / len(tickers)
    
    return portfolio_return, individual_returns


def calculate_daily_portfolio_values(
    tickers: List[str],
    start_date: date,
    end_date: date,
    workers: int = 10
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate daily portfolio values and identify key metrics.
    
    Args:
        tickers: List of ticker symbols in the portfolio
        start_date: Portfolio start date
        end_date: Portfolio end date
        workers: Number of parallel workers
        
    Returns:
        Tuple of:
        - DataFrame with daily portfolio values (normalized to 100)
        - Dictionary with metrics: max_drawdown, max_drawdown_date, 
          peak_return, peak_date, daily_returns
    """
    from .data_fetcher import fetch_batch_data, get_sp500_index_data
    
    # Fetch data for all tickers
    stock_data = fetch_batch_data(
        tickers, start_date, end_date,
        workers=workers, show_progress=False
    )
    
    # Build a price matrix with all stocks aligned by date
    price_dfs = []
    for ticker in tickers:
        if ticker in stock_data and not stock_data[ticker].empty:
            df = stock_data[ticker][['Close']].copy()
            df.columns = [ticker]
            price_dfs.append(df)
    
    if not price_dfs:
        return pd.DataFrame(), {}
    
    # Combine all prices and forward fill missing values
    prices = pd.concat(price_dfs, axis=1)
    prices = prices.ffill().bfill()
    
    # Calculate normalized prices (start at 100)
    normalized = prices / prices.iloc[0] * 100
    
    # Equal-weighted portfolio value
    portfolio_values = normalized.mean(axis=1)
    portfolio_values.name = 'Portfolio'
    
    # Get S&P 500 for comparison
    sp500_data = get_sp500_index_data(start_date, end_date)
    if sp500_data is not None and not sp500_data.empty:
        sp500_normalized = sp500_data['Close'] / sp500_data['Close'].iloc[0] * 100
        sp500_normalized.name = 'SP500'
    else:
        sp500_normalized = None
    
    # Combine into one DataFrame
    result_df = pd.DataFrame(portfolio_values)
    if sp500_normalized is not None:
        result_df = result_df.join(sp500_normalized, how='outer')
        result_df = result_df.ffill().bfill()
    
    # Calculate key metrics
    # Peak value (highest point)
    peak_idx = portfolio_values.idxmax()
    peak_value = portfolio_values.max()
    peak_return = (peak_value - 100) / 100  # Return from start
    
    # Calculate running maximum for drawdown
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown_idx = drawdown.idxmin()
    max_drawdown = drawdown.min()
    max_drawdown_value = portfolio_values.loc[max_drawdown_idx]
    
    # Find the peak before the max drawdown
    pre_drawdown_data = running_max.loc[:max_drawdown_idx]
    drawdown_peak_idx = pre_drawdown_data.idxmax()
    drawdown_peak_value = pre_drawdown_data.max()
    
    metrics = {
        'peak_date': peak_idx,
        'peak_value': peak_value,
        'peak_return': peak_return,
        'max_drawdown': max_drawdown,
        'max_drawdown_date': max_drawdown_idx,
        'max_drawdown_value': max_drawdown_value,
        'drawdown_peak_date': drawdown_peak_idx,
        'drawdown_peak_value': drawdown_peak_value,
        'final_value': portfolio_values.iloc[-1],
        'final_return': (portfolio_values.iloc[-1] - 100) / 100,
        'daily_values': result_df
    }
    
    return result_df, metrics


def run_backtest(
    strategy_start_date: date,
    lookback_years: int = 5,
    top_n: int = 10,
    holding_period_months: int = 12,
    workers: int = 10,
    show_progress: bool = True
) -> Dict:
    """
    Run a backtest for the momentum strategy.
    
    The engine looks back `lookback_years` from the start date to identify
    stocks, then tracks performance for the following `holding_period_months`.
    
    Args:
        strategy_start_date: Date to start the strategy (portfolio formation date)
        lookback_years: Number of years to look back for stock selection
        top_n: Number of stocks to select
        holding_period_months: How long to hold the portfolio
        workers: Number of parallel workers
        show_progress: Whether to show progress
        
    Returns:
        Dictionary with backtest results including:
        - 'start_date': Portfolio start date
        - 'end_date': Portfolio end date
        - 'selected_stocks': List of selected tickers
        - 'portfolio_return': Strategy portfolio return
        - 'benchmark_return': S&P 500 return for same period
        - 'alpha': Excess return over benchmark
        - 'individual_returns': Returns for each stock
        - 'selection_returns': Returns used for stock selection
    """
    if isinstance(strategy_start_date, str):
        strategy_start_date = pd.to_datetime(strategy_start_date).date()
    
    if show_progress:
        print(f"\n{'#'*60}")
        print(f"BACKTEST: Strategy Start Date = {strategy_start_date}")
        print(f"{'#'*60}")
    
    # Step 1: Run the screening process looking back from strategy_start_date
    top_stocks, selection_benchmark_returns, summary_df = run_momentum_screen(
        analysis_date=strategy_start_date,
        lookback_years=lookback_years,
        top_n=top_n,
        workers=workers,
        show_progress=show_progress
    )
    
    if not top_stocks:
        return {
            'start_date': strategy_start_date,
            'end_date': None,
            'selected_stocks': [],
            'portfolio_return': None,
            'benchmark_return': None,
            'alpha': None,
            'individual_returns': {},
            'selection_returns': {},
            'error': 'No stocks passed the screening filter'
        }
    
    selected_tickers = [t[0] for t in top_stocks]
    selection_returns = {t[0]: t[2] for t in top_stocks}
    
    if show_progress:
        print(f"\nSelected {len(selected_tickers)} stocks: {selected_tickers}")
    
    # Step 2: Calculate holding period dates
    portfolio_start = strategy_start_date
    portfolio_end = date(
        strategy_start_date.year + (strategy_start_date.month + holding_period_months - 1) // 12,
        ((strategy_start_date.month + holding_period_months - 1) % 12) + 1,
        min(strategy_start_date.day, 28)  # Safe day to avoid month-end issues
    )
    
    if show_progress:
        print(f"\nHolding Period: {portfolio_start} to {portfolio_end}")
    
    # Step 3: Calculate portfolio return
    portfolio_return, individual_returns = calculate_portfolio_return(
        selected_tickers, portfolio_start, portfolio_end, workers=workers
    )
    
    # Step 4: Calculate benchmark return for the same period
    sp500_data = get_sp500_index_data(portfolio_start, portfolio_end)
    if sp500_data is not None and len(sp500_data) >= 2:
        first_price = sp500_data['Close'].iloc[0]
        last_price = sp500_data['Close'].iloc[-1]
        benchmark_return = (last_price - first_price) / first_price
    else:
        benchmark_return = 0.0
    
    # Step 5: Calculate alpha (excess return)
    alpha = portfolio_return - benchmark_return
    
    results = {
        'start_date': portfolio_start,
        'end_date': portfolio_end,
        'year': portfolio_start.year,
        'selected_stocks': selected_tickers,
        'portfolio_return': portfolio_return,
        'benchmark_return': benchmark_return,
        'alpha': alpha,
        'individual_returns': individual_returns,
        'selection_returns': selection_returns,
        'summary_df': summary_df
    }
    
    if show_progress:
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Portfolio Return: {portfolio_return*100:.2f}%")
        print(f"S&P 500 Return:   {benchmark_return*100:.2f}%")
        print(f"Alpha:            {alpha*100:.2f}%")
        print(f"\nIndividual Stock Returns:")
        for ticker, ret in sorted(individual_returns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker:6}: {ret*100:.2f}%")
    
    return results


def run_multi_year_backtest(
    start_years: List[int],
    lookback_years: int = 5,
    top_n: int = 10,
    workers: int = 10,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Run backtests for multiple years and compile results.
    
    Args:
        start_years: List of years to start backtests (e.g., [2020, 2021, 2022])
        lookback_years: Number of years to look back for each backtest
        top_n: Number of stocks to select
        workers: Number of parallel workers
        show_progress: Whether to show progress
        
    Returns:
        DataFrame with results for each year
    """
    all_results = []
    
    for year in start_years:
        start_date = date(year, 1, 1)
        
        if show_progress:
            print(f"\n\n{'*'*60}")
            print(f"Running backtest for {year}")
            print(f"{'*'*60}")
        
        result = run_backtest(
            strategy_start_date=start_date,
            lookback_years=lookback_years,
            top_n=top_n,
            workers=workers,
            show_progress=show_progress
        )
        
        all_results.append({
            'Year': year,
            'Portfolio_Return': result.get('portfolio_return'),
            'Benchmark_Return': result.get('benchmark_return'),
            'Alpha': result.get('alpha'),
            'Stocks': ', '.join(result.get('selected_stocks', []))
        })
    
    results_df = pd.DataFrame(all_results)
    
    if show_progress and not results_df.empty:
        print(f"\n\n{'='*60}")
        print("MULTI-YEAR BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(results_df.to_string())
        
        # Summary statistics
        valid_results = results_df.dropna(subset=['Portfolio_Return', 'Benchmark_Return'])
        if not valid_results.empty:
            avg_portfolio = valid_results['Portfolio_Return'].mean()
            avg_benchmark = valid_results['Benchmark_Return'].mean()
            avg_alpha = valid_results['Alpha'].mean()
            win_rate = (valid_results['Alpha'] > 0).sum() / len(valid_results)
            
            print(f"\nAggregate Statistics:")
            print(f"  Avg Portfolio Return: {avg_portfolio*100:.2f}%")
            print(f"  Avg Benchmark Return: {avg_benchmark*100:.2f}%")
            print(f"  Avg Alpha:            {avg_alpha*100:.2f}%")
            print(f"  Win Rate:             {win_rate*100:.1f}%")
    
    return results_df


if __name__ == "__main__":
    # Test backtest
    print("Testing backtester...")
    
    # Run a single backtest for 2024
    result = run_backtest(
        strategy_start_date=date(2024, 1, 1),
        lookback_years=5,
        top_n=10
    )
    
    print("\n\nBacktest complete!")
