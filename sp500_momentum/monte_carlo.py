"""
Monte Carlo Simulation Module

Stress-test portfolio performance using Monte Carlo simulation
with multivariate normal distribution to preserve stock correlations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .data_fetcher import fetch_batch_data


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def calculate_daily_returns(
    tickers: List[str],
    start_date: date,
    end_date: date,
    workers: int = 10
) -> pd.DataFrame:
    """
    Calculate daily returns for a list of stocks.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        workers: Number of parallel workers
        
    Returns:
        DataFrame of daily returns (dates x tickers)
    """
    # Fetch data from cache
    stock_data = fetch_batch_data(
        tickers, start_date, end_date,
        workers=workers, show_progress=False
    )
    
    # Build price matrix
    prices = {}
    for ticker in tickers:
        if ticker in stock_data and not stock_data[ticker].empty:
            prices[ticker] = stock_data[ticker]['Close']
    
    if not prices:
        return pd.DataFrame()
    
    # Combine prices
    price_df = pd.DataFrame(prices)
    price_df = price_df.ffill().bfill()
    
    # Calculate daily returns
    returns = price_df.pct_change().dropna()
    
    return returns


def run_monte_carlo(
    tickers: List[str],
    n_simulations: int = 10000,
    n_days: int = 252,
    initial_investment: float = 10000.0,
    reference_year: int = 2025,
    workers: int = 10,
    show_progress: bool = True
) -> Dict:
    """
    Run Monte Carlo simulation for portfolio stress testing.
    
    Uses historical daily returns to estimate mean and covariance,
    then generates synthetic returns using multivariate normal distribution.
    
    Args:
        tickers: List of ticker symbols in the portfolio
        n_simulations: Number of simulation iterations (default: 10,000)
        n_days: Number of trading days to simulate (default: 252)
        initial_investment: Starting portfolio value (default: $10,000)
        reference_year: Year to use for historical data (default: 2025)
        workers: Number of parallel workers for data fetching
        show_progress: Whether to print progress messages
        
    Returns:
        Dictionary containing:
        - 'final_values': Array of final portfolio values (n_simulations,)
        - 'equity_curves': Array of all equity curves (n_simulations, n_days+1)
        - 'mean_returns': Mean daily returns by stock
        - 'cov_matrix': Covariance matrix of returns
        - 'statistics': Dict with median, percentiles, probabilities
    """
    if show_progress:
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION")
        print("="*60)
        print(f"Portfolio: {tickers}")
        print(f"Simulations: {n_simulations:,}")
        print(f"Trading days: {n_days}")
        print(f"Initial investment: ${initial_investment:,.2f}")
    
    # Step 1: Get historical daily returns
    start_date = date(reference_year, 1, 1)
    end_date = date(reference_year, 12, 31)
    
    if show_progress:
        print(f"\nFetching historical returns for {reference_year}...")
    
    returns_df = calculate_daily_returns(tickers, start_date, end_date, workers)
    
    if returns_df.empty:
        raise ValueError("Could not calculate historical returns")
    
    # Filter to only stocks with data
    valid_tickers = [t for t in tickers if t in returns_df.columns]
    returns_df = returns_df[valid_tickers]
    
    if show_progress:
        print(f"Using {len(valid_tickers)} stocks with {len(returns_df)} days of data")
    
    # Step 2: Calculate mean returns and covariance matrix
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    
    if show_progress:
        print(f"\nMean daily returns: {mean_returns.mean()*100:.4f}%")
        print(f"Avg daily volatility: {np.sqrt(np.diag(cov_matrix)).mean()*100:.4f}%")
    
    # Step 3: Run vectorized Monte Carlo simulation
    if show_progress:
        print(f"\nRunning {n_simulations:,} simulations...")
    
    n_stocks = len(valid_tickers)
    
    # Generate all random returns at once (n_simulations x n_days x n_stocks)
    # Using multivariate normal to preserve correlations
    np.random.seed(42)  # For reproducibility
    
    # Generate random samples
    random_returns = np.random.multivariate_normal(
        mean_returns, 
        cov_matrix, 
        size=(n_simulations, n_days)
    )
    
    # Calculate portfolio returns (equal-weighted)
    portfolio_returns = random_returns.mean(axis=2)  # (n_simulations, n_days)
    
    # Calculate equity curves
    # Start with initial investment, compound daily returns
    equity_curves = np.zeros((n_simulations, n_days + 1))
    equity_curves[:, 0] = initial_investment
    
    for day in range(n_days):
        equity_curves[:, day + 1] = equity_curves[:, day] * (1 + portfolio_returns[:, day])
    
    # Final portfolio values
    final_values = equity_curves[:, -1]
    
    # Step 4: Calculate statistics
    median_value = np.median(final_values)
    percentile_5 = np.percentile(final_values, 5)
    percentile_95 = np.percentile(final_values, 95)
    mean_value = np.mean(final_values)
    std_value = np.std(final_values)
    
    # Calculate returns
    median_return = (median_value - initial_investment) / initial_investment
    var_5_return = (percentile_5 - initial_investment) / initial_investment
    best_case_return = (percentile_95 - initial_investment) / initial_investment
    
    # Probability of beating benchmark (assume 8% annual S&P return)
    benchmark_return = 0.08
    benchmark_value = initial_investment * (1 + benchmark_return)
    prob_beat_benchmark = (final_values > benchmark_value).mean()
    
    # Probability of positive return
    prob_positive = (final_values > initial_investment).mean()
    
    # Maximum and minimum
    max_value = final_values.max()
    min_value = final_values.min()
    
    statistics = {
        'median_value': median_value,
        'median_return': median_return,
        'mean_value': mean_value,
        'std_value': std_value,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95,
        'var_5_return': var_5_return,
        'best_case_return': best_case_return,
        'max_value': max_value,
        'min_value': min_value,
        'prob_beat_benchmark': prob_beat_benchmark,
        'prob_positive': prob_positive,
        'benchmark_return': benchmark_return,
        'benchmark_value': benchmark_value,
        'n_simulations': n_simulations,
        'n_days': n_days,
        'initial_investment': initial_investment
    }
    
    if show_progress:
        print("\n" + "-"*50)
        print("SIMULATION RESULTS")
        print("-"*50)
        print(f"\nInitial Investment: ${initial_investment:,.2f}")
        print(f"\nExpected Outcomes (after {n_days} trading days):")
        print(f"  Median Final Value:    ${median_value:,.2f} ({median_return*100:+.2f}%)")
        print(f"  Mean Final Value:      ${mean_value:,.2f}")
        print(f"\nRisk Analysis:")
        print(f"  5th Percentile (VaR):  ${percentile_5:,.2f} ({var_5_return*100:+.2f}%)")
        print(f"  95th Percentile:       ${percentile_95:,.2f} ({best_case_return*100:+.2f}%)")
        print(f"  Worst Case:            ${min_value:,.2f}")
        print(f"  Best Case:             ${max_value:,.2f}")
        print(f"\nProbabilities:")
        print(f"  P(Beat {benchmark_return*100:.0f}% Benchmark): {prob_beat_benchmark*100:.1f}%")
        print(f"  P(Positive Return):    {prob_positive*100:.1f}%")
    
    return {
        'final_values': final_values,
        'equity_curves': equity_curves,
        'mean_returns': dict(zip(valid_tickers, mean_returns)),
        'cov_matrix': pd.DataFrame(cov_matrix, index=valid_tickers, columns=valid_tickers),
        'statistics': statistics,
        'tickers': valid_tickers
    }


def plot_equity_curves(
    equity_curves: np.ndarray,
    statistics: Dict,
    n_curves: int = 100,
    title: str = "Monte Carlo Simulation: Portfolio Equity Curves",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a fan of simulated equity curves.
    
    Args:
        equity_curves: Array of shape (n_simulations, n_days+1)
        statistics: Dictionary with simulation statistics
        n_curves: Number of curves to display (default: 100)
        title: Chart title
        save_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    initial = statistics['initial_investment']
    n_days = statistics['n_days']
    
    # Plot first n_curves
    days = np.arange(n_days + 1)
    
    for i in range(min(n_curves, len(equity_curves))):
        final = equity_curves[i, -1]
        color = '#2ecc71' if final >= initial else '#e74c3c'
        alpha = 0.3
        ax.plot(days, equity_curves[i], color=color, alpha=alpha, linewidth=0.5)
    
    # Plot median curve
    median_curve = np.median(equity_curves, axis=0)
    ax.plot(days, median_curve, color='#2c3e50', linewidth=2.5, label='Median Path')
    
    # Plot percentile bands
    p5_curve = np.percentile(equity_curves, 5, axis=0)
    p95_curve = np.percentile(equity_curves, 95, axis=0)
    ax.fill_between(days, p5_curve, p95_curve, alpha=0.2, color='#3498db', label='5th-95th Percentile')
    
    # Add horizontal lines
    ax.axhline(y=initial, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Initial Investment')
    ax.axhline(y=statistics['benchmark_value'], color='#f39c12', linestyle='--', 
               linewidth=1.5, label=f'{statistics["benchmark_return"]*100:.0f}% Benchmark')
    
    # Annotations
    ax.annotate(
        f'Median: ${statistics["median_value"]:,.0f} ({statistics["median_return"]*100:+.1f}%)',
        xy=(n_days, median_curve[-1]),
        xytext=(10, 0),
        textcoords='offset points',
        fontsize=10, fontweight='bold',
        color='#2c3e50'
    )
    
    # Title and labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_distribution(
    final_values: np.ndarray,
    statistics: Dict,
    title: str = "Monte Carlo Simulation: Distribution of Final Portfolio Values",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot histogram of final portfolio values.
    
    Args:
        final_values: Array of final portfolio values
        statistics: Dictionary with simulation statistics
        title: Chart title
        save_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    initial = statistics['initial_investment']
    median = statistics['median_value']
    p5 = statistics['percentile_5']
    p95 = statistics['percentile_95']
    benchmark = statistics['benchmark_value']
    
    # Create histogram
    n, bins, patches = ax.hist(final_values, bins=80, edgecolor='white', linewidth=0.5)
    
    # Color bars based on value
    for i, (b, patch) in enumerate(zip(bins[:-1], patches)):
        if b < initial:
            patch.set_facecolor('#e74c3c')  # Red for loss
        elif b < benchmark:
            patch.set_facecolor('#f39c12')  # Orange for below benchmark
        else:
            patch.set_facecolor('#2ecc71')  # Green for above benchmark
    
    # Add vertical lines
    ax.axvline(x=initial, color='gray', linestyle='--', linewidth=2, label='Initial ($10,000)')
    ax.axvline(x=median, color='#2c3e50', linestyle='-', linewidth=2.5, label=f'Median (${median:,.0f})')
    ax.axvline(x=p5, color='#c0392b', linestyle='-', linewidth=2, label=f'5th Pctl (${p5:,.0f})')
    ax.axvline(x=benchmark, color='#f39c12', linestyle='--', linewidth=2, label=f'8% Benchmark (${benchmark:,.0f})')
    
    # Shade danger zone
    ax.axvspan(final_values.min(), p5, alpha=0.2, color='#e74c3c', label='5% Worst Case Zone')
    
    # Statistics box
    stats_text = (
        f"Simulations: {statistics['n_simulations']:,}\n"
        f"Median Return: {statistics['median_return']*100:+.1f}%\n"
        f"VaR (5%): {statistics['var_5_return']*100:+.1f}%\n"
        f"Best Case (95%): {statistics['best_case_return']*100:+.1f}%\n"
        f"P(Beat 8%): {statistics['prob_beat_benchmark']*100:.1f}%\n"
        f"P(Profit): {statistics['prob_positive']*100:.1f}%"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
    
    # Title and labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Final Portfolio Value ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    
    # Format x-axis as currency
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Test the Monte Carlo simulation
    test_tickers = ['FIX', 'AVGO', 'ANET', 'APH', 'EME', 'PWR', 'LLY', 'AXP', 'JPM', 'ORCL']
    
    results = run_monte_carlo(
        tickers=test_tickers,
        n_simulations=10000,
        n_days=252,
        initial_investment=10000,
        reference_year=2025
    )
    
    plot_equity_curves(
        results['equity_curves'],
        results['statistics'],
        n_curves=100,
        show=True
    )
    
    plot_distribution(
        results['final_values'],
        results['statistics'],
        show=True
    )
