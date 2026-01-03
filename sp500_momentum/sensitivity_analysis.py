"""
Sensitivity Analysis Module

Perform parameter sensitivity analysis to test strategy robustness
and detect potential overfitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from itertools import product

from .data_fetcher import fetch_batch_data, get_sp500_index_data
from .constituents import get_constituents_at_date
from .strategy import calculate_returns_for_years


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# Weight schemes to test
WEIGHT_SCHEMES = {
    'baseline': {'name': 'Baseline (40/30/30)', 'weights': [0.40, 0.30, 0.30]},
    'equal': {'name': 'Equal (33/33/33)', 'weights': [0.333, 0.333, 0.334]},
    'aggressive': {'name': 'Aggressive (60/20/20)', 'weights': [0.60, 0.20, 0.20]},
    'decaying': {'name': 'Decaying (20/40/40)', 'weights': [0.20, 0.40, 0.40]},
}


def run_strategy_variant(
    stock_data: Dict[str, pd.DataFrame],
    benchmark_returns: Dict[int, float],
    lookback_years: int,
    weight_scheme: List[float],
    analysis_year: int,
    top_n: int = 10
) -> Dict:
    """
    Run a variant of the momentum strategy with different parameters.
    
    Args:
        stock_data: Dict of ticker -> price DataFrame
        benchmark_returns: Dict of year -> S&P 500 return
        lookback_years: Number of years for consecutive outperformance filter
        weight_scheme: List of weights for momentum scoring [recent, mid, old]
        analysis_year: Year to analyze (e.g., 2026)
        top_n: Number of top stocks to select
        
    Returns:
        Dict with selected stocks, scores, and metrics
    """
    # Determine years for filtering
    filter_years = list(range(analysis_year - lookback_years, analysis_year))
    
    # Determine years for scoring (last 3 years before analysis)
    score_years = list(range(analysis_year - 3, analysis_year))
    
    # Filter stocks that beat benchmark in ALL lookback years
    outperformers = {}
    
    for ticker, prices in stock_data.items():
        if prices is None or prices.empty:
            continue
            
        stock_returns = calculate_returns_for_years(prices, filter_years)
        
        # Check if stock beat benchmark in ALL years
        all_years_beat = True
        valid_returns = {}
        
        for year in filter_years:
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
    
    if not outperformers:
        return {
            'selected_stocks': [],
            'scores': {},
            'n_passed_filter': 0
        }
    
    # Calculate weighted momentum scores using weight scheme
    scored = []
    for ticker, returns in outperformers.items():
        # Get returns for scoring years
        score_returns = [returns.get(y, 0) for y in score_years]
        
        # Apply weights (weights are for [most_recent, mid, oldest])
        if len(score_returns) >= 3:
            score = sum(r * w for r, w in zip(score_returns[-3:], weight_scheme))
        else:
            score = np.mean(score_returns) if score_returns else 0
        
        scored.append((ticker, score, returns))
    
    # Sort by score and select top N
    scored.sort(key=lambda x: x[1], reverse=True)
    top_stocks = scored[:top_n]
    
    return {
        'selected_stocks': [t[0] for t in top_stocks],
        'scores': {t[0]: t[1] for t in top_stocks},
        'all_returns': {t[0]: t[2] for t in top_stocks},
        'n_passed_filter': len(outperformers)
    }


def calculate_portfolio_performance(
    tickers: List[str],
    stock_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame,
    year: int
) -> Dict:
    """
    Calculate portfolio performance for a given year.
    
    Returns:
        Dict with portfolio_return, benchmark_return, alpha, sharpe
    """
    if not tickers:
        return {
            'portfolio_return': None,
            'benchmark_return': None,
            'alpha': None,
            'sharpe': None
        }
    
    # Calculate individual stock returns
    individual_returns = []
    daily_returns_list = []
    
    for ticker in tickers:
        if ticker not in stock_data or stock_data[ticker].empty:
            continue
            
        df = stock_data[ticker]
        year_data = df[df.index.year == year]
        
        if len(year_data) < 10:
            continue
        
        # Annual return
        first_price = year_data['Close'].iloc[0]
        last_price = year_data['Close'].iloc[-1]
        if first_price > 0:
            individual_returns.append((last_price - first_price) / first_price)
        
        # Daily returns for Sharpe
        daily_ret = year_data['Close'].pct_change().dropna()
        daily_returns_list.append(daily_ret)
    
    if not individual_returns:
        return {
            'portfolio_return': None,
            'benchmark_return': None,
            'alpha': None,
            'sharpe': None
        }
    
    portfolio_return = np.mean(individual_returns)
    
    # Benchmark return
    if benchmark_data is not None and not benchmark_data.empty:
        bench_year = benchmark_data[benchmark_data.index.year == year]
        if len(bench_year) >= 2:
            benchmark_return = (bench_year['Close'].iloc[-1] - bench_year['Close'].iloc[0]) / bench_year['Close'].iloc[0]
        else:
            benchmark_return = 0
    else:
        benchmark_return = 0
    
    alpha = portfolio_return - benchmark_return
    
    # Calculate Sharpe ratio (annualized)
    if daily_returns_list:
        # Combine daily returns (equal weighted)
        combined_daily = pd.concat(daily_returns_list, axis=1).mean(axis=1)
        if len(combined_daily) > 0 and combined_daily.std() > 0:
            sharpe = (combined_daily.mean() * 252) / (combined_daily.std() * np.sqrt(252))
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    return {
        'portfolio_return': portfolio_return,
        'benchmark_return': benchmark_return,
        'alpha': alpha,
        'sharpe': sharpe
    }


def run_sensitivity_analysis(
    analysis_year: int = 2026,
    backtest_years: List[int] = None,
    lookback_variants: List[int] = None,
    weight_variants: Dict = None,
    top_n: int = 10,
    workers: int = 10,
    show_progress: bool = True
) -> Dict:
    """
    Run comprehensive sensitivity analysis on strategy parameters.
    
    Args:
        analysis_year: Year for stock selection
        backtest_years: Years to backtest performance (default: last 5 years)
        lookback_variants: List of lookback periods to test (default: [3, 4, 5, 6])
        weight_variants: Dict of weight schemes to test
        top_n: Number of top stocks to select
        workers: Number of parallel workers
        show_progress: Whether to print progress
        
    Returns:
        Dict with all analysis results
    """
    if backtest_years is None:
        backtest_years = list(range(analysis_year - 5, analysis_year))
    
    if lookback_variants is None:
        lookback_variants = [3, 4, 5, 6]
    
    if weight_variants is None:
        weight_variants = WEIGHT_SCHEMES
    
    if show_progress:
        print("\n" + "="*70)
        print("STRATEGY SENSITIVITY ANALYSIS")
        print("="*70)
        print(f"Analysis year: {analysis_year}")
        print(f"Backtest years: {backtest_years}")
        print(f"Lookback variants: {lookback_variants}")
        print(f"Weight variants: {list(weight_variants.keys())}")
    
    # Fetch all required data
    min_year = min(min(backtest_years), analysis_year - max(lookback_variants)) - 1
    max_year = analysis_year
    
    start_date = date(min_year, 1, 1)
    end_date = date(max_year, 12, 31)
    
    if show_progress:
        print(f"\nFetching data from {start_date} to {end_date}...")
    
    # Get constituents
    constituents = get_constituents_at_date(date(min_year, 1, 1))
    
    # Fetch all stock data
    stock_data = fetch_batch_data(
        constituents, start_date, end_date,
        workers=workers, show_progress=show_progress
    )
    
    # Fetch benchmark data
    benchmark_data = get_sp500_index_data(start_date, end_date)
    
    # Calculate benchmark returns for all years
    all_years = list(range(min_year, max_year + 1))
    benchmark_returns = {}
    for year in all_years:
        if benchmark_data is not None:
            year_data = benchmark_data[benchmark_data.index.year == year]
            if len(year_data) >= 2:
                first = year_data['Close'].iloc[0]
                last = year_data['Close'].iloc[-1]
                benchmark_returns[year] = (last - first) / first
    
    if show_progress:
        print(f"\nRunning {len(lookback_variants) * len(weight_variants)} parameter combinations...")
    
    # Results storage
    results = []
    stock_selection_matrix = {}
    
    # Test all combinations
    for lookback in lookback_variants:
        for weight_key, weight_info in weight_variants.items():
            weights = weight_info['weights']
            weight_name = weight_info['name']
            
            if show_progress:
                print(f"\n  Testing: Lookback={lookback}yr, Weights={weight_key}")
            
            # Run strategy for current analysis
            variant_result = run_strategy_variant(
                stock_data=stock_data,
                benchmark_returns=benchmark_returns,
                lookback_years=lookback,
                weight_scheme=weights,
                analysis_year=analysis_year,
                top_n=top_n
            )
            
            selected_stocks = variant_result['selected_stocks']
            n_passed = variant_result['n_passed_filter']
            
            # Store stock selection
            key = f"{lookback}yr_{weight_key}"
            stock_selection_matrix[key] = set(selected_stocks)
            
            # Calculate performance for each backtest year
            yearly_alphas = []
            yearly_sharpes = []
            wins = 0
            
            for bt_year in backtest_years:
                # For backtest, we need to select stocks using data before bt_year
                bt_variant = run_strategy_variant(
                    stock_data=stock_data,
                    benchmark_returns=benchmark_returns,
                    lookback_years=lookback,
                    weight_scheme=weights,
                    analysis_year=bt_year,
                    top_n=top_n
                )
                
                bt_stocks = bt_variant['selected_stocks']
                
                if bt_stocks:
                    perf = calculate_portfolio_performance(
                        bt_stocks, stock_data, benchmark_data, bt_year
                    )
                    
                    if perf['alpha'] is not None:
                        yearly_alphas.append(perf['alpha'])
                        if perf['alpha'] > 0:
                            wins += 1
                    
                    if perf['sharpe'] is not None:
                        yearly_sharpes.append(perf['sharpe'])
            
            # Aggregate metrics
            avg_alpha = np.mean(yearly_alphas) if yearly_alphas else 0
            win_rate = wins / len(backtest_years) if backtest_years else 0
            avg_sharpe = np.mean(yearly_sharpes) if yearly_sharpes else 0
            
            results.append({
                'lookback': lookback,
                'weight_scheme': weight_key,
                'weight_name': weight_name,
                'avg_alpha': avg_alpha,
                'win_rate': win_rate,
                'avg_sharpe': avg_sharpe,
                'n_passed_filter': n_passed,
                'selected_stocks': selected_stocks,
                'yearly_alphas': yearly_alphas
            })
            
            if show_progress:
                print(f"    Passed filter: {n_passed}, Avg Alpha: {avg_alpha*100:.2f}%, "
                      f"Win Rate: {win_rate*100:.0f}%, Sharpe: {avg_sharpe:.2f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate stability score
    alpha_std = results_df['avg_alpha'].std()
    alpha_range = results_df['avg_alpha'].max() - results_df['avg_alpha'].min()
    baseline_alpha = results_df[
        (results_df['lookback'] == 5) & 
        (results_df['weight_scheme'] == 'baseline')
    ]['avg_alpha'].values
    
    if len(baseline_alpha) > 0:
        baseline_alpha = baseline_alpha[0]
    else:
        baseline_alpha = results_df['avg_alpha'].mean()
    
    # Strategy is robust if alpha stays within +/- 5% of baseline
    within_range = ((results_df['avg_alpha'] >= baseline_alpha - 0.05) & 
                    (results_df['avg_alpha'] <= baseline_alpha + 0.05)).mean()
    
    is_robust = within_range >= 0.7 and alpha_std < 0.10
    stability_score = "ROBUST" if is_robust else "SENSITIVE/POTENTIALLY OVERFIT"
    
    # Find high-conviction stocks (appear in most variants)
    all_stocks = []
    for stocks in stock_selection_matrix.values():
        all_stocks.extend(stocks)
    
    stock_counts = pd.Series(all_stocks).value_counts()
    total_variants = len(stock_selection_matrix)
    high_conviction = stock_counts[stock_counts >= total_variants * 0.7].index.tolist()
    
    analysis_results = {
        'results_df': results_df,
        'stock_selection_matrix': stock_selection_matrix,
        'high_conviction_stocks': high_conviction,
        'stability_score': stability_score,
        'alpha_std': alpha_std,
        'alpha_range': alpha_range,
        'baseline_alpha': baseline_alpha,
        'within_5pct_range': within_range,
        'benchmark_returns': benchmark_returns
    }
    
    if show_progress:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nStability Score: {stability_score}")
        print(f"Alpha Standard Deviation: {alpha_std*100:.2f}%")
        print(f"Alpha Range: {alpha_range*100:.2f}%")
        print(f"Variants within +/-5% of baseline: {within_range*100:.0f}%")
        print(f"\nHigh-Conviction Stocks (appear in 70%+ of variants):")
        for stock in high_conviction:
            count = stock_counts[stock]
            print(f"  {stock}: {count}/{total_variants} variants ({count/total_variants*100:.0f}%)")
    
    return analysis_results


def plot_sensitivity_heatmap(
    results_df: pd.DataFrame,
    metric: str = 'avg_alpha',
    title: str = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate a heatmap of strategy performance across parameter combinations.
    
    Args:
        results_df: DataFrame with sensitivity analysis results
        metric: Metric to plot ('avg_alpha', 'win_rate', 'avg_sharpe')
        title: Chart title
        save_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pivot data for heatmap
    pivot = results_df.pivot(
        index='weight_name', 
        columns='lookback', 
        values=metric
    )
    
    # Format values for display
    if metric in ['avg_alpha', 'win_rate']:
        annot_fmt = '.1%'
        pivot_display = pivot * 100
        cbar_label = 'Percentage (%)'
    else:
        annot_fmt = '.2f'
        pivot_display = pivot
        cbar_label = 'Value'
    
    # Create heatmap
    sns.heatmap(
        pivot_display,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0 if metric == 'avg_alpha' else None,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': cbar_label}
    )
    
    # Title
    metric_names = {
        'avg_alpha': 'Average Alpha',
        'win_rate': 'Win Rate',
        'avg_sharpe': 'Average Sharpe Ratio'
    }
    if title is None:
        title = f"Sensitivity Analysis: {metric_names.get(metric, metric)}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlabel('Lookback Period (Years)', fontsize=12)
    ax.set_ylabel('Weight Scheme', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_stock_stability(
    stock_selection_matrix: Dict[str, set],
    title: str = "Stock Selection Stability Across Parameter Variants",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot how often each stock appears across different parameter combinations.
    
    Args:
        stock_selection_matrix: Dict of variant_name -> set of selected stocks
        title: Chart title
        save_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Matplotlib figure
    """
    # Count stock appearances
    all_stocks = []
    for stocks in stock_selection_matrix.values():
        all_stocks.extend(stocks)
    
    stock_counts = pd.Series(all_stocks).value_counts()
    total_variants = len(stock_selection_matrix)
    
    # Take top 20 stocks
    top_stocks = stock_counts.head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors based on frequency
    colors = ['#27ae60' if c >= total_variants * 0.7 
              else '#f39c12' if c >= total_variants * 0.4 
              else '#e74c3c' for c in top_stocks.values]
    
    bars = ax.barh(range(len(top_stocks)), top_stocks.values, color=colors)
    ax.set_yticks(range(len(top_stocks)))
    ax.set_yticklabels(top_stocks.index)
    ax.invert_yaxis()
    
    # Add percentage labels
    for i, (bar, count) in enumerate(zip(bars, top_stocks.values)):
        pct = count / total_variants * 100
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{count}/{total_variants} ({pct:.0f}%)',
                va='center', fontsize=10)
    
    ax.axvline(x=total_variants * 0.7, color='#27ae60', linestyle='--', 
               linewidth=2, label='High Conviction (70%+)')
    ax.axvline(x=total_variants * 0.4, color='#f39c12', linestyle='--',
               linewidth=2, label='Medium Conviction (40%+)')
    
    ax.set_xlabel('Number of Variants Selected In', fontsize=12)
    ax.set_ylabel('Stock', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    
    ax.set_xlim(0, total_variants + 2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Run sensitivity analysis
    results = run_sensitivity_analysis(
        analysis_year=2026,
        lookback_variants=[3, 4, 5, 6],
        show_progress=True
    )
    
    # Generate visualizations
    plot_sensitivity_heatmap(
        results['results_df'],
        metric='avg_alpha',
        show=True
    )
    
    plot_stock_stability(
        results['stock_selection_matrix'],
        show=True
    )
