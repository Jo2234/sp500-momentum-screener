"""
Out-of-Sample (OOS) Validation Module

Conduct holdout tests and walk-forward analysis to validate
strategy robustness across different market regimes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats

from .data_fetcher import fetch_batch_data, get_sp500_index_data
from .constituents import get_constituents_at_date
from .strategy import calculate_returns_for_years


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# Market regime classification
MARKET_REGIMES = {
    2010: 'Recovery',
    2011: 'Volatile',
    2012: 'Bull',
    2013: 'Bull',
    2014: 'Bull',
    2015: 'Volatile',
    2016: 'Bull',
    2017: 'Bull',
    2018: 'Bear',
    2019: 'Bull',
    2020: 'Volatile',
    2021: 'Bull',
    2022: 'Bear',
    2023: 'Bull',
    2024: 'Bull',
    2025: 'Bull',
}


def run_strategy_for_year(
    stock_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame,
    test_year: int,
    lookback_years: int = 5,
    weights: List[float] = None,
    top_n: int = 10
) -> Dict:
    """
    Run the momentum strategy for a specific test year.
    
    Args:
        stock_data: Dict of ticker -> price DataFrame
        benchmark_data: S&P 500 price DataFrame
        test_year: Year to test performance
        lookback_years: Years of consecutive outperformance required
        weights: Momentum scoring weights [recent, mid, old]
        top_n: Number of top stocks to select
        
    Returns:
        Dict with strategy results
    """
    if weights is None:
        weights = [0.40, 0.30, 0.30]
    
    # Calculate benchmark returns for lookback period
    filter_years = list(range(test_year - lookback_years, test_year))
    
    benchmark_returns = {}
    for year in filter_years + [test_year]:
        year_data = benchmark_data[benchmark_data.index.year == year]
        if len(year_data) >= 2:
            benchmark_returns[year] = (year_data['Close'].iloc[-1] - year_data['Close'].iloc[0]) / year_data['Close'].iloc[0]
    
    # Filter stocks that beat benchmark in ALL lookback years
    outperformers = {}
    
    for ticker, prices in stock_data.items():
        if prices is None or prices.empty:
            continue
            
        stock_returns = calculate_returns_for_years(prices, filter_years)
        
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
            'portfolio_return': None,
            'benchmark_return': benchmark_returns.get(test_year),
            'alpha': None,
            'n_passed_filter': 0
        }
    
    # Calculate weighted momentum scores
    score_years = list(range(test_year - 3, test_year))
    scored = []
    
    for ticker, returns in outperformers.items():
        score_returns = [returns.get(y, 0) for y in score_years if y in returns]
        if len(score_returns) >= 3:
            score = sum(r * w for r, w in zip(score_returns[-3:], weights))
        else:
            score = np.mean(score_returns) if score_returns else 0
        scored.append((ticker, score, returns))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [t[0] for t in scored[:top_n]]
    
    # Calculate test year performance
    individual_returns = []
    daily_returns_list = []
    
    for ticker in selected:
        if ticker not in stock_data:
            continue
        df = stock_data[ticker]
        year_data = df[df.index.year == test_year]
        
        if len(year_data) >= 2:
            ret = (year_data['Close'].iloc[-1] - year_data['Close'].iloc[0]) / year_data['Close'].iloc[0]
            individual_returns.append(ret)
            daily_returns_list.append(year_data['Close'].pct_change().dropna())
    
    if not individual_returns:
        return {
            'selected_stocks': selected,
            'portfolio_return': None,
            'benchmark_return': benchmark_returns.get(test_year),
            'alpha': None,
            'n_passed_filter': len(outperformers)
        }
    
    portfolio_return = np.mean(individual_returns)
    benchmark_return = benchmark_returns.get(test_year, 0)
    alpha = portfolio_return - benchmark_return
    
    # Calculate Sharpe ratio
    if daily_returns_list:
        combined_daily = pd.concat(daily_returns_list, axis=1).mean(axis=1)
        if len(combined_daily) > 0 and combined_daily.std() > 0:
            sharpe = (combined_daily.mean() * 252) / (combined_daily.std() * np.sqrt(252))
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    return {
        'test_year': test_year,
        'selected_stocks': selected,
        'portfolio_return': portfolio_return,
        'benchmark_return': benchmark_return,
        'alpha': alpha,
        'sharpe': sharpe,
        'n_passed_filter': len(outperformers),
        'individual_returns': dict(zip(selected, individual_returns)),
        'regime': MARKET_REGIMES.get(test_year, 'Unknown')
    }


def run_oos_validation(
    is_years: List[int] = None,
    oos_years: List[int] = None,
    lookback_years: int = 5,
    weights: List[float] = None,
    top_n: int = 10,
    workers: int = 10,
    show_progress: bool = True
) -> Dict:
    """
    Run out-of-sample validation comparing IS and OOS periods.
    
    Args:
        is_years: In-sample test years (default: 2021-2025)
        oos_years: Out-of-sample test years (default: 2015-2019)
        lookback_years: Years of consecutive outperformance
        weights: Momentum scoring weights
        top_n: Number of top stocks
        workers: Parallel workers
        show_progress: Print progress
        
    Returns:
        Dict with validation results
    """
    if is_years is None:
        is_years = [2021, 2022, 2023, 2024, 2025]
    if oos_years is None:
        oos_years = [2015, 2016, 2017, 2018, 2019]
    if weights is None:
        weights = [0.40, 0.30, 0.30]
    
    if show_progress:
        print("\n" + "="*70)
        print("OUT-OF-SAMPLE VALIDATION")
        print("="*70)
        print(f"In-Sample Period: {is_years}")
        print(f"Out-of-Sample Period: {oos_years}")
        print(f"Strategy: {lookback_years}-year filter, weights={weights}")
    
    # Determine data range needed
    min_year = min(min(oos_years) - lookback_years, min(is_years) - lookback_years)
    max_year = max(max(oos_years), max(is_years))
    
    start_date = date(min_year, 1, 1)
    end_date = date(max_year, 12, 31)
    
    if show_progress:
        print(f"\nFetching data from {start_date} to {end_date}...")
    
    # Get constituents and fetch data
    constituents = get_constituents_at_date(start_date)
    stock_data = fetch_batch_data(
        constituents, start_date, end_date,
        workers=workers, show_progress=show_progress
    )
    benchmark_data = get_sp500_index_data(start_date, end_date)
    
    # Run strategy on both periods
    is_results = []
    oos_results = []
    
    if show_progress:
        print("\n" + "-"*50)
        print("IN-SAMPLE RESULTS")
        print("-"*50)
    
    for year in is_years:
        result = run_strategy_for_year(
            stock_data, benchmark_data, year,
            lookback_years=lookback_years,
            weights=weights,
            top_n=top_n
        )
        is_results.append(result)
        if show_progress and result['alpha'] is not None:
            print(f"  {year} ({result['regime']:8}): Alpha={result['alpha']*100:+.2f}%, "
                  f"Sharpe={result['sharpe']:.2f}")
    
    if show_progress:
        print("\n" + "-"*50)
        print("OUT-OF-SAMPLE RESULTS (First Try - No Tweaking!)")
        print("-"*50)
    
    for year in oos_years:
        result = run_strategy_for_year(
            stock_data, benchmark_data, year,
            lookback_years=lookback_years,
            weights=weights,
            top_n=top_n
        )
        oos_results.append(result)
        if show_progress and result['alpha'] is not None:
            print(f"  {year} ({result['regime']:8}): Alpha={result['alpha']*100:+.2f}%, "
                  f"Sharpe={result['sharpe']:.2f}")
    
    # Calculate aggregate metrics
    is_valid = [r for r in is_results if r['alpha'] is not None]
    oos_valid = [r for r in oos_results if r['alpha'] is not None]
    
    is_alphas = [r['alpha'] for r in is_valid]
    oos_alphas = [r['alpha'] for r in oos_valid]
    is_returns = [r['portfolio_return'] for r in is_valid]
    oos_returns = [r['portfolio_return'] for r in oos_valid]
    is_sharpes = [r['sharpe'] for r in is_valid]
    oos_sharpes = [r['sharpe'] for r in oos_valid]
    
    # CAGR calculation
    if is_returns:
        is_cumulative = np.prod([1 + r for r in is_returns])
        is_cagr = is_cumulative ** (1/len(is_returns)) - 1
    else:
        is_cagr = 0
    
    if oos_returns:
        oos_cumulative = np.prod([1 + r for r in oos_returns])
        oos_cagr = oos_cumulative ** (1/len(oos_returns)) - 1
    else:
        oos_cagr = 0
    
    # Kruskal-Wallis test (non-parametric test for same distribution)
    if len(is_alphas) >= 3 and len(oos_alphas) >= 3:
        kw_stat, kw_pvalue = stats.kruskal(is_alphas, oos_alphas)
        same_distribution = kw_pvalue > 0.05
    else:
        kw_stat, kw_pvalue = None, None
        same_distribution = None
    
    # Alpha degradation
    is_avg_alpha = np.mean(is_alphas) if is_alphas else 0
    oos_avg_alpha = np.mean(oos_alphas) if oos_alphas else 0
    alpha_degradation = is_avg_alpha - oos_avg_alpha
    
    # Regime analysis
    regime_analysis = {}
    for result in oos_results:
        if result['alpha'] is not None:
            regime = result['regime']
            if regime not in regime_analysis:
                regime_analysis[regime] = {'alphas': [], 'returns': [], 'years': []}
            regime_analysis[regime]['alphas'].append(result['alpha'])
            regime_analysis[regime]['returns'].append(result['portfolio_return'])
            regime_analysis[regime]['years'].append(result['test_year'])
    
    for regime in regime_analysis:
        regime_analysis[regime]['avg_alpha'] = np.mean(regime_analysis[regime]['alphas'])
        regime_analysis[regime]['avg_return'] = np.mean(regime_analysis[regime]['returns'])
    
    results = {
        'is_results': is_results,
        'oos_results': oos_results,
        'is_years': is_years,
        'oos_years': oos_years,
        'is_cagr': is_cagr,
        'oos_cagr': oos_cagr,
        'is_avg_alpha': is_avg_alpha,
        'oos_avg_alpha': oos_avg_alpha,
        'is_avg_sharpe': np.mean(is_sharpes) if is_sharpes else 0,
        'oos_avg_sharpe': np.mean(oos_sharpes) if oos_sharpes else 0,
        'alpha_degradation': alpha_degradation,
        'kw_statistic': kw_stat,
        'kw_pvalue': kw_pvalue,
        'same_distribution': same_distribution,
        'regime_analysis': regime_analysis,
        'is_alphas': is_alphas,
        'oos_alphas': oos_alphas
    }
    
    if show_progress:
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"\n{'Metric':<25} {'In-Sample':>15} {'Out-of-Sample':>15} {'Degradation':>12}")
        print("-"*70)
        print(f"{'Avg Alpha':<25} {is_avg_alpha*100:>14.2f}% {oos_avg_alpha*100:>14.2f}% {alpha_degradation*100:>11.2f}%")
        print(f"{'CAGR':<25} {is_cagr*100:>14.2f}% {oos_cagr*100:>14.2f}%")
        print(f"{'Avg Sharpe':<25} {results['is_avg_sharpe']:>15.2f} {results['oos_avg_sharpe']:>15.2f}")
        
        if kw_pvalue is not None:
            print(f"\nKruskal-Wallis Test:")
            print(f"  Statistic: {kw_stat:.4f}")
            print(f"  P-Value: {kw_pvalue:.4f}")
            if same_distribution:
                print(f"  Result: IS and OOS likely from SAME distribution (p > 0.05)")
            else:
                print(f"  Result: IS and OOS likely from DIFFERENT distributions (p <= 0.05)")
        
        print(f"\nRegime Analysis (OOS):")
        for regime, data in sorted(regime_analysis.items()):
            print(f"  {regime:10}: Avg Alpha={data['avg_alpha']*100:+.2f}%, Years={data['years']}")
    
    return results


def run_walk_forward_analysis(
    start_year: int = 2010,
    end_year: int = 2025,
    train_years: int = 5,
    lookback_years: int = 5,
    weights: List[float] = None,
    top_n: int = 10,
    workers: int = 10,
    show_progress: bool = True
) -> Dict:
    """
    Run walk-forward analysis with rolling windows.
    
    Args:
        start_year: First training period start
        end_year: Last test year
        train_years: Years in each training window
        lookback_years: Years for consistency filter
        weights: Momentum scoring weights
        top_n: Number of top stocks
        workers: Parallel workers
        show_progress: Print progress
        
    Returns:
        Dict with walk-forward results
    """
    if weights is None:
        weights = [0.40, 0.30, 0.30]
    
    if show_progress:
        print("\n" + "="*70)
        print("WALK-FORWARD ANALYSIS")
        print("="*70)
        print(f"Train window: {train_years} years")
        print(f"Test window: 1 year (rolling)")
        print(f"Period: {start_year} to {end_year}")
    
    # Determine full data range needed
    min_year = start_year - lookback_years
    
    start_date = date(min_year, 1, 1)
    end_date = date(end_year, 12, 31)
    
    if show_progress:
        print(f"\nFetching data from {start_date} to {end_date}...")
    
    # Fetch all data once
    constituents = get_constituents_at_date(start_date)
    stock_data = fetch_batch_data(
        constituents, start_date, end_date,
        workers=workers, show_progress=show_progress
    )
    benchmark_data = get_sp500_index_data(start_date, end_date)
    
    # Generate test years
    # First test year is start_year + train_years
    first_test_year = start_year + train_years
    test_years = list(range(first_test_year, end_year + 1))
    
    if show_progress:
        print(f"\nTest years: {test_years}")
        print("\nRunning walk-forward...")
    
    wf_results = []
    
    for test_year in test_years:
        # Training uses data from test_year - train_years to test_year - 1
        # But the strategy filter looks back lookback_years from test_year
        result = run_strategy_for_year(
            stock_data, benchmark_data, test_year,
            lookback_years=lookback_years,
            weights=weights,
            top_n=top_n
        )
        wf_results.append(result)
        
        if show_progress and result['alpha'] is not None:
            print(f"  {test_year} ({result['regime']:8}): Alpha={result['alpha']*100:+.2f}%, "
                  f"Stocks={result['n_passed_filter']}")
    
    # Calculate cumulative equity curve
    valid_results = [r for r in wf_results if r['portfolio_return'] is not None]
    
    if valid_results:
        portfolio_equity = [10000]
        benchmark_equity = [10000]
        years = []
        
        for r in valid_results:
            years.append(r['test_year'])
            portfolio_equity.append(portfolio_equity[-1] * (1 + r['portfolio_return']))
            benchmark_equity.append(benchmark_equity[-1] * (1 + r['benchmark_return']))
        
        total_portfolio_return = (portfolio_equity[-1] - 10000) / 10000
        total_benchmark_return = (benchmark_equity[-1] - 10000) / 10000
        cagr = (portfolio_equity[-1] / 10000) ** (1/len(valid_results)) - 1
        
        alphas = [r['alpha'] for r in valid_results]
        wins = sum(1 for a in alphas if a > 0)
        win_rate = wins / len(alphas)
    else:
        portfolio_equity = [10000]
        benchmark_equity = [10000]
        years = []
        total_portfolio_return = 0
        total_benchmark_return = 0
        cagr = 0
        win_rate = 0
        alphas = []
    
    results = {
        'wf_results': wf_results,
        'test_years': test_years,
        'portfolio_equity': portfolio_equity,
        'benchmark_equity': benchmark_equity,
        'equity_years': [test_years[0] - 1] + years if years else [],
        'total_portfolio_return': total_portfolio_return,
        'total_benchmark_return': total_benchmark_return,
        'cagr': cagr,
        'win_rate': win_rate,
        'avg_alpha': np.mean(alphas) if alphas else 0,
        'n_years': len(valid_results)
    }
    
    if show_progress:
        print("\n" + "-"*50)
        print("WALK-FORWARD SUMMARY")
        print("-"*50)
        print(f"Years tested: {len(valid_results)}")
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Avg Alpha: {results['avg_alpha']*100:+.2f}%")
        print(f"CAGR: {cagr*100:.2f}%")
        print(f"Total Return: {total_portfolio_return*100:.2f}%")
        print(f"vs Benchmark: {total_benchmark_return*100:.2f}%")
    
    return results


def plot_oos_comparison(
    validation_results: Dict,
    title: str = "In-Sample vs Out-of-Sample Performance",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison of IS and OOS performance.
    
    Args:
        validation_results: Dict from run_oos_validation
        title: Chart title
        save_path: Optional path to save
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar chart of annual alphas
    ax1 = axes[0]
    
    is_results = [r for r in validation_results['is_results'] if r['alpha'] is not None]
    oos_results = [r for r in validation_results['oos_results'] if r['alpha'] is not None]
    
    is_years = [r['test_year'] for r in is_results]
    is_alphas = [r['alpha'] * 100 for r in is_results]
    oos_years = [r['test_year'] for r in oos_results]
    oos_alphas = [r['alpha'] * 100 for r in oos_results]
    
    x = np.arange(max(len(is_years), len(oos_years)))
    width = 0.35
    
    # Pad shorter array
    while len(oos_alphas) < len(is_alphas):
        oos_alphas.append(0)
        oos_years.append('')
    while len(is_alphas) < len(oos_alphas):
        is_alphas.append(0)
        is_years.append('')
    
    bars1 = ax1.bar(x - width/2, oos_alphas, width, label='Out-of-Sample', 
                    color=['#e74c3c' if a < 0 else '#3498db' for a in oos_alphas])
    bars2 = ax1.bar(x + width/2, is_alphas, width, label='In-Sample',
                    color=['#c0392b' if a < 0 else '#2ecc71' for a in is_alphas])
    
    ax1.set_xlabel('Year Index')
    ax1.set_ylabel('Alpha (%)')
    ax1.set_title('Annual Alpha Comparison')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Right: Summary metrics comparison
    ax2 = axes[1]
    
    metrics = ['Avg Alpha', 'CAGR', 'Avg Sharpe']
    is_vals = [
        validation_results['is_avg_alpha'] * 100,
        validation_results['is_cagr'] * 100,
        validation_results['is_avg_sharpe']
    ]
    oos_vals = [
        validation_results['oos_avg_alpha'] * 100,
        validation_results['oos_cagr'] * 100,
        validation_results['oos_avg_sharpe']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, oos_vals, width, label='Out-of-Sample', color='#3498db')
    bars2 = ax2.bar(x + width/2, is_vals, width, label='In-Sample', color='#2ecc71')
    
    ax2.set_ylabel('Value')
    ax2.set_title('Aggregate Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_walk_forward_equity(
    wf_results: Dict,
    title: str = "Walk-Forward Analysis: Equity Curve",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot walk-forward equity curve.
    
    Args:
        wf_results: Dict from run_walk_forward_analysis
        title: Chart title
        save_path: Optional path to save
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    years = wf_results['equity_years']
    portfolio = wf_results['portfolio_equity']
    benchmark = wf_results['benchmark_equity']
    
    ax.plot(years, portfolio, 'g-', linewidth=2, label='Strategy Portfolio', marker='o')
    ax.plot(years, benchmark, 'b--', linewidth=2, label='S&P 500', marker='s', alpha=0.7)
    
    ax.axhline(y=10000, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Fill between
    ax.fill_between(years, portfolio, benchmark, 
                    where=[p > b for p, b in zip(portfolio, benchmark)],
                    alpha=0.2, color='green', label='Outperformance')
    ax.fill_between(years, portfolio, benchmark,
                    where=[p <= b for p, b in zip(portfolio, benchmark)],
                    alpha=0.2, color='red', label='Underperformance')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add summary stats
    stats_text = (
        f"CAGR: {wf_results['cagr']*100:.1f}%\n"
        f"Avg Alpha: {wf_results['avg_alpha']*100:+.1f}%\n"
        f"Win Rate: {wf_results['win_rate']*100:.0f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_regime_analysis(
    regime_analysis: Dict,
    title: str = "Performance by Market Regime",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot performance across market regimes.
    
    Args:
        regime_analysis: Dict of regime -> metrics
        title: Chart title
        save_path: Optional path to save
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regimes = list(regime_analysis.keys())
    avg_alphas = [regime_analysis[r]['avg_alpha'] * 100 for r in regimes]
    
    colors = ['#e74c3c' if a < 0 else '#2ecc71' for a in avg_alphas]
    
    bars = ax.bar(regimes, avg_alphas, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, alpha in zip(bars, avg_alphas):
        height = bar.get_height()
        ax.annotate(f'{alpha:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Average Alpha (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Run OOS validation
    validation_results = run_oos_validation(
        is_years=[2021, 2022, 2023, 2024, 2025],
        oos_years=[2015, 2016, 2017, 2018, 2019],
        show_progress=True
    )
    
    # Run walk-forward analysis
    wf_results = run_walk_forward_analysis(
        start_year=2010,
        end_year=2025,
        show_progress=True
    )
    
    # Generate visualizations
    plot_oos_comparison(validation_results, show=True)
    plot_walk_forward_equity(wf_results, show=True)
    plot_regime_analysis(validation_results['regime_analysis'], show=True)
