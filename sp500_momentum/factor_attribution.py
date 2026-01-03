"""
Factor Attribution & Sector Analysis Module

Analyze whether strategy alpha comes from sector timing or stock selection,
and measure portfolio diversification via correlation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .data_fetcher import fetch_batch_data, get_sp500_index_data


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# GICS Sector mapping for major S&P 500 stocks
SECTOR_MAPPING = {
    # Information Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'AVGO': 'Technology', 'ORCL': 'Technology', 'CRM': 'Technology',
    'AMD': 'Technology', 'ADBE': 'Technology', 'CSCO': 'Technology',
    'ACN': 'Technology', 'INTC': 'Technology', 'IBM': 'Technology',
    'QCOM': 'Technology', 'TXN': 'Technology', 'AMAT': 'Technology',
    'NOW': 'Technology', 'INTU': 'Technology', 'ADI': 'Technology',
    'MU': 'Technology', 'LRCX': 'Technology', 'KLAC': 'Technology',
    'SNPS': 'Technology', 'CDNS': 'Technology', 'MRVL': 'Technology',
    'ANET': 'Technology', 'FTNT': 'Technology', 'PANW': 'Technology',
    'MPWR': 'Technology', 'NXPI': 'Technology', 'MCHP': 'Technology',
    'ON': 'Technology', 'APH': 'Technology', 'TEL': 'Technology',
    'KEYS': 'Technology', 'SWKS': 'Technology', 'ZBRA': 'Technology',
    'FFIV': 'Technology', 'JNPR': 'Technology', 'NTAP': 'Technology',
    
    # Industrials
    'GE': 'Industrials', 'CAT': 'Industrials', 'HON': 'Industrials',
    'UNP': 'Industrials', 'RTX': 'Industrials', 'BA': 'Industrials',
    'UPS': 'Industrials', 'LMT': 'Industrials', 'DE': 'Industrials',
    'GD': 'Industrials', 'MMM': 'Industrials', 'CSX': 'Industrials',
    'NSC': 'Industrials', 'ITW': 'Industrials', 'EMR': 'Industrials',
    'ETN': 'Industrials', 'PH': 'Industrials', 'PCAR': 'Industrials',
    'FDX': 'Industrials', 'WM': 'Industrials', 'RSG': 'Industrials',
    'JCI': 'Industrials', 'ROK': 'Industrials', 'CMI': 'Industrials',
    'FAST': 'Industrials', 'ODFL': 'Industrials', 'URI': 'Industrials',
    'PWR': 'Industrials', 'TT': 'Industrials', 'VRSK': 'Industrials',
    'FIX': 'Industrials', 'EME': 'Industrials', 'IR': 'Industrials',
    'SWK': 'Industrials', 'GWW': 'Industrials', 'RHI': 'Industrials',
    
    # Health Care
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare',
    'PFE': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare',
    'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
    'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'ISRG': 'Healthcare', 'SYK': 'Healthcare', 'MDT': 'Healthcare',
    'CVS': 'Healthcare', 'CI': 'Healthcare', 'ELV': 'Healthcare',
    'ZTS': 'Healthcare', 'VRTX': 'Healthcare', 'REGN': 'Healthcare',
    'BSX': 'Healthcare', 'BDX': 'Healthcare', 'MCK': 'Healthcare',
    'HCA': 'Healthcare', 'DXCM': 'Healthcare', 'IQV': 'Healthcare',
    
    # Financials
    'BRK.B': 'Financials', 'JPM': 'Financials', 'V': 'Financials',
    'MA': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'GS': 'Financials', 'MS': 'Financials', 'BLK': 'Financials',
    'SCHW': 'Financials', 'C': 'Financials', 'AXP': 'Financials',
    'CB': 'Financials', 'PGR': 'Financials', 'CME': 'Financials',
    'ICE': 'Financials', 'MMC': 'Financials', 'AON': 'Financials',
    'MCO': 'Financials', 'SPGI': 'Financials', 'MET': 'Financials',
    'AIG': 'Financials', 'TRV': 'Financials', 'AFL': 'Financials',
    'ALL': 'Financials', 'USB': 'Financials', 'PNC': 'Financials',
    'ARES': 'Financials', 'KKR': 'Financials', 'BX': 'Financials',
    
    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
    'BKNG': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary',
    'MAR': 'Consumer Discretionary', 'HLT': 'Consumer Discretionary',
    'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary',
    'ORLY': 'Consumer Discretionary', 'AZO': 'Consumer Discretionary',
    'ROST': 'Consumer Discretionary', 'DHI': 'Consumer Discretionary',
    'LEN': 'Consumer Discretionary', 'PHM': 'Consumer Discretionary',
    'DECK': 'Consumer Discretionary', 'NVR': 'Consumer Discretionary',
    
    # Communication Services
    'GOOGL': 'Communication Services', 'GOOG': 'Communication Services',
    'META': 'Communication Services', 'NFLX': 'Communication Services',
    'DIS': 'Communication Services', 'CMCSA': 'Communication Services',
    'VZ': 'Communication Services', 'T': 'Communication Services',
    'TMUS': 'Communication Services', 'CHTR': 'Communication Services',
    'EA': 'Communication Services', 'TTWO': 'Communication Services',
    
    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'COST': 'Consumer Staples',
    'WMT': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
    'CL': 'Consumer Staples', 'KMB': 'Consumer Staples',
    'GIS': 'Consumer Staples', 'K': 'Consumer Staples',
    'HSY': 'Consumer Staples', 'STZ': 'Consumer Staples',
    'KHC': 'Consumer Staples', 'SYY': 'Consumer Staples',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy',
    'PSX': 'Energy', 'VLO': 'Energy', 'PXD': 'Energy',
    'OXY': 'Energy', 'HAL': 'Energy', 'DVN': 'Energy',
    'HES': 'Energy', 'BKR': 'Energy', 'FANG': 'Energy',
    
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
    'D': 'Utilities', 'AEP': 'Utilities', 'EXC': 'Utilities',
    'SRE': 'Utilities', 'XEL': 'Utilities', 'ED': 'Utilities',
    'WEC': 'Utilities', 'ES': 'Utilities', 'AWK': 'Utilities',
    'CEG': 'Utilities', 'PCG': 'Utilities', 'EIX': 'Utilities',
    
    # Real Estate
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate',
    'EQIX': 'Real Estate', 'PSA': 'Real Estate', 'SPG': 'Real Estate',
    'O': 'Real Estate', 'WELL': 'Real Estate', 'DLR': 'Real Estate',
    'AVB': 'Real Estate', 'EQR': 'Real Estate', 'VICI': 'Real Estate',
    
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
    'FCX': 'Materials', 'ECL': 'Materials', 'NEM': 'Materials',
    'NUE': 'Materials', 'DD': 'Materials', 'DOW': 'Materials',
    'VMC': 'Materials', 'MLM': 'Materials', 'PPG': 'Materials',
}


# Sector ETF mapping  
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Industrials': 'XLI',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
}


def get_sector(ticker: str) -> str:
    """Get GICS sector for a ticker."""
    return SECTOR_MAPPING.get(ticker, 'Unknown')


def calculate_sector_returns(
    year: int,
    workers: int = 10
) -> Dict[str, float]:
    """
    Calculate returns for all sector ETFs for a given year.
    
    Args:
        year: Year to calculate returns for
        workers: Number of parallel workers
        
    Returns:
        Dict of sector -> return
    """
    etf_tickers = list(SECTOR_ETFS.values())
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    etf_data = fetch_batch_data(
        etf_tickers, start_date, end_date,
        workers=workers, show_progress=False
    )
    
    sector_returns = {}
    for sector, etf in SECTOR_ETFS.items():
        if etf in etf_data and not etf_data[etf].empty:
            df = etf_data[etf]
            if len(df) >= 2:
                first_price = df['Close'].iloc[0]
                last_price = df['Close'].iloc[-1]
                sector_returns[sector] = (last_price - first_price) / first_price
    
    return sector_returns


def run_factor_attribution(
    tickers: List[str],
    year: int,
    workers: int = 10,
    show_progress: bool = True
) -> Dict:
    """
    Run factor attribution analysis on a portfolio.
    
    Args:
        tickers: List of portfolio tickers
        year: Year to analyze
        workers: Number of parallel workers
        show_progress: Whether to print progress
        
    Returns:
        Dict with attribution results
    """
    if show_progress:
        print("\n" + "="*70)
        print("FACTOR ATTRIBUTION ANALYSIS")
        print("="*70)
        print(f"Portfolio: {tickers}")
        print(f"Analysis Year: {year}")
    
    # Fetch stock data
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    stock_data = fetch_batch_data(
        tickers, start_date, end_date,
        workers=workers, show_progress=False
    )
    
    # Get sector returns
    if show_progress:
        print("\nFetching sector ETF returns...")
    sector_returns = calculate_sector_returns(year, workers)
    
    # Get benchmark return
    sp500_data = get_sp500_index_data(start_date, end_date)
    if sp500_data is not None and len(sp500_data) >= 2:
        benchmark_return = (sp500_data['Close'].iloc[-1] - sp500_data['Close'].iloc[0]) / sp500_data['Close'].iloc[0]
    else:
        benchmark_return = 0
    
    # Calculate individual stock metrics
    stock_analysis = []
    
    for ticker in tickers:
        if ticker not in stock_data or stock_data[ticker].empty:
            continue
            
        df = stock_data[ticker]
        if len(df) < 2:
            continue
        
        # Stock return
        first_price = df['Close'].iloc[0]
        last_price = df['Close'].iloc[-1]
        stock_return = (last_price - first_price) / first_price
        
        # Sector info
        sector = get_sector(ticker)
        sector_return = sector_returns.get(sector, 0)
        
        # Residual alpha (stock return - sector return)
        residual_alpha = stock_return - sector_return
        
        # Sector contribution (sector return - benchmark return)
        sector_contribution = sector_return - benchmark_return
        
        stock_analysis.append({
            'ticker': ticker,
            'sector': sector,
            'stock_return': stock_return,
            'sector_return': sector_return,
            'residual_alpha': residual_alpha,
            'sector_contribution': sector_contribution,
            'vs_benchmark': stock_return - benchmark_return
        })
    
    if not stock_analysis:
        return {'error': 'No valid stock data'}
    
    analysis_df = pd.DataFrame(stock_analysis)
    
    # Portfolio level aggregation
    n_stocks = len(analysis_df)
    portfolio_return = analysis_df['stock_return'].mean()
    total_alpha = portfolio_return - benchmark_return
    
    # Decompose alpha
    avg_sector_contribution = analysis_df['sector_contribution'].mean()
    avg_residual_alpha = analysis_df['residual_alpha'].mean()
    
    # Sector exposure analysis
    sector_exposure = analysis_df.groupby('sector').agg({
        'ticker': 'count',
        'stock_return': 'mean',
        'sector_return': 'first',
        'residual_alpha': 'mean'
    }).rename(columns={'ticker': 'count'})
    sector_exposure['weight'] = sector_exposure['count'] / n_stocks
    
    # Calculate daily returns for correlation matrix
    if show_progress:
        print("\nCalculating correlation matrix...")
    
    daily_returns = {}
    for ticker in tickers:
        if ticker in stock_data and not stock_data[ticker].empty:
            df = stock_data[ticker]
            daily_returns[ticker] = df['Close'].pct_change().dropna()
    
    if daily_returns:
        returns_df = pd.DataFrame(daily_returns)
        correlation_matrix = returns_df.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    else:
        correlation_matrix = pd.DataFrame()
        avg_correlation = 0
    
    # Portfolio diversification assessment
    if avg_correlation > 0.8:
        diversification_warning = "CRITICAL: Portfolio highly correlated (>0.8) - single point of failure risk"
    elif avg_correlation > 0.6:
        diversification_warning = "WARNING: Portfolio moderately correlated (0.6-0.8) - limited diversification"
    else:
        diversification_warning = "OK: Portfolio reasonably diversified (<0.6 correlation)"
    
    results = {
        'analysis_df': analysis_df,
        'sector_exposure': sector_exposure,
        'correlation_matrix': correlation_matrix,
        'benchmark_return': benchmark_return,
        'portfolio_return': portfolio_return,
        'total_alpha': total_alpha,
        'sector_timing_alpha': avg_sector_contribution,
        'stock_selection_alpha': avg_residual_alpha,
        'avg_correlation': avg_correlation,
        'diversification_warning': diversification_warning,
        'sector_returns': sector_returns,
        'year': year
    }
    
    if show_progress:
        print("\n" + "-"*50)
        print("ATTRIBUTION BREAKDOWN")
        print("-"*50)
        print(f"\nS&P 500 Benchmark Return: {benchmark_return*100:.2f}%")
        print(f"Portfolio Return:         {portfolio_return*100:.2f}%")
        print(f"Total Alpha:              {total_alpha*100:+.2f}%")
        print(f"\nAlpha Decomposition:")
        print(f"  Sector Timing:          {avg_sector_contribution*100:+.2f}%")
        print(f"  Stock Selection:        {avg_residual_alpha*100:+.2f}%")
        print(f"\nDiversification:")
        print(f"  Avg Pairwise Correlation: {avg_correlation:.3f}")
        print(f"  Assessment: {diversification_warning}")
        print(f"\nSector Exposure:")
        for sector, row in sector_exposure.iterrows():
            print(f"  {sector}: {row['count']} stocks ({row['weight']*100:.0f}%)")
        print(f"\nIndividual Stock Analysis:")
        for _, row in analysis_df.iterrows():
            print(f"  {row['ticker']:6} ({row['sector'][:8]:8}): "
                  f"Return={row['stock_return']*100:+.1f}%, "
                  f"Sector={row['sector_return']*100:+.1f}%, "
                  f"Residual={row['residual_alpha']*100:+.1f}%")
    
    return results


def plot_waterfall_attribution(
    results: Dict,
    title: str = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate a waterfall chart showing alpha attribution.
    
    Args:
        results: Dict from run_factor_attribution
        title: Chart title
        save_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    year = results['year']
    benchmark = results['benchmark_return'] * 100
    sector_alpha = results['sector_timing_alpha'] * 100
    stock_alpha = results['stock_selection_alpha'] * 100
    total_return = results['portfolio_return'] * 100
    
    # Waterfall components
    categories = ['S&P 500\nBenchmark', 'Sector\nTiming', 'Stock\nSelection', 'Portfolio\nReturn']
    values = [benchmark, sector_alpha, stock_alpha, total_return]
    
    # Calculate positions
    cumulative = [benchmark, benchmark + sector_alpha, benchmark + sector_alpha + stock_alpha, total_return]
    bottoms = [0, benchmark, benchmark + sector_alpha, 0]
    
    # Colors
    colors = ['#3498db', '#27ae60' if sector_alpha >= 0 else '#e74c3c', 
              '#27ae60' if stock_alpha >= 0 else '#e74c3c', '#2c3e50']
    
    # Create bars
    bars = ax.bar(categories, [benchmark, abs(sector_alpha), abs(stock_alpha), total_return],
                  bottom=[0, min(benchmark, benchmark + sector_alpha), 
                         min(benchmark + sector_alpha, cumulative[2]), 0],
                  color=colors, edgecolor='black', linewidth=0.5)
    
    # For the intermediate bars, we need special handling
    ax.bar(categories[1], sector_alpha, bottom=benchmark if sector_alpha >= 0 else benchmark + sector_alpha,
           color='#27ae60' if sector_alpha >= 0 else '#e74c3c', edgecolor='black', linewidth=0.5)
    ax.bar(categories[2], stock_alpha, bottom=cumulative[1] if stock_alpha >= 0 else cumulative[2],
           color='#27ae60' if stock_alpha >= 0 else '#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Add connecting lines
    for i in range(len(categories) - 1):
        if i == 0:
            ax.hlines(y=benchmark, xmin=i - 0.4, xmax=i + 1.4, color='gray', linestyle='--', linewidth=1)
        elif i == 1:
            ax.hlines(y=cumulative[1], xmin=i - 0.4, xmax=i + 1.4, color='gray', linestyle='--', linewidth=1)
    
    # Add value labels
    for i, (cat, val, cum) in enumerate(zip(categories, values, [benchmark, cumulative[1], cumulative[2], total_return])):
        if i == 0 or i == 3:
            ax.text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            sign = '+' if val >= 0 else ''
            y_pos = cum + 1 if val >= 0 else cum - 2
            ax.text(i, y_pos, f'{sign}{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Title
    if title is None:
        title = f"Alpha Attribution Waterfall ({year})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Return (%)', fontsize=12)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Benchmark'),
        Patch(facecolor='#27ae60', label='Positive Contribution'),
        Patch(facecolor='#e74c3c', label='Negative Contribution'),
        Patch(facecolor='#2c3e50', label='Final Result')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    title: str = "Portfolio Correlation Matrix",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot correlation heatmap for portfolio stocks.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        title: Chart title
        save_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=0.5,
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_sector_breakdown(
    analysis_df: pd.DataFrame,
    title: str = "Sector Attribution Breakdown",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot horizontal bar chart showing stock returns decomposed by sector.
    
    Args:
        analysis_df: DataFrame from factor attribution analysis
        title: Chart title
        save_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by total return
    df = analysis_df.sort_values('stock_return', ascending=True)
    
    y_pos = np.arange(len(df))
    
    # Plot sector contribution
    bars1 = ax.barh(y_pos, df['sector_return'] * 100, 
                    label='Sector Return', color='#3498db', alpha=0.7)
    
    # Plot residual alpha on top
    bars2 = ax.barh(y_pos, df['residual_alpha'] * 100, 
                    left=df['sector_return'] * 100,
                    label='Residual Alpha', 
                    color=['#27ae60' if r > 0 else '#e74c3c' for r in df['residual_alpha']],
                    alpha=0.8)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['ticker']} ({row['sector'][:4]})" for _, row in df.iterrows()])
    
    # Add total return annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['stock_return'] * 100 + 1, i, 
                f"{row['stock_return']*100:+.1f}%",
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Return (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    
    # Add zero line
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
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
    # Test factor attribution
    test_tickers = ['FIX', 'AVGO', 'ANET', 'APH', 'EME', 'PWR', 'LLY', 'AXP', 'JPM', 'ORCL']
    
    results = run_factor_attribution(
        tickers=test_tickers,
        year=2025,
        show_progress=True
    )
    
    plot_waterfall_attribution(results, show=True)
    plot_correlation_matrix(results['correlation_matrix'], show=True)
    plot_sector_breakdown(results['analysis_df'], show=True)
