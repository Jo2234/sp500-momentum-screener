"""
Visualization Module

Generate charts and visualizations for backtest results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Output directory for charts
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_backtest_comparison(
    benchmark_return: float,
    portfolio_return: float,
    year: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a grouped bar chart comparing S&P 500 vs Strategy portfolio.
    
    Args:
        benchmark_return: S&P 500 return (as decimal)
        portfolio_return: Strategy portfolio return (as decimal)
        year: Year of the backtest
        title: Optional custom title
        save_path: Optional path to save the chart
        show: Whether to display the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    categories = ['S&P 500', 'Momentum Strategy']
    returns = [benchmark_return * 100, portfolio_return * 100]
    colors = ['#3498db', '#2ecc71']  # Blue for benchmark, green for strategy
    
    # Adjust color if strategy underperformed
    if portfolio_return < benchmark_return:
        colors[1] = '#e74c3c'  # Red for underperformance
    
    # Create bars
    bars = ax.bar(categories, returns, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.5 if height >= 0 else -0.5
        ax.annotate(
            f'{ret:.2f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset * 10),
            textcoords="offset points",
            ha='center', va=va,
            fontsize=14, fontweight='bold'
        )
    
    # Title and labels
    if title is None:
        title = f"Backtest Results: {year}"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Return (%)', fontsize=12)
    
    # Alpha annotation
    alpha = (portfolio_return - benchmark_return) * 100
    alpha_text = f"Alpha: {alpha:+.2f}%"
    alpha_color = '#2ecc71' if alpha >= 0 else '#e74c3c'
    ax.annotate(
        alpha_text,
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        ha='right', va='top',
        fontsize=12, fontweight='bold',
        color=alpha_color,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=alpha_color, alpha=0.8)
    )
    
    # Add zero line
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_multi_year_backtest(
    results_df: pd.DataFrame,
    title: str = "Multi-Year Backtest Results",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a grouped bar chart for multiple years of backtesting.
    
    Args:
        results_df: DataFrame with columns ['Year', 'Portfolio_Return', 'Benchmark_Return']
        title: Chart title
        save_path: Optional path to save the chart
        show: Whether to display the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Filter valid results
    df = results_df.dropna(subset=['Portfolio_Return', 'Benchmark_Return']).copy()
    
    if df.empty:
        ax.text(0.5, 0.5, 'No valid data available', ha='center', va='center', fontsize=14)
        return fig
    
    # Data
    years = df['Year'].astype(str).tolist()
    portfolio_returns = (df['Portfolio_Return'] * 100).tolist()
    benchmark_returns = (df['Benchmark_Return'] * 100).tolist()
    
    # Bar positions
    x = np.arange(len(years))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, benchmark_returns, width, label='S&P 500', 
                   color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, portfolio_returns, width, label='Momentum Strategy', 
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    # Color strategy bars based on outperformance
    for bar, port_ret, bench_ret in zip(bars2, portfolio_returns, benchmark_returns):
        if port_ret < bench_ret:
            bar.set_color('#e74c3c')
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha='center', va=va,
                fontsize=9, fontweight='bold'
            )
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11)
    
    # Legend
    ax.legend(loc='upper left', fontsize=11)
    
    # Add summary statistics
    avg_alpha = df['Alpha'].mean() * 100
    win_rate = (df['Alpha'] > 0).sum() / len(df) * 100
    
    stats_text = f"Avg Alpha: {avg_alpha:+.2f}% | Win Rate: {win_rate:.0f}%"
    ax.annotate(
        stats_text,
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        ha='right', va='top',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9)
    )
    
    # Add zero line
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_individual_stock_returns(
    individual_returns: Dict[str, float],
    year: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a horizontal bar chart showing individual stock returns.
    
    Args:
        individual_returns: Dictionary of ticker -> return
        year: Year of the backtest
        title: Optional custom title
        save_path: Optional path to save the chart
        show: Whether to display the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(individual_returns) * 0.5)))
    
    # Sort by return
    sorted_returns = sorted(individual_returns.items(), key=lambda x: x[1], reverse=True)
    tickers = [t[0] for t in sorted_returns]
    returns = [t[1] * 100 for t in sorted_returns]
    
    # Colors based on positive/negative
    colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in returns]
    
    # Create horizontal bars
    y_pos = np.arange(len(tickers))
    bars = ax.barh(y_pos, returns, color=colors, edgecolor='black', linewidth=0.5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers, fontsize=11)
    ax.invert_yaxis()  # Top-to-bottom
    
    # Add value labels
    for bar, ret in zip(bars, returns):
        width = bar.get_width()
        ha = 'left' if width >= 0 else 'right'
        offset = 5 if width >= 0 else -5
        ax.annotate(
            f'{ret:.1f}%',
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=ha, va='center',
            fontsize=10, fontweight='bold'
        )
    
    # Title
    if title is None:
        title = f"Individual Stock Returns: {year}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Return (%)', fontsize=11)
    
    # Add zero line
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_portfolio_movement(
    daily_values: pd.DataFrame,
    metrics: Dict,
    year: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot daily portfolio movement with peak and max drawdown annotations.
    
    Args:
        daily_values: DataFrame with 'Portfolio' and optionally 'SP500' columns
        metrics: Dictionary containing peak and drawdown information
        year: Year of the backtest
        title: Optional custom title
        save_path: Optional path to save the chart
        show: Whether to display the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot portfolio line
    portfolio = daily_values['Portfolio']
    ax.plot(portfolio.index, portfolio.values, 
            color='#2ecc71', linewidth=2, label='Momentum Portfolio', zorder=3)
    
    # Plot S&P 500 if available
    if 'SP500' in daily_values.columns:
        sp500 = daily_values['SP500']
        ax.plot(sp500.index, sp500.values, 
                color='#3498db', linewidth=2, label='S&P 500', alpha=0.8, zorder=2)
    
    # Add horizontal line at 100 (starting value)
    ax.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5, zorder=1)
    
    # Annotate PEAK (highest return point)
    peak_date = metrics['peak_date']
    peak_value = metrics['peak_value']
    peak_return = metrics['peak_return']
    
    ax.scatter([peak_date], [peak_value], color='#27ae60', s=150, zorder=5, 
               marker='^', edgecolors='black', linewidths=1)
    ax.annotate(
        f'📈 PEAK\n{peak_date.strftime("%b %d")}\n+{peak_return*100:.1f}%',
        xy=(peak_date, peak_value),
        xytext=(20, 20),
        textcoords='offset points',
        fontsize=10, fontweight='bold',
        color='#27ae60',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', edgecolor='#27ae60', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
        zorder=6
    )
    
    # Annotate MAX DRAWDOWN point
    dd_date = metrics['max_drawdown_date']
    dd_value = metrics['max_drawdown_value']
    max_drawdown = metrics['max_drawdown']
    
    ax.scatter([dd_date], [dd_value], color='#e74c3c', s=150, zorder=5,
               marker='v', edgecolors='black', linewidths=1)
    ax.annotate(
        f'📉 MAX DRAWDOWN\n{dd_date.strftime("%b %d")}\n{max_drawdown*100:.1f}%',
        xy=(dd_date, dd_value),
        xytext=(-20, -40),
        textcoords='offset points',
        fontsize=10, fontweight='bold',
        color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8', edgecolor='#e74c3c', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
        zorder=6
    )
    
    # Draw drawdown shading from peak to trough
    dd_peak_date = metrics.get('drawdown_peak_date', peak_date)
    dd_peak_value = metrics.get('drawdown_peak_value', peak_value)
    
    # Shade the drawdown region
    mask = (portfolio.index >= dd_peak_date) & (portfolio.index <= dd_date)
    if mask.any():
        ax.fill_between(portfolio.index[mask], portfolio.values[mask], 
                        dd_peak_value, alpha=0.2, color='#e74c3c', zorder=1)
    
    # Title and labels
    if title is None:
        title = f"Portfolio Movement: {year}"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value (Indexed to 100)', fontsize=12)
    
    # Final return annotation
    final_value = metrics['final_value']
    final_return = metrics['final_return']
    final_color = '#27ae60' if final_return >= 0 else '#e74c3c'
    
    stats_text = f"Final Return: {final_return*100:+.2f}%\nMax Drawdown: {max_drawdown*100:.2f}%\nPeak Return: +{peak_return*100:.2f}%"
    ax.annotate(
        stats_text,
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        ha='left', va='top',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9)
    )
    
    # Legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_multi_year_movements(
    years_data: List[Tuple[int, pd.DataFrame, Dict]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a grid of portfolio movement charts for multiple years.
    
    Args:
        years_data: List of (year, daily_values_df, metrics_dict) tuples
        save_path: Optional path to save the chart
        show: Whether to display the chart
        
    Returns:
        Matplotlib figure object
    """
    n_years = len(years_data)
    
    # Calculate grid dimensions
    n_cols = min(3, n_years)
    n_rows = (n_years + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Flatten axes for easy iteration
    if n_years == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (year, daily_values, metrics) in enumerate(years_data):
        ax = axes[idx]
        
        # Plot portfolio line
        portfolio = daily_values['Portfolio']
        ax.plot(portfolio.index, portfolio.values, 
                color='#2ecc71', linewidth=1.5, label='Portfolio')
        
        # Plot S&P 500 if available
        if 'SP500' in daily_values.columns:
            sp500 = daily_values['SP500']
            ax.plot(sp500.index, sp500.values, 
                    color='#3498db', linewidth=1.5, label='S&P 500', alpha=0.7)
        
        # Baseline
        ax.axhline(y=100, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        
        # Mark peak
        peak_date = metrics['peak_date']
        peak_value = metrics['peak_value']
        ax.scatter([peak_date], [peak_value], color='#27ae60', s=80, 
                   marker='^', edgecolors='black', linewidths=0.5, zorder=5)
        
        # Mark max drawdown
        dd_date = metrics['max_drawdown_date']
        dd_value = metrics['max_drawdown_value']
        ax.scatter([dd_date], [dd_value], color='#e74c3c', s=80,
                   marker='v', edgecolors='black', linewidths=0.5, zorder=5)
        
        # Title with key stats
        final_return = metrics['final_return']
        max_drawdown = metrics['max_drawdown']
        title_color = '#27ae60' if final_return >= 0 else '#e74c3c'
        ax.set_title(f"{year}: {final_return*100:+.1f}% (DD: {max_drawdown*100:.1f}%)", 
                     fontsize=12, fontweight='bold', color=title_color)
        
        ax.set_ylabel('Value (100=start)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Rotate x-axis labels
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    # Hide unused subplots
    for idx in range(len(years_data), len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend to first plot
    if len(years_data) > 0:
        axes[0].legend(loc='upper left', fontsize=9)
    
    plt.suptitle('Portfolio Movement by Year (▲ Peak, ▼ Max Drawdown)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _ensure_output_dir()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def save_chart(fig: plt.Figure, filename: str) -> str:
    """
    Save a matplotlib figure to the output directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Filename (with or without extension)
        
    Returns:
        Full path to the saved file
    """
    _ensure_output_dir()
    
    if not filename.endswith('.png'):
        filename += '.png'
    
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {filepath}")
    
    return str(filepath)


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization...")
    
    # Test single year chart
    plot_backtest_comparison(
        benchmark_return=0.15,
        portfolio_return=0.25,
        year=2024,
        show=True
    )
    
    # Test multi-year chart
    test_df = pd.DataFrame({
        'Year': [2020, 2021, 2022, 2023, 2024],
        'Portfolio_Return': [0.30, 0.45, -0.10, 0.35, 0.28],
        'Benchmark_Return': [0.18, 0.28, -0.18, 0.26, 0.25],
        'Alpha': [0.12, 0.17, 0.08, 0.09, 0.03]
    })
    
    plot_multi_year_backtest(test_df, show=True)
