"""
S&P 500 Momentum Stock Screener & Backtesting Engine

A Python-based stock screening and backtesting engine that identifies
high-performance momentum stocks within the S&P 500 universe.
"""

from .constituents import get_constituents_at_date, download_historical_constituents
from .data_fetcher import fetch_stock_data, fetch_batch_data, get_sp500_index_data
from .strategy import (
    calculate_annual_return,
    filter_consistent_outperformers,
    calculate_momentum_score,
    select_top_stocks,
    run_momentum_screen
)
from .backtester import run_backtest, calculate_portfolio_return, calculate_daily_portfolio_values
from .visualization import (
    plot_backtest_comparison,
    plot_multi_year_backtest,
    plot_portfolio_movement,
    plot_multi_year_movements
)
from .monte_carlo import (
    run_monte_carlo,
    plot_equity_curves,
    plot_distribution
)
from .sensitivity_analysis import (
    run_sensitivity_analysis,
    plot_sensitivity_heatmap,
    plot_stock_stability
)
from .factor_attribution import (
    run_factor_attribution,
    plot_waterfall_attribution,
    plot_correlation_matrix,
    plot_sector_breakdown
)
from .oos_validation import (
    run_oos_validation,
    run_walk_forward_analysis,
    plot_oos_comparison,
    plot_walk_forward_equity,
    plot_regime_analysis
)

__version__ = "1.4.0"
__all__ = [
    "get_constituents_at_date",
    "download_historical_constituents",
    "fetch_stock_data",
    "fetch_batch_data",
    "get_sp500_index_data",
    "calculate_annual_return",
    "filter_consistent_outperformers",
    "calculate_momentum_score",
    "select_top_stocks",
    "run_momentum_screen",
    "run_backtest",
    "calculate_portfolio_return",
    "plot_backtest_comparison",
    "plot_multi_year_backtest",
]
