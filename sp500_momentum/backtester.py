"""
Backtesting Engine

Run historical backtests to evaluate the momentum strategy's performance
against the S&P 500 benchmark.
"""

from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .data_fetcher import fetch_batch_data, get_sp500_index_data
from .strategy import run_momentum_screen


def apply_round_trip_costs(
    gross_return: float,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> float:
    """Apply per-side transaction cost and slippage to a buy-and-sell return.

    Costs are modeled once on entry and once on exit. For example, 10 bps of
    transaction costs plus 5 bps of slippage creates a 15 bps drag at entry and
    another 15 bps drag at exit.
    """
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("transaction_cost_bps and slippage_bps must be non-negative")

    per_side_cost = (transaction_cost_bps + slippage_bps) / 10_000
    return (1 - per_side_cost) * (1 + gross_return) * (1 - per_side_cost) - 1


def calculate_holding_period_end(start_date: date, holding_period_months: int) -> date:
    """Calculate the nominal end date for a month-based holding period."""
    if holding_period_months <= 0:
        raise ValueError("holding_period_months must be positive")

    return date(
        start_date.year + (start_date.month + holding_period_months - 1) // 12,
        ((start_date.month + holding_period_months - 1) % 12) + 1,
        min(start_date.day, 28),
    )


def calculate_portfolio_return(
    tickers: List[str],
    start_date: date,
    end_date: date,
    equal_weight: bool = True,
    workers: int = 10,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the return of an equal-weighted portfolio.

    Args:
        tickers: List of ticker symbols in the portfolio
        start_date: Portfolio start date
        end_date: Portfolio end date
        equal_weight: If True, average over successfully fetched tickers
        workers: Number of parallel workers
        transaction_cost_bps: Per-side transaction/commission cost in basis points
        slippage_bps: Per-side slippage estimate in basis points

    Returns:
        Tuple of (portfolio_return, individual_returns_dict)
    """
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
                    gross_return = (last_price - first_price) / first_price
                    individual_returns[ticker] = apply_round_trip_costs(
                        gross_return,
                        transaction_cost_bps=transaction_cost_bps,
                        slippage_bps=slippage_bps,
                    )

    if not individual_returns:
        return 0.0, {}

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
    stock_data = fetch_batch_data(
        tickers, start_date, end_date,
        workers=workers, show_progress=False
    )

    price_dfs = []
    for ticker in tickers:
        if ticker in stock_data and not stock_data[ticker].empty:
            df = stock_data[ticker][['Close']].copy()
            df.columns = [ticker]
            price_dfs.append(df)

    if not price_dfs:
        return pd.DataFrame(), {}

    prices = pd.concat(price_dfs, axis=1)
    prices = prices.ffill().bfill()

    normalized = prices / prices.iloc[0] * 100

    portfolio_values = normalized.mean(axis=1)
    portfolio_values.name = 'Portfolio'

    sp500_data = get_sp500_index_data(start_date, end_date)
    if sp500_data is not None and not sp500_data.empty:
        sp500_normalized = sp500_data['Close'] / sp500_data['Close'].iloc[0] * 100
        sp500_normalized.name = 'SP500'
    else:
        sp500_normalized = None

    result_df = pd.DataFrame(portfolio_values)
    if sp500_normalized is not None:
        result_df = result_df.join(sp500_normalized, how='outer')
        result_df = result_df.ffill().bfill()

    peak_idx = portfolio_values.idxmax()
    peak_value = portfolio_values.max()
    peak_return = (peak_value - 100) / 100

    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max

    max_drawdown_idx = drawdown.idxmin()
    max_drawdown = drawdown.min()
    max_drawdown_value = portfolio_values.loc[max_drawdown_idx]

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


def _round_for_manifest(value: Any) -> Any:
    """Convert result values to stable JSON-friendly values for manifests."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(Decimal(str(value)).quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP))
    if isinstance(value, dict):
        return {str(k): _round_for_manifest(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_round_for_manifest(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [_round_for_manifest(row) for row in value.to_dict(orient="records")]
    return value


def _git_commit() -> Optional[str]:
    """Return current git commit if available."""
    try:
        repo_root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_backtest_manifest(
    result: Dict,
    parameters: Dict[str, Any],
    output_files: Optional[List[Path]] = None,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a stable, sorted manifest describing a backtest run and outputs."""
    output_entries = []
    for path in output_files or []:
        path = Path(path)
        if path.exists() and path.is_file():
            output_entries.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })

    summary = {
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date"),
        "year": result.get("year"),
        "selected_stocks": result.get("selected_stocks", []),
        "portfolio_return": result.get("portfolio_return"),
        "benchmark_return": result.get("benchmark_return"),
        "alpha": result.get("alpha"),
        "individual_returns": result.get("individual_returns", {}),
        "selection_returns": result.get("selection_returns", {}),
        "error": result.get("error"),
    }

    return _round_for_manifest({
        "schema_version": 1,
        "command": command,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "parameters": parameters,
        "result": summary,
        "outputs": sorted(output_entries, key=lambda item: item["path"]),
    })


def save_backtest_manifest(manifest: Dict[str, Any], path: Path) -> Path:
    """Write a deterministic JSON backtest manifest with sorted keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_backtest(
    strategy_start_date: date,
    lookback_years: int = 5,
    top_n: int = 10,
    holding_period_months: int = 12,
    workers: int = 10,
    show_progress: bool = True,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Dict:
    """
    Run a backtest for the momentum strategy.

    The engine looks back `lookback_years` from the start date to identify
    stocks, then tracks performance for the following `holding_period_months`.
    Screening data ends before the strategy start year, avoiding look-ahead into
    the holding period.
    """
    if isinstance(strategy_start_date, str):
        strategy_start_date = pd.to_datetime(strategy_start_date).date()

    if show_progress:
        print(f"\n{'#'*60}")
        print(f"BACKTEST: Strategy Start Date = {strategy_start_date}")
        print(f"{'#'*60}")

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
            'transaction_cost_bps': transaction_cost_bps,
            'slippage_bps': slippage_bps,
            'error': 'No stocks passed the screening filter'
        }

    selected_tickers = [t[0] for t in top_stocks]
    selection_returns = {t[0]: t[2] for t in top_stocks}

    if show_progress:
        print(f"\nSelected {len(selected_tickers)} stocks: {selected_tickers}")

    portfolio_start = strategy_start_date
    portfolio_end = calculate_holding_period_end(strategy_start_date, holding_period_months)

    if show_progress:
        print(f"\nHolding Period: {portfolio_start} to {portfolio_end}")

    portfolio_return, individual_returns = calculate_portfolio_return(
        selected_tickers,
        portfolio_start,
        portfolio_end,
        workers=workers,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )

    sp500_data = get_sp500_index_data(portfolio_start, portfolio_end)
    if sp500_data is not None and len(sp500_data) >= 2:
        first_price = sp500_data['Close'].iloc[0]
        last_price = sp500_data['Close'].iloc[-1]
        benchmark_return = (last_price - first_price) / first_price
    else:
        benchmark_return = 0.0

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
        'summary_df': summary_df,
        'transaction_cost_bps': transaction_cost_bps,
        'slippage_bps': slippage_bps,
    }

    if show_progress:
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Portfolio Return: {portfolio_return*100:.2f}%")
        if transaction_cost_bps or slippage_bps:
            print(f"Costs:            {transaction_cost_bps:.2f} bps commission + {slippage_bps:.2f} bps slippage per side")
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
    show_progress: bool = True,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Run backtests for multiple years and compile results.
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
            show_progress=show_progress,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )

        all_results.append({
            'Year': year,
            'Portfolio_Return': result.get('portfolio_return'),
            'Benchmark_Return': result.get('benchmark_return'),
            'Alpha': result.get('alpha'),
            'Transaction_Cost_Bps': transaction_cost_bps,
            'Slippage_Bps': slippage_bps,
            'Stocks': ', '.join(result.get('selected_stocks', []))
        })

    results_df = pd.DataFrame(all_results)

    if show_progress and not results_df.empty:
        print(f"\n\n{'='*60}")
        print("MULTI-YEAR BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(results_df.to_string())

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
    print("Testing backtester...")
    result = run_backtest(
        strategy_start_date=date(2024, 1, 1),
        lookback_years=5,
        top_n=10
    )
    print("\n\nBacktest complete!")
