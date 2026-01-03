#!/usr/bin/env python3
"""
S&P 500 Momentum Stock Screener & Backtesting Engine

CLI interface for running stock screening and backtesting.

Usage:
    python main.py --mode screen              # Screen for top 10 stocks for 2026
    python main.py --mode backtest --start-date 2021-01-01  # Run backtest
    python main.py --mode multi-backtest --years 2020 2021 2022 2023 2024  # Multi-year
"""

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent))

from sp500_momentum.constituents import download_historical_constituents
from sp500_momentum.strategy import run_momentum_screen
from sp500_momentum.backtester import run_backtest, run_multi_year_backtest, calculate_daily_portfolio_values
from sp500_momentum.visualization import (
    plot_backtest_comparison,
    plot_multi_year_backtest,
    plot_individual_stock_returns,
    plot_portfolio_movement,
    plot_multi_year_movements,
    OUTPUT_DIR
)
from sp500_momentum.monte_carlo import (
    run_monte_carlo,
    plot_equity_curves,
    plot_distribution
)
from sp500_momentum.sensitivity_analysis import (
    run_sensitivity_analysis,
    plot_sensitivity_heatmap,
    plot_stock_stability
)


def run_screen_mode(args):
    """Run stock screening for current/next year."""
    analysis_date = date.today()
    
    print("\n" + "="*70)
    print("S&P 500 MOMENTUM STOCK SCREENER")
    print("="*70)
    print(f"Analysis Date: {analysis_date}")
    print(f"Looking back {args.lookback} years to identify top {args.top_n} stocks")
    
    top_stocks, benchmark_returns, summary_df = run_momentum_screen(
        analysis_date=analysis_date,
        lookback_years=args.lookback,
        top_n=args.top_n,
        workers=args.workers,
        show_progress=True
    )
    
    if top_stocks:
        print("\n" + "="*70)
        print(f"RECOMMENDED PORTFOLIO FOR {analysis_date.year}")
        print("="*70)
        
        for i, (ticker, score, returns) in enumerate(top_stocks, 1):
            print(f"\n{i:2}. {ticker}")
            print(f"    Momentum Score: {score*100:.2f}%")
            print(f"    Historical Returns:")
            for y, r in sorted(returns.items()):
                bench = benchmark_returns.get(y, 0)
                diff = r - bench
                print(f"      {y}: {r*100:+.2f}% (vs S&P {bench*100:+.2f}%, alpha: {diff*100:+.2f}%)")
        
        # Save summary
        if not summary_df.empty:
            output_file = OUTPUT_DIR / f"screen_{analysis_date.year}.csv"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_file, index=False)
            print(f"\nSummary saved to: {output_file}")
    else:
        print("\nNo stocks passed the screening criteria!")


def run_backtest_mode(args):
    """Run single-year backtest."""
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: Invalid date format '{args.start_date}'. Use YYYY-MM-DD.")
        return
    
    print("\n" + "="*70)
    print("S&P 500 MOMENTUM STRATEGY BACKTEST")
    print("="*70)
    
    result = run_backtest(
        strategy_start_date=start_date,
        lookback_years=args.lookback,
        top_n=args.top_n,
        workers=args.workers,
        show_progress=True
    )
    
    if result.get('portfolio_return') is not None:
        # Generate visualization
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = OUTPUT_DIR / f"backtest_{result['year']}.png"
        
        fig = plot_backtest_comparison(
            benchmark_return=result['benchmark_return'],
            portfolio_return=result['portfolio_return'],
            year=result['year'],
            save_path=str(chart_path),
            show=not args.no_show
        )
        
        # Also plot individual returns
        if result.get('individual_returns'):
            ind_chart_path = OUTPUT_DIR / f"backtest_{result['year']}_stocks.png"
            plot_individual_stock_returns(
                individual_returns=result['individual_returns'],
                year=result['year'],
                save_path=str(ind_chart_path),
                show=not args.no_show
            )
        
        # Save result to CSV
        if 'summary_df' in result and not result['summary_df'].empty:
            csv_path = OUTPUT_DIR / f"backtest_{result['year']}_stocks.csv"
            result['summary_df'].to_csv(csv_path, index=False)
            print(f"\nStock details saved to: {csv_path}")


def run_multi_backtest_mode(args):
    """Run multi-year backtest."""
    if not args.years:
        print("Error: No years specified. Use --years 2020 2021 2022 ...")
        return
    
    years = [int(y) for y in args.years]
    
    print("\n" + "="*70)
    print("S&P 500 MOMENTUM STRATEGY MULTI-YEAR BACKTEST")
    print("="*70)
    print(f"Years: {years}")
    
    results_df = run_multi_year_backtest(
        start_years=years,
        lookback_years=args.lookback,
        top_n=args.top_n,
        workers=args.workers,
        show_progress=True
    )
    
    if not results_df.empty:
        # Generate visualization
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        years_str = f"{min(years)}-{max(years)}"
        chart_path = OUTPUT_DIR / f"multi_backtest_{years_str}.png"
        
        plot_multi_year_backtest(
            results_df=results_df,
            title=f"Momentum Strategy Backtest: {years_str}",
            save_path=str(chart_path),
            show=not args.no_show
        )
        
        # Save results to CSV
        csv_path = OUTPUT_DIR / f"multi_backtest_{years_str}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")


def run_movement_analysis_mode(args):
    """Run portfolio movement analysis with daily tracking."""
    if not args.years:
        print("Error: No years specified. Use --years 2020 2021 2022 ...")
        return
    
    years = [int(y) for y in args.years]
    
    print("\n" + "="*70)
    print("S&P 500 MOMENTUM STRATEGY - PORTFOLIO MOVEMENT ANALYSIS")
    print("="*70)
    print(f"Analyzing years: {years}")
    print("This will show daily portfolio movement with peak and max drawdown points.")
    
    years_data = []
    
    for year in years:
        print(f"\n{'='*50}")
        print(f"Processing {year}...")
        print(f"{'='*50}")
        
        start_date = date(year, 1, 1)
        
        # First run the screening to get selected stocks
        result = run_backtest(
            strategy_start_date=start_date,
            lookback_years=args.lookback,
            top_n=args.top_n,
            workers=args.workers,
            show_progress=True
        )
        
        if not result.get('selected_stocks'):
            print(f"No stocks selected for {year}, skipping...")
            continue
        
        selected_stocks = result['selected_stocks']
        portfolio_start = result['start_date']
        portfolio_end = result['end_date']
        
        # Calculate daily portfolio values
        print(f"\nCalculating daily portfolio values for {year}...")
        daily_values, metrics = calculate_daily_portfolio_values(
            tickers=selected_stocks,
            start_date=portfolio_start,
            end_date=portfolio_end,
            workers=args.workers
        )
        
        if daily_values.empty:
            print(f"Could not calculate daily values for {year}")
            continue
        
        # Print key metrics
        print(f"\n  📊 Portfolio Metrics for {year}:")
        print(f"     Peak Return:  +{metrics['peak_return']*100:.2f}% on {metrics['peak_date'].strftime('%Y-%m-%d')}")
        print(f"     Max Drawdown: {metrics['max_drawdown']*100:.2f}% on {metrics['max_drawdown_date'].strftime('%Y-%m-%d')}")
        print(f"     Final Return: {metrics['final_return']*100:+.2f}%")
        
        # Store for multi-year grid plot
        years_data.append((year, daily_values, metrics))
        
        # Generate individual year chart
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = OUTPUT_DIR / f"movement_{year}.png"
        
        plot_portfolio_movement(
            daily_values=daily_values,
            metrics=metrics,
            year=year,
            save_path=str(chart_path),
            show=not args.no_show
        )
    
    # Generate multi-year grid chart
    if len(years_data) > 1:
        years_str = f"{min(years)}-{max(years)}"
        grid_chart_path = OUTPUT_DIR / f"movement_grid_{years_str}.png"
        
        plot_multi_year_movements(
            years_data=years_data,
            save_path=str(grid_chart_path),
            show=not args.no_show
        )
        
        print(f"\n\n{'='*70}")
        print("MOVEMENT ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Individual charts saved to: output/movement_YEAR.png")
        print(f"Grid chart saved to: {grid_chart_path}")


def run_monte_carlo_mode(args):
    """Run Monte Carlo simulation for portfolio stress testing."""
    print("\n" + "="*70)
    print("MONTE CARLO STRESS TEST")
    print("="*70)
    
    # First, get the current top 10 stocks
    analysis_date = date.today()
    
    print(f"\nStep 1: Getting top {args.top_n} momentum stocks...")
    top_stocks, benchmark_returns, summary_df = run_momentum_screen(
        analysis_date=analysis_date,
        lookback_years=args.lookback,
        top_n=args.top_n,
        workers=args.workers,
        show_progress=True
    )
    
    if not top_stocks:
        print("No stocks passed the screening filter!")
        return
    
    selected_tickers = [t[0] for t in top_stocks]
    print(f"\nSelected stocks for simulation: {selected_tickers}")
    
    # Run Monte Carlo simulation
    print(f"\nStep 2: Running Monte Carlo simulation...")
    
    n_simulations = args.simulations if hasattr(args, 'simulations') else 10000
    
    results = run_monte_carlo(
        tickers=selected_tickers,
        n_simulations=n_simulations,
        n_days=252,
        initial_investment=10000,
        reference_year=analysis_date.year - 1,  # Use previous year's data
        workers=args.workers,
        show_progress=True
    )
    
    # Generate visualizations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Equity curves chart
    curves_path = OUTPUT_DIR / f"monte_carlo_curves_{analysis_date.year}.png"
    plot_equity_curves(
        results['equity_curves'],
        results['statistics'],
        n_curves=100,
        title=f"Monte Carlo Simulation: {analysis_date.year} Portfolio Projections",
        save_path=str(curves_path),
        show=not args.no_show
    )
    
    # Distribution histogram
    dist_path = OUTPUT_DIR / f"monte_carlo_distribution_{analysis_date.year}.png"
    plot_distribution(
        results['final_values'],
        results['statistics'],
        title=f"Monte Carlo Simulation: Distribution of {analysis_date.year} Outcomes",
        save_path=str(dist_path),
        show=not args.no_show
    )
    
    print(f"\n\n{'='*70}")
    print("MONTE CARLO SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Equity curves chart: {curves_path}")
    print(f"Distribution chart: {dist_path}")


def main():
    parser = argparse.ArgumentParser(
        description="S&P 500 Momentum Stock Screener & Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode screen
      Screen for top 10 momentum stocks for the coming year
  
  python main.py --mode backtest --start-date 2024-01-01
      Backtest strategy starting January 2024
  
  python main.py --mode multi-backtest --years 2020 2021 2022 2023 2024
      Run backtests for multiple years
  
  python main.py --mode movement-analysis --years 2020 2021 2022 2023 2024
      Analyze daily portfolio movement with peak/drawdown markers
  
  python main.py --mode monte-carlo
      Run Monte Carlo stress test on the current top 10 portfolio
  
  python main.py --mode sensitivity
      Run parameter sensitivity analysis to test strategy robustness
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['screen', 'backtest', 'multi-backtest', 'movement-analysis', 'monte-carlo', 'sensitivity'],
        required=True,
        help="Operation mode"
    )
    
    parser.add_argument(
        '--start-date', '-d',
        type=str,
        help="Strategy start date for backtest (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        '--years', '-y',
        nargs='+',
        type=int,
        help="Years for multi-year backtest or movement analysis (e.g., 2020 2021 2022)"
    )
    
    parser.add_argument(
        '--lookback', '-l',
        type=int,
        default=5,
        help="Number of years to look back for stock selection (default: 5)"
    )
    
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=10,
        help="Number of top stocks to select (default: 10)"
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=10,
        help="Number of parallel workers for data fetching (default: 10)"
    )
    
    parser.add_argument(
        '--simulations', '-s',
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10,000)"
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help="Don't display charts (only save to files)"
    )
    
    parser.add_argument(
        '--download-data',
        action='store_true',
        help="Force re-download of historical constituent data"
    )
    
    args = parser.parse_args()
    
    # Download historical data if requested
    if args.download_data:
        download_historical_constituents(force=True)
    
    # Run appropriate mode
    if args.mode == 'screen':
        run_screen_mode(args)
    elif args.mode == 'backtest':
        if not args.start_date:
            print("Error: --start-date is required for backtest mode")
            return
        run_backtest_mode(args)
    elif args.mode == 'multi-backtest':
        run_multi_backtest_mode(args)
    elif args.mode == 'movement-analysis':
        run_movement_analysis_mode(args)
    elif args.mode == 'monte-carlo':
        run_monte_carlo_mode(args)
    elif args.mode == 'sensitivity':
        run_sensitivity_mode(args)


def run_sensitivity_mode(args):
    """Run parameter sensitivity analysis."""
    analysis_year = date.today().year
    
    print("\n" + "="*70)
    print("STRATEGY SENSITIVITY ANALYSIS")
    print("="*70)
    print("Testing strategy robustness across parameter variations...")
    
    # Run sensitivity analysis
    results = run_sensitivity_analysis(
        analysis_year=analysis_year,
        lookback_variants=[3, 4, 5, 6],
        top_n=args.top_n,
        workers=args.workers,
        show_progress=True
    )
    
    # Generate visualizations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Alpha heatmap
    alpha_path = OUTPUT_DIR / f"sensitivity_alpha_{analysis_year}.png"
    plot_sensitivity_heatmap(
        results['results_df'],
        metric='avg_alpha',
        title=f"Sensitivity Analysis: Average Alpha ({analysis_year})",
        save_path=str(alpha_path),
        show=not args.no_show
    )
    
    # Win rate heatmap
    winrate_path = OUTPUT_DIR / f"sensitivity_winrate_{analysis_year}.png"
    plot_sensitivity_heatmap(
        results['results_df'],
        metric='win_rate',
        title=f"Sensitivity Analysis: Win Rate ({analysis_year})",
        save_path=str(winrate_path),
        show=not args.no_show
    )
    
    # Sharpe heatmap
    sharpe_path = OUTPUT_DIR / f"sensitivity_sharpe_{analysis_year}.png"
    plot_sensitivity_heatmap(
        results['results_df'],
        metric='avg_sharpe',
        title=f"Sensitivity Analysis: Average Sharpe Ratio ({analysis_year})",
        save_path=str(sharpe_path),
        show=not args.no_show
    )
    
    # Stock stability chart
    stability_path = OUTPUT_DIR / f"sensitivity_stocks_{analysis_year}.png"
    plot_stock_stability(
        results['stock_selection_matrix'],
        title=f"Stock Selection Stability Across Parameter Variants ({analysis_year})",
        save_path=str(stability_path),
        show=not args.no_show
    )
    
    # Save results to CSV
    csv_path = OUTPUT_DIR / f"sensitivity_results_{analysis_year}.csv"
    results['results_df'].to_csv(csv_path, index=False)
    
    print(f"\n\n{'='*70}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nStability Score: {results['stability_score']}")
    print(f"\nHigh-Conviction Stocks: {results['high_conviction_stocks']}")
    print(f"\nCharts saved to: output/sensitivity_*.png")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
