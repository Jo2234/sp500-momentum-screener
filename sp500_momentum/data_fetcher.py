"""
Multithreaded Stock Data Fetcher with Persistent Caching

Efficiently fetches stock price data using yfinance with:
- Parquet-based local caching
- Concurrent data fetching via ThreadPoolExecutor
- Automatic retry with exponential backoff
"""

import os
import time
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm


# Cache directory for storing price data
CACHE_DIR = Path(__file__).parent.parent / "cache"

# Default settings
DEFAULT_WORKERS = 10
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(ticker: str) -> Path:
    """Get the cache file path for a ticker."""
    # Sanitize ticker for filesystem
    safe_ticker = ticker.replace('/', '_').replace('\\', '_')
    return CACHE_DIR / f"{safe_ticker}.parquet"


def _load_cached_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached data for a ticker if it exists."""
    cache_path = _get_cache_path(ticker)
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Warning: Could not read cache for {ticker}: {e}")
    return None


def _save_to_cache(ticker: str, df: pd.DataFrame) -> None:
    """Save data to cache."""
    _ensure_cache_dir()
    cache_path = _get_cache_path(ticker)
    try:
        df.to_parquet(cache_path)
    except Exception as e:
        print(f"Warning: Could not save cache for {ticker}: {e}")


def _merge_cached_data(cached_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cached data with newly fetched data."""
    if cached_df is None or cached_df.empty:
        return new_df
    if new_df is None or new_df.empty:
        return cached_df
    
    # Combine and remove duplicates, keeping newer data
    combined = pd.concat([cached_df, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    return combined


def fetch_stock_data(
    ticker: str,
    start_date: date,
    end_date: date,
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch stock data for a single ticker with caching.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        use_cache: Whether to use cached data
        
    Returns:
        DataFrame with OHLCV data, or None if fetch failed
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    
    cached_df = None
    fetch_start = start_date
    fetch_end = end_date
    
    if use_cache:
        cached_df = _load_cached_data(ticker)
        
        if cached_df is not None and not cached_df.empty:
            cache_start = cached_df.index.min().date()
            cache_end = cached_df.index.max().date()
            
            # Check if cache covers the entire range
            if cache_start <= start_date and cache_end >= end_date:
                # Filter to requested range
                mask = (cached_df.index.date >= start_date) & (cached_df.index.date <= end_date)
                return cached_df[mask]
            
            # Determine what we need to fetch
            if cache_start <= start_date:
                # Only need to fetch newer data
                fetch_start = cache_end + timedelta(days=1)
            elif cache_end >= end_date:
                # Only need to fetch older data
                fetch_end = cache_start - timedelta(days=1)
    
    # Fetch from Yahoo Finance with retry logic
    new_df = None
    for attempt in range(MAX_RETRIES):
        try:
            stock = yf.Ticker(ticker)
            new_df = stock.history(
                start=fetch_start.isoformat(),
                end=(fetch_end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                auto_adjust=True  # Adjust for splits/dividends
            )
            
            if new_df is not None and not new_df.empty:
                break
                
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
            else:
                return cached_df  # Return cached data if available
    
    # Merge with cached data
    if new_df is not None and not new_df.empty:
        result = _merge_cached_data(cached_df, new_df)
        
        # Update cache with all data
        if use_cache:
            _save_to_cache(ticker, result)
        
        # Filter to requested range
        mask = (result.index.date >= start_date) & (result.index.date <= end_date)
        return result[mask]
    
    return cached_df


def fetch_batch_data(
    tickers: List[str],
    start_date: date,
    end_date: date,
    workers: int = DEFAULT_WORKERS,
    use_cache: bool = True,
    show_progress: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple tickers in parallel.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        workers: Number of parallel workers
        use_cache: Whether to use cached data
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    results = {}
    failed = []
    
    def fetch_single(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
        df = fetch_stock_data(ticker, start_date, end_date, use_cache)
        return ticker, df
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_single, t): t for t in tickers}
        
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(tickers), desc="Fetching stock data")
        
        for future in iterator:
            ticker = futures[future]
            try:
                ticker, df = future.result()
                if df is not None and not df.empty:
                    results[ticker] = df
                else:
                    failed.append(ticker)
            except Exception as e:
                failed.append(ticker)
    
    if failed and show_progress:
        print(f"Failed to fetch data for {len(failed)} tickers: {failed[:10]}...")
    
    return results


def get_sp500_index_data(
    start_date: date,
    end_date: date,
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch S&P 500 index data (^GSPC).
    
    Args:
        start_date: Start date
        end_date: End date
        use_cache: Whether to use cache
        
    Returns:
        DataFrame with S&P 500 index data
    """
    return fetch_stock_data("^GSPC", start_date, end_date, use_cache)


def get_adjusted_close_prices(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Extract adjusted close prices from fetched data.
    
    Args:
        data: Dictionary of ticker -> DataFrame
        
    Returns:
        DataFrame with tickers as columns and dates as index
    """
    closes = {}
    for ticker, df in data.items():
        if df is not None and 'Close' in df.columns:
            closes[ticker] = df['Close']
    
    return pd.DataFrame(closes)


def clear_cache(tickers: Optional[List[str]] = None) -> int:
    """
    Clear cached data.
    
    Args:
        tickers: Specific tickers to clear, or None to clear all
        
    Returns:
        Number of files deleted
    """
    deleted = 0
    
    if tickers is None:
        # Clear all cache
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.parquet"):
                f.unlink()
                deleted += 1
    else:
        for ticker in tickers:
            cache_path = _get_cache_path(ticker)
            if cache_path.exists():
                cache_path.unlink()
                deleted += 1
    
    return deleted


if __name__ == "__main__":
    # Test the fetcher
    print("Testing data fetcher...")
    
    # Test single fetch
    print("\n1. Fetching AAPL data...")
    df = fetch_stock_data("AAPL", date(2024, 1, 1), date(2024, 12, 31))
    if df is not None:
        print(f"   Got {len(df)} rows")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Test S&P 500 index
    print("\n2. Fetching S&P 500 index...")
    sp_df = get_sp500_index_data(date(2024, 1, 1), date(2024, 12, 31))
    if sp_df is not None:
        print(f"   Got {len(sp_df)} rows")
    
    # Test batch fetch
    print("\n3. Batch fetching 5 stocks...")
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    batch_data = fetch_batch_data(test_tickers, date(2024, 1, 1), date(2024, 12, 31))
    print(f"   Successfully fetched {len(batch_data)} tickers")
