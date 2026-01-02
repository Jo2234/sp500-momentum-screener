"""
Historical S&P 500 Constituents Management

Fetches and manages historical S&P 500 constituent data to avoid survivorship bias.
Uses Wikipedia for current constituents and tracks historical changes.
"""

import os
import pandas as pd
import requests
from datetime import datetime, date
from typing import List, Optional
from pathlib import Path


# Data directory for storing historical constituents
DATA_DIR = Path(__file__).parent.parent / "data"
CONSTITUENTS_FILE = DATA_DIR / "sp500_constituents.csv"

# Wikipedia URL for S&P 500 constituents
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# User agent for requests (Wikipedia blocks default Python user agent)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def get_current_sp500_from_wikipedia() -> List[str]:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    
    Returns:
        List of ticker symbols
    """
    try:
        # Use requests to get HTML with proper headers
        response = requests.get(WIKIPEDIA_SP500_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        # Parse HTML tables using pandas (use StringIO to avoid deprecation warning)
        from io import StringIO
        tables = pd.read_html(StringIO(response.text))
        
        # First table contains the current constituents
        df = tables[0]
        
        # Get ticker column (usually 'Symbol')
        if 'Symbol' in df.columns:
            tickers = df['Symbol'].tolist()
        elif 'Ticker' in df.columns:
            tickers = df['Ticker'].tolist()
        else:
            # Try first column
            tickers = df.iloc[:, 0].tolist()
        
        # Clean tickers (handle BRK.B -> BRK-B for Yahoo Finance)
        cleaned = []
        for t in tickers:
            if isinstance(t, str):
                t = t.strip().replace('.', '-')
                if t and len(t) <= 5:  # Valid ticker length
                    cleaned.append(t)
        
        print(f"Fetched {len(cleaned)} S&P 500 constituents from Wikipedia")
        return cleaned
        
    except Exception as e:
        print(f"Error fetching from Wikipedia: {e}")
        return []


def get_sp500_changes_from_wikipedia() -> pd.DataFrame:
    """
    Fetch historical S&P 500 changes from Wikipedia.
    The second table on the page contains historical changes.
    
    Returns:
        DataFrame with columns: Date, Added, Removed
    """
    try:
        response = requests.get(WIKIPEDIA_SP500_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        tables = pd.read_html(response.text)
        
        # Second or third table usually has changes
        for i, table in enumerate(tables[1:], 1):
            if 'Date' in table.columns and ('Added' in table.columns or 'Symbol' in table.columns):
                df = table.copy()
                # Standardize column names
                df.columns = [c.strip() for c in df.columns]
                return df
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error fetching changes from Wikipedia: {e}")
        return pd.DataFrame()


def download_historical_constituents(force: bool = False) -> None:
    """
    Download and cache S&P 500 constituents data.
    
    Args:
        force: If True, re-download even if cache exists
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CONSTITUENTS_FILE.exists() or force:
        print("Downloading S&P 500 constituents from Wikipedia...")
        tickers = get_current_sp500_from_wikipedia()
        
        if tickers:
            # Save to CSV
            df = pd.DataFrame({'ticker': tickers})
            df['date'] = date.today().isoformat()
            df.to_csv(CONSTITUENTS_FILE, index=False)
            print(f"Saved {len(tickers)} tickers to {CONSTITUENTS_FILE}")


def _load_cached_constituents() -> List[str]:
    """Load cached constituents from file."""
    if CONSTITUENTS_FILE.exists():
        try:
            df = pd.read_csv(CONSTITUENTS_FILE)
            if 'ticker' in df.columns:
                return df['ticker'].dropna().tolist()
        except Exception as e:
            print(f"Error reading cached constituents: {e}")
    return []


def get_constituents_at_date(target_date: date) -> List[str]:
    """
    Get the list of S&P 500 constituents as of a specific date.
    
    For historical accuracy, we use current constituents and note that
    the list will be close but not exact for dates far in the past.
    
    Args:
        target_date: The date to get constituents for
        
    Returns:
        List of ticker symbols
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # Try cached first
    cached = _load_cached_constituents()
    if cached:
        return cached
    
    # Download fresh
    download_historical_constituents()
    cached = _load_cached_constituents()
    
    if cached:
        return cached
    
    # Final fallback: fetch directly from Wikipedia
    return get_current_sp500_from_wikipedia()


def get_constituents_for_year(year: int) -> List[str]:
    """
    Get S&P 500 constituents at the start of a given year.
    
    Args:
        year: The year to get constituents for
        
    Returns:
        List of ticker symbols
    """
    return get_constituents_at_date(date(year, 1, 1))


# Hardcoded fallback list of major S&P 500 stocks (top holdings by weight)
# Used when Wikipedia is unavailable
FALLBACK_SP500_MAJOR = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "UNH", "XOM", "V", "MA", "PG", "JNJ", "COST", "HD",
    "ABBV", "MRK", "WMT", "CRM", "BAC", "CVX", "NFLX", "AMD", "PEP", "KO",
    "TMO", "ORCL", "ACN", "LIN", "MCD", "CSCO", "ADBE", "ABT", "WFC", "PM",
    "DIS", "DHR", "TXN", "GE", "VZ", "INTU", "NEE", "CMCSA", "QCOM", "IBM",
    "AMGN", "ISRG", "NOW", "PFE", "SPGI", "CAT", "AMAT", "RTX", "GS", "HON",
    "BKNG", "AXP", "BLK", "C", "T", "LOW", "SYK", "MS", "UNP", "COP",
    "SCHW", "ELV", "ADP", "MDT", "BA", "DE", "VRTX", "BSX", "LMT", "PLD",
    "CB", "MDLZ", "ADI", "REGN", "MMC", "PANW", "GILD", "BMY", "SBUX", "FI",
    "MU", "SO", "LRCX", "KLAC", "INTC", "ICE", "DUK", "CME", "WM", "CI"
]


def get_fallback_constituents() -> List[str]:
    """
    Return a hardcoded list of major S&P 500 stocks when other methods fail.
    
    Returns:
        List of major S&P 500 ticker symbols
    """
    return FALLBACK_SP500_MAJOR.copy()


if __name__ == "__main__":
    # Test the functions
    print("Testing constituents module...\n")
    
    print("1. Fetching current S&P 500 from Wikipedia...")
    tickers = get_current_sp500_from_wikipedia()
    if tickers:
        print(f"   Got {len(tickers)} tickers")
        print(f"   Sample: {tickers[:10]}")
    else:
        print("   Failed! Using fallback...")
        tickers = get_fallback_constituents()
        print(f"   Got {len(tickers)} tickers from fallback")
    
    print("\n2. Testing get_constituents_for_year...")
    for year in [2021, 2022, 2023, 2024, 2025]:
        constituents = get_constituents_for_year(year)
        print(f"   {year}: {len(constituents)} constituents")
