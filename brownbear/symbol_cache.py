"""
Symbol cache management.
"""

import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf

import brownbear as bb


def fetch_timeseries(symbols, start=None, end=None, refresh=False, throttle_limit=100, wait_time=30):
    """
    Fetch timeseries for one or more symbols from Yahoo Finance.

    Write the timeseries to the symbol cache as `symbol.csv`.

    Parameters
    ----------
    symbols : list of str
        The list of symbols for securities.
    start : (string, int, date, datetime, Timestamp), optional
        Starting date. Parses many different kinds of date representations
        (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980') (default is None, which implies 01-01-2019).
    end : (string, int, date, datetime, Timestamp), optional
        Ending date, timestamp. Same format as starting date (default is None, which implies yesterday).
    refresh : bool, optional
        True to retrieve timeseries from the internet instead of using symbol cache (default is False).
    throttle_limit : int, optional
        The number of symbols to fetch before waiting (default is 100).
    wait_time : int, optional
        The number of seconds to wait after reaching the throttle limit (default is 30).

    Returns
    -------
    None
    """
    if start is None:
        start = datetime.datetime(2019, 1, 1)
    if end is None:
        end = datetime.datetime.now() - datetime.timedelta(1)

    symbol_cache_path = Path(bb.SYMBOL_CACHE)
    symbol_cache_path.mkdir(parents=True, exist_ok=True)

    request_count = 0  # Counter to track the number of requests made

    for i, symbol in enumerate(symbols):
        print(symbol, end=' ')
        filepath = symbol_cache_path / f"{symbol}.csv"

        if refresh or not filepath.is_file():
            try:
                df = yf.download(symbol, start=datetime.datetime(from_year, 1, 1),
            		             progress=False, auto_adjust=False, multi_level_index=False)
                if df.empty:
                    print(f'No Data for {symbol}')
                    continue
            except Exception as e:
                print(f'\n{e}')
            else:
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.to_csv(filepath, encoding='utf-8')

        request_count += 1

        # Throttle: wait after every `throttle_limit` requests
        if request_count >= throttle_limit:
            print(f"\nThrottle limit reached. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            request_count = 0  # Reset the counter after waiting

    print()


def compile_timeseries(symbols):
    """
    Compile one or more symbols' timeseries into a single dataframe.

    The timeseries are read from the symbol cache only, so the
    timeseries must exists for every symbol in `symbols`.  Otherwise
    an exception will be thrown.

    The compiled timeseries has a column for each symbol.  Each row
    contains the daily closing prices for the symbols.  This timeseries
    is written to 'symbols-timeseries.csv' in the current directory.

    Parameters
    ----------
    symbols : list of str
        The list of symbols for securities.

    Returns
    -------
    None
    """
    compiled_df = pd.DataFrame()

    for symbol in symbols:
        filepath = bb.SYMBOL_CACHE / (symbol + '.csv')
        df = pd.read_csv(filepath)
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': symbol}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

        if compiled_df.empty:
            compiled_df = df
        else:
            compiled_df = compiled_df.join(df, how='outer')
    compiled_df.to_csv('symbols-timeseries.csv', encoding='utf-8')


def remove_cache_symbols(symbols=None):
    """
    Remove cached timeseries for list of symbols.

    Filter out any symbols prefixed with '__'.

    Parameters
    ----------
    symbols : list of str, optional
        The list of symbols for securities (default is None, which
        implies all symbols in symbol cache).

    Returns
    -------
    None
    """
    symbol_cache_path = Path(bb.SYMBOL_CACHE)

    if symbols:
        # In case user forgot to put a single symbol in a list.
        if not isinstance(symbols, list):
            symbols = [symbols]
        filenames = [symbol.upper() + '.csv' for symbol in symbols]
    else:
        filenames = [filename.name for filename in symbol_cache_path.iterdir() 
                     if filename.suffix == '.csv']

    # Filter out any filename prefixed with '__'.
    filenames = [filename for filename in filenames if not filename.startswith('__')]

    print('removing symbols:')
    for i, filename in enumerate(filenames):
        symbol = Path(filename).stem
        print(symbol + ' ', end='')
        if i % 10 == 0 and i != 0:
            print()

        filepath = symbol_cache_path / filename
        if filepath.exists():
            filepath.unlink()
        else:
            print(f'\n({filename} not found)')
    print()


def update_cache_symbols(symbols=None):
    """
    Update cached timeseries for list of symbols.

    Filter out any symbols prefixed with '__'.

    Parameters
    ----------
    symbols : list of str, optional
        The list of symbols for securities (default is None, which
        implies all symbols in symbol cache).

    Returns
    -------
    None
    """
    symbol_cache_path = Path(bb.SYMBOL_CACHE)

    if symbols:
        # In case user forgot to put single symbol in list.
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = [filename.name for filename in symbol_cache_path.iterdir()
                     if filename.suffix == '.csv' and not filename.stem.startswith('__')]
        symbols = [Path(filename).stem for filename in filenames]

    # Make symbol names uppercase.
    symbols = [symbol.upper() for symbol in symbols]

    # Update timeseries for symbols.
    bb.fetch_timeseries(symbols, refresh=True)


def get_symbol_metadata(symbols=None):
    """
    Get symbol metadata for list of symbols

    Filter out any symbols prefixed with '__'.

    Parameters
    ----------
    symbols : list of str, optional
        The list of symbols for securities (default is None, which
        implies all symbols in symbol cache).

    Returns
    -------
    pd.DataFrame
    """
    def _difference_in_years(start, end):
        """ Calculate the number of years between two dates. """
        diff = end - start
        diff_in_years = (diff.days + diff.seconds / 86400) / 365.2425
        return diff_in_years

    symbol_cache_path = Path(bb.SYMBOL_CACHE)

    if symbols:
        # In case user forgot to put a single symbol in a list.
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = [filename.name for filename in symbol_cache_path.iterdir()
                     if filename.suffix == '.csv' and not filename.stem.startswith('__')]
        symbols = [Path(filename).stem for filename in filenames]

    # Make symbol names uppercase.
    symbols = [symbol.upper() for symbol in symbols]

    l = []
    for i, symbol in enumerate(symbols):
        filepath = symbol_cache_path / f"{symbol}.csv"
        ts = pd.read_csv(filepath)
        ts.set_index('Date', inplace=True)
        start = datetime.datetime.strptime(ts.index[0], '%Y-%m-%d')
        end = datetime.datetime.strptime(ts.index[-1], '%Y-%m-%d')
        num_years = _difference_in_years(start, end)
        start = start.strftime('%Y-%m-%d')
        end = end.strftime('%Y-%m-%d')
        t = (symbol, start, end, num_years)
        l.append(t)

    columns = ['symbol', 'start_date', 'end_date', 'num_years']
    df = pd.DataFrame(l, columns=columns)
    return df


def get_symbol_fundamentals(symbols=None, throttle_limit=100, wait_time=30):
    """
    Get fundamental data for list of symbols with throttling to avoid hitting rate limits.

    Filter out any symbols prefixed with '__'.

    Parameters
    ----------
    symbols : list of str, optional
        The list of symbols for securities (default is None, which implies all symbols
        in symbol cache).
    throttle_limit : int, optional
        The number of symbols to fetch before waiting (default is 100).
    wait_time : int, optional
        The number of seconds to wait after reaching the throttle limit (default is 30).

    Returns
    -------
    DataFrame
        DataFrame containing the fundamental data for the provided symbols.
    """
    symbol_cache_path = Path(bb.SYMBOL_CACHE)

    if symbols:
        # In case user forgot to put a single symbol in a list.
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = [filename.name for filename in symbol_cache_path.iterdir()
                     if filename.suffix == '.csv' and not filename.stem.startswith('__')]
        symbols = [Path(filename).stem for filename in filenames]

    # Make symbol names uppercase.
    symbols = [symbol.upper() for symbol in symbols]

    l = []
    request_count = 0  # Counter to track the number of requests made

    for i, symbol in enumerate(symbols):
        print(symbol, end=' ')

        # Use yfinance to fetch data for the symbol
        ticker = yf.Ticker(symbol)

        # Fetch the fundamental data
        previousClose = trailingPE = dividendYield = marketCap = np.nan
        try:
            info = ticker.info
            previousClose = info.get('previousClose', np.nan)
            trailingPE = info.get('trailingPE', 0)
            dividendYield = info.get('dividendYield', 0) * 100  # Convert to percentage
            marketCap = info.get('marketCap', 0) / 1_000_000  # Convert to million
            companyName = info.get('shortName', None)  # Short name of the company

        except Exception as e:
            print(f"\nError fetching data for {symbol}: {e}")

        # Add the data to the list
        t = (symbol, companyName, previousClose, trailingPE, dividendYield, marketCap)
        l.append(t)

        request_count += 1

        # Throttle: wait after every 100 requests
        if request_count >= throttle_limit:
            print(f"\nThrottle limit reached. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)  # Wait for the specified time
            request_count = 0  # Reset the counter after waiting

    # Define the columns and create the DataFrame
    columns = ['symbol', 'companyName', 'previousClose', 'trailingPE', 'dividendYield', 'marketCap']
    df = pd.DataFrame(l, columns=columns)
    df.set_index('symbol', inplace=True)

    return df
