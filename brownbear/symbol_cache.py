"""
Symbol cache management.
"""

import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import _yfinance_config  # noqa: F401
import yfinance as yf

import brownbear as bb


def _get_cutoff_date(reference_date=None):
    """
    Return the default end date for symbol-cache downloads.

    - If today is before the 25th: return last day of two months ago
    - If today is on or after the 25th: return last day of previous month

    Parameters
    ----------
    reference_date : date, optional
        Date used for the cutoff rule (default is today).

    Returns
    -------
    date
        Last day of the selected month.
    """
    if reference_date is None:
        reference_date = datetime.date.today()

    first_of_this_month = reference_date.replace(day=1)
    last_of_prev_month = first_of_this_month - datetime.timedelta(days=1)

    if reference_date.day < 25:
        # Go to the first day of the previous month
        first_of_prev_month = last_of_prev_month.replace(day=1)
        # Then subtract 1 day to get last of two months ago
        return first_of_prev_month - datetime.timedelta(days=1)
    else:
        # Just return last of previous month
        return last_of_prev_month



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
        Ending date, timestamp. Same format as starting date (default is None).
        Can be:
        - None -> compute default cutoff date
        - 'latest' -> yesterday's date
        - any -> use as-is
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
        end = _get_cutoff_date()
    elif isinstance(end, str) and end.lower() == 'latest':
        end = datetime.datetime.now() - datetime.timedelta(1)
    else:
        pass

    symbol_cache_path = Path(bb.SYMBOL_CACHE)
    symbol_cache_path.mkdir(parents=True, exist_ok=True)

    request_count = 0  # Counter to track the number of requests made

    for i, symbol in enumerate(symbols):
        print(symbol, end=' ')
        filepath = symbol_cache_path / f'{symbol}.csv'

        if refresh or not filepath.is_file():
            try:
                df = yf.download(
                    symbol, start, end, progress=False,
                    auto_adjust=False, multi_level_index=False,
                    threads=False, timeout=30,
                )
                if df.empty:
                    print(f'No Data for {symbol}')
                    continue
            except Exception as e:
                print(f'\n{e}')
            else:
                df.reset_index(inplace=True)
                df.set_index('Date', inplace=True)
                df.to_csv(filepath, encoding='utf-8')

        request_count += 1

        # Throttle: wait after every `throttle_limit` requests
        if request_count >= throttle_limit:
            print(f'\nThrottle limit reached. Waiting for {wait_time} seconds...')
            time.sleep(wait_time)
            request_count = 0  # Reset the counter after waiting

    print()


def compile_timeseries(symbols, output_path='symbols-timeseries.csv'):
    """
    Compile one or more symbols' timeseries into a single dataframe.

    The timeseries are read from the symbol cache only, so the
    timeseries must exists for every symbol in `symbols`.  Otherwise
    an exception will be thrown.

    The compiled timeseries has a column for each symbol.  Each row
    contains the daily closing prices for the symbols.  This timeseries
    is written to ``output_path`` (default ``symbols-timeseries.csv`` in
    the current directory).

    Parameters
    ----------
    symbols : list of str
        The list of symbols for securities.
    output_path : str or Path, optional
        Output CSV path for the compiled timeseries.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
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
    compiled_df.to_csv(output_path, encoding='utf-8')


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
        """
        Calculate the number of years between two dates.

        Parameters
        ----------
        start : datetime
            Start datetime.
        end : datetime
            End datetime.

        Returns
        -------
        float
            Elapsed time in years.
        """
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
        filepath = symbol_cache_path / f'{symbol}.csv'
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


def _fundamentals_dir():
    """
    Return the directory used for fundamentals cache files.

    Returns
    -------
    Path
        ``tools/symbol-cache`` under the project root.
    """
    return bb.ROOT / 'tools' / 'symbol-cache'


def _fundamentals_cache_path():
    """
    Return the path to the fundamentals JSON cache file.

    Returns
    -------
    Path
        ``fundamentals_cache.json`` under :func:`_fundamentals_dir`.
    """
    return _fundamentals_dir() / 'fundamentals_cache.json'


def reset_fundamentals_cache():
    """
    Delete the fundamentals JSON cache used by get_symbol_fundamentals.

    Returns
    -------
    bool
        True if a cache file was deleted, False if none existed.
    """
    cache_file = _fundamentals_cache_path()
    if cache_file.exists():
        cache_file.unlink()
        return True
    return False


def get_symbol_fundamentals(symbols=None, throttle_limit=100, wait_time=30, reset_cache=False):
    """
    Get fundamental data for list of symbols with caching and throttling.

    Parameters
    ----------
    symbols : list of str, optional
        The list of symbols for securities (default is None, which implies all symbols
        in symbol cache).
    throttle_limit : int, optional
        The number of symbols to fetch before waiting (default is 100).
    wait_time : int, optional
        The number of seconds to wait after reaching the throttle limit (default is 30).
    reset_cache : bool, optional
        If True, delete the existing cache and start fresh (default is False).

    Returns
    -------
    DataFrame
        DataFrame containing the fundamental data for the provided symbols.
    """
    symbol_cache_path = Path(bb.SYMBOL_CACHE)
    cache_file = _fundamentals_cache_path()

    if reset_cache:
        reset_fundamentals_cache()

    # Load existing cache if available
    cache = {}
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)

    # Get the list of symbols
    if symbols:
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = [filename.name for filename in symbol_cache_path.iterdir()
                     if filename.suffix == '.csv' and not filename.stem.startswith('__')]
        symbols = [Path(filename).stem for filename in filenames]

    # Convert symbols to uppercase
    symbols = [symbol.upper() for symbol in symbols]

    # Remove already cached symbols from the list
    symbols_to_fetch = [symbol for symbol in symbols if symbol not in cache]
    print(f'Fetching fundamental data for {len(symbols_to_fetch)} symbols...')

    request_count = 0

    for symbol in symbols_to_fetch:
        print(symbol, end=' ')

        try:
            # Fetch fundamental data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract relevant data
            previousClose = info.get('previousClose', np.nan)
            trailingPE = info.get('trailingPE', 0)
            dividendYield = info.get('dividendYield', 0) * 100  # Convert to percentage
            marketCap = info.get('marketCap', 0) / 1_000_000  # Convert to million
            companyName = info.get('shortName', None)

            # Store result in cache
            cache[symbol] = {
                'companyName': companyName,
                'previousClose': previousClose,
                'trailingPE': trailingPE,
                'dividendYield': dividendYield,
                'marketCap': marketCap
            }

            # Save updated cache to file after each symbol
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=4)

        except Exception as e:
            print(f'\nError fetching data for {symbol}: {e}')

        request_count += 1

        # Throttle after hitting the limit
        if request_count >= throttle_limit:
            print(f'\nThrottle limit reached. Waiting for {wait_time} seconds...')
            time.sleep(wait_time)
            request_count = 0  # Reset counter

    # Convert cache dict to DataFrame
    df = pd.DataFrame.from_dict(cache, orient='index')
    df.index.name = 'symbol'

    return df
