"""
Symbol cache management.
"""

import datetime
import os

import numpy as np
import pandas as pd
from pandas_datareader._utils import RemoteDataError
import pandas_datareader.data as pdr
import yfinance as yf
from yahooquery import Ticker

import brownbear as bb


# Override pandas_datareader with yfinance
yf.pdr_override()


def fetch_timeseries(symbols, start=None, end=None, refresh=False):
    """
    Fetch timeseries for one or more symbols from yahoo finance.

    Write the timeseries to the symbol cache as `symbol`.csv

    Parameters
    ----------
    symbols : list of str
        The list of symbols for securities.
    start : (string, int, date, datetime, Timestamp), optional
        Starting date. Parses many different kind of date
        representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
        (Default is None, which implies 01-01-2015).
    end : (string, int, date, datetime, Timestamp), optional
        Ending date, timestamp. Same format as starting date
        (Default is NONE, which implies yesterday).
    refresh : bool, optional
        True to retrieve timeseries from internet instead of using
        symbol cache (default is False).

    Returns
    -------
    None
    """
    if start is None:
        start = datetime.datetime(2015, 1, 1)
    if end is None:
        end = datetime.datetime.now() - datetime.timedelta(1)

    if not os.path.exists(bb.SYMBOL_CACHE):
        os.makedirs(bb.SYMBOL_CACHE)

    for symbol in symbols:
        print(symbol, end=' ')
        filepath = bb.SYMBOL_CACHE / (symbol + '.csv')
        if refresh or not os.path.isfile(filepath):
            try:
                #df = pdr.DataReader(symbol, 'yahoo', start, end)
                df = pdr.get_data_yahoo(symbol, start, end, progress=False)
            except RemoteDataError as e:
                print(f'\n{e}')
            except Exception as e:
                print(f'\n{e}')
            else:
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.to_csv(filepath)
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
    compiled_df.to_csv('symbols-timeseries.csv')


def remove_cache_symbols(symbols=None):
    """
    Remove cached timeseries for list of symbols.

    Filter out any symbols prefixed with '__'.

    Parameters
    ----------
    symbols : list of str, optional
        The list of symbols for securities (default is None, which
        imples all symbols in symbol cache).

    Returns
    -------
    None
    """
    if symbols:
        # In case user forgot to put single symbol in a list.
        if not isinstance(symbols, list):
            symbols = [symbols]
        filenames = [symbol.upper() + '.csv' for symbol in symbols]
    else:
        filenames = [f for f in os.listdir(bb.SYMBOL_CACHE) if f.endswith('.csv')]

    # Filter out any filename prefixed with '__'.
    filenames = [f for f in filenames if not f.startswith('__')]

    print('removing symbols:')
    for i, f in enumerate(filenames):
        symbol = os.path.splitext(f)[0]
        print(symbol + ' ', end='')
        if i % 10 == 0 and i != 0: print()

        filepath = bb.SYMBOL_CACHE / f
        if os.path.exists(filepath):
            os.remove(filepath)
        else:
            print(f'\n({f} not found)')
    print()


def update_cache_symbols(symbols=None):
    """
    Update cached timeseries for list of symbols.

    Filter out any symbols prefixed with '__'.

    Parameters
    ----------
    symbols : list of str, optional
        The list of symbols for securities (default is None, which
        imples all symbols in symbol cache).

    Returns
    -------
    None
    """
    if symbols:
        # In case user forgot to put single symbol in list.
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = ([f for f in os.listdir(bb.SYMBOL_CACHE)
                     if f.endswith('.csv') and not f.startswith('__')])
        symbols = [os.path.splitext(filename)[0] for filename in filenames]

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
        imples all symbols in symbol cache).

    Returns
    -------
    None
    """
    def _difference_in_years(start, end):
        """ Calculate the number of years between two dates. """
        diff  = end - start
        diff_in_years = (diff.days + diff.seconds/86400)/365.2425
        return diff_in_years

    if symbols:
        # In case user forgot to put single symbol in list.
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = ([f for f in os.listdir(bb.SYMBOL_CACHE)
                      if f.endswith('.csv') and not f.startswith('__')])
        symbols = [os.path.splitext(filename)[0] for filename in filenames]

    # Make symbol names uppercase.
    symbols = [symbol.upper() for symbol in symbols]

    l = []
    for i, symbol in enumerate(symbols):
        filepath = bb.SYMBOL_CACHE / (symbol + '.csv')
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


def get_symbol_fundamentals(symbols=None):
    """
    Get fundamental data for list of symbols.

     Filter out any symbols prefixed with '__'.

    Parameters
    ----------
    symbols : list of str, optional
        The list of symbols for securities (default is None, which
        imples all symbols in symbol cache).

    Returns
    -------
    None
    """
    if symbols:
        # In case user forgot to put single symbol in list.
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = ([f for f in os.listdir(bb.SYMBOL_CACHE)
                      if f.endswith('.csv') and not f.startswith('__')])
        symbols = [os.path.splitext(filename)[0] for filename in filenames]

    # Make symbol names uppercase.
    symbols = [symbol.upper() for symbol in symbols]

    l = []
    for i, symbol in enumerate(symbols):
        print(symbol, end=' ')
        ticker = Ticker(symbol)
        d = ticker.summary_detail[symbol]

        previousClose = trailingPE = dividendYield = marketCap = np.nan
        if isinstance(d, dict):
            try:
                previousClose = d.get('previousClose', np.nan)
                trailingPE = d.get('trailingPE', 0)
                dividendYield = d.get('dividendYield', 0) * 100
                marketCap = d.get('marketCap', 0) / 1000000
            except Exception as e:
                print(e)
        else:
            print(f'\n({d})')

        t = (symbol, previousClose, trailingPE, dividendYield, marketCap)
        l.append(t)

    columns = ['symbol', 'previousClose', 'trailingPE', 'dividendYield', 'marketCap']
    df = pd.DataFrame(l, columns=columns)
    df.set_index('symbol', inplace=True)
    return df
