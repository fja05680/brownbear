"""
symbol_cache
---------
symbol cache management
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
from pandas_datareader._utils import RemoteDataError
from yahooquery import Ticker
from pathlib import Path
import datetime
import sys
import os
import brownbear as bb

#####################################################################
# FETCH TIMESERIES

def fetch_timeseries(symbols, start=None, end=None, refresh=False):

    if start is None:
        start = datetime.datetime(2015, 1, 1)
    if end is None:
        end = datetime.datetime.now() - datetime.timedelta(1)

    if not os.path.exists(bb.SYMBOL_CACHE):
        os.makedirs(bb.SYMBOL_CACHE)  

    for symbol in symbols:
        print('.', end='')
        filepath = Path('{}/{}.csv'.format(bb.SYMBOL_CACHE, symbol))
        if refresh or not os.path.isfile(filepath):
            try:
                df = pdr.DataReader(symbol, 'yahoo', start, end)
            except RemoteDataError as e:
                print('\n({})'.format(e))
            except Exception as e:
                print('\n({})'.format(e))
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv(filepath)
    print()

#####################################################################
# COMPILE TIMESERIES

def compile_timeseries(symbols):

    compiled_df = pd.DataFrame()

    for symbol in symbols:
        filepath = Path('{}/{}.csv'.format(bb.SYMBOL_CACHE, symbol))
        df = pd.read_csv(filepath)
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': symbol}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if compiled_df.empty:
            compiled_df = df
        else:
            compiled_df = compiled_df.join(df, how='outer')
    compiled_df.to_csv('symbols-timeseries.csv')

#####################################################################
# REMOVE CACHE SYMBOLS

def remove_cache_symbols(symbols=None):
    """
    Remove cached timeseries for list of symbols.
    If symbols is None, remove all timeseries.
    Filter out any symbols prefixed with '__'
    """

    if symbols:
        # in case user forgot to put single symbol in list
        if not isinstance(symbols, list):
            symbols = [symbols]
        filenames = [symbol.upper() + '.csv' for symbol in symbols]
    else:
        filenames = [f for f in os.listdir(bb.SYMBOL_CACHE) if f.endswith('.csv')]

    # filter out any filename prefixed with '__'
    filenames = [f for f in filenames if not f.startswith('__')]

    print('removing symbols:')
    for i, f in enumerate(filenames):
        symbol = os.path.splitext(f)[0]
        print(symbol + ' ', end='')
        if i % 10 == 0 and i != 0: print()

        filepath = os.path.join(bb.SYMBOL_CACHE, f)
        if os.path.exists(filepath):
            os.remove(filepath)
        else:
            print('\n({} not found)'.format(f))
    print()


#####################################################################
# UPDATE CACHE SYMBOLS

def update_cache_symbols(symbols=None, from_year=None):
    """
    Update cached timeseries for list of symbols.
    If symbols is None, update all timeseries.
    Filter out any filename prefixed with '__'
    """

    if symbols:
        # in case user forgot to put single symbol in list
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = ([f for f in os.listdir(bb.SYMBOL_CACHE)
                     if f.endswith('.csv') and not f.startswith('__')])
        symbols = [os.path.splitext(filename)[0] for filename in filenames]

    # make symbol names uppercase
    symbols = [symbol.upper() for symbol in symbols]

    # update timeseries for symbols
    bb.fetch_timeseries(symbols, refresh=True)


def _difference_in_years(start, end):
    """ calculate the number of years between two dates """
    diff  = end - start
    diff_in_years = (diff.days + diff.seconds/86400)/365.2425
    return diff_in_years

#####################################################################
# GET SYMBOL METADATA

def get_symbol_metadata(symbols=None):
    """
    Get symbol metadata for list of symbols.
    If symbols is None, get metadata for all timeseries.
    Filter out any filename prefixed with '__'
    """

    if symbols:
        # in case user forgot to put single symbol in list
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = ([f for f in os.listdir(bb.SYMBOL_CACHE)
                     if f.endswith('.csv') and not f.startswith('__')])
        symbols = [os.path.splitext(filename)[0] for filename in filenames]

    # make symbol names uppercase
    symbols = [symbol.upper() for symbol in symbols]

    l = []
    for i, symbol in enumerate(symbols):
            filepath = Path('{}/{}.csv'.format(bb.SYMBOL_CACHE, symbol))
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


#####################################################################
# GET SYMBOL FUNDAMENTALS

def get_symbol_fundamentals(symbols=None):
    """
    Get symbol fundamental data for list of symbols.
    If symbols is None, get fundamental data for all timeseries.
    Filter out any filename prefixed with '__'
    """

    if symbols:
        # in case user forgot to put single symbol in list
        if not isinstance(symbols, list):
            symbols = [symbols]
    else:
        filenames = ([f for f in os.listdir(bb.SYMBOL_CACHE)
                     if f.endswith('.csv') and not f.startswith('__')])
        symbols = [os.path.splitext(filename)[0] for filename in filenames]

    # make symbol names uppercase
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
                print('\n({})'.format(d))

            t = (symbol, previousClose, trailingPE, dividendYield, marketCap)
            l.append(t)
    
    columns = ['symbol', 'previousClose', 'trailingPE', 'dividendYield', 'marketCap']
    df = pd.DataFrame(l, columns=columns)
    df.set_index('symbol', inplace=True)
    return df

