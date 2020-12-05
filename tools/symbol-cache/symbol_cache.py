"""
symbol_cache
---------
symbol cache management
"""

import sys
import pandas as pd
from pathlib import Path
import datetime
import os
import brownbear as bb

#####################################################################
# CACHE SYMBOLS (remove, update, get_symbol_metadata)

def _difference_in_years(start, end):
    """ calculate the number of years between two dates """
    diff  = end - start
    diff_in_years = (diff.days + diff.seconds/86400)/365.2425
    return diff_in_years

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
