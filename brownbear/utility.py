"""
Utility functions.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


########################################################################
# CONSTANTS

TRADING_DAYS_PER_YEAR = 252
"""
int : The number of trading days per year.
"""
TRADING_DAYS_PER_MONTH = 20
"""
int : The number of trading days per month.
"""
TRADING_DAYS_PER_WEEK = 5
"""
int : The number of trading days per week.
"""

ROOT = Path(str(Path.cwd()).split('brownbear')[0]) / 'brownbear'
"""
str : Full path to brownbear project root dir.
"""

SYMBOL_CACHE = ROOT / 'symbol-cache'
"""
str : Full path to symbol-cache dir.
"""


########################################################################
# FUNCTIONS

def csv_to_df(filepaths):
    """
    Read multiple csv files into a dataframe.

    Parameters
    ----------
    filepaths : list of str
        List of of full path to csv files.

    Returns
    -------
    df : pd.DataFrame
        Dataframe representing the concatination of the csv files
        listed in `filepaths`.
    """
    l = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, skip_blank_lines=True, comment='#')
        l.append(df)
    df = pd.concat(l)
    return df


class dotdict(dict):
    """
    Provides dot.notation access to dictionary attributes.

    Examples
    --------
    >>> mydict = {'val' : 'it works'}
    >>> mydict = dotdict(mydict)
    >>> mydict.val
    'it works'
    >>> nested_dict = {'val' : 'nested works too'}
    >>> mydict.nested = dotdict(nested_dict)
    >>> mydict.nested.val
    'nested works too'
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_full(x):
    """
    Print every row of list-like object.
    """
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def get_quote(symbols):
    """
    Returns the current quote for a list of symbols as a dict.
    """
    d = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        d[symbol] = ticker.info.get('currentPrice', 'N/A')

    return d
