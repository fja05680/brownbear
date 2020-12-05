"""
utils
---------
some useful utility functions
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
from pandas_datareader._utils import RemoteDataError
from pathlib import Path
import pkg_resources
import seaborn
import datetime
import os
import math
import brownbear as bb

# brownbear project root dir
ROOT = str(Path(os.getcwd().split('brownbear')[0] + '/brownbear'))

# symbol cache location
SYMBOL_CACHE = str(Path(ROOT + '/symbol-cache'))

# constants
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 20
TRADING_DAYS_PER_WEEK = 5

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def correlation_map(df, method='log', days=None):
    """ return correlation dataframe; show correlation map between symbols"""

    # default is all days
    if days is None:
        days = 0;
    df = df[-days:]

    if method == 'price':
        pass
    elif method == 'log':
        df = np.log(df.pct_change()+1)
    elif method == 'returns':
        df = df.pct_change()

    df = df.corr(method='pearson')
    # take the bottom triangle since it repeats itself
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True
    # generate plot
    fig = plt.figure(figsize=(16,12))
    seaborn.heatmap(df, cmap='RdYlGn', vmax=1.0, vmin=-1.0 ,
                    mask = mask, linewidths=2.5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    return df

def fetch_timeseries(symbols, start=None, end=None, refresh=False):

    if start is None:
        start = datetime.datetime(2015, 1, 1)
    if end is None:
        end = datetime.datetime.now()

    if not os.path.exists(SYMBOL_CACHE):
        os.makedirs(SYMBOL_CACHE)  

    for symbol in symbols:
        print('.', end='')
        filepath = Path('{}/{}.csv'.format(SYMBOL_CACHE, symbol))
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

def compile_timeseries(symbols):

    compiled_df = pd.DataFrame()

    for symbol in symbols:
        filepath = Path('{}/{}.csv'.format(SYMBOL_CACHE, symbol))
        df = pd.read_csv(filepath)
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': symbol}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if compiled_df.empty:
            compiled_df = df
        else:
            compiled_df = compiled_df.join(df, how='outer')
    compiled_df.to_csv('symbols-timeseries.csv')

def cagr(B, A, n):
    """ calculate compound annual growth rate
        B = end balance; A = begin balance; n = num years
    """
    if B < 0: B = 0
    return (math.pow(B / A, 1 / n) - 1) * 100

def annualize_returns(prices, timeperiod='daily', years=5):
    ''' calculate the annualized returns of entire dataframe
        for the timeframes: daily, weekly, monthly, or quartely
        returns a series of annualized returns
    '''

    def _cagr(column, n, f):
        time_units = int(n*f)
        A = column[-(time_units+1)]
        B = column[-1]
        annual_return = bb.cagr(B, A, n)
        return annual_return

    factor = None
    if   timeperiod == 'daily':     factor = 252
    elif timeperiod == 'weekly':    factor = 52
    elif timeperiod == 'monthly':   factor = 12
    elif timeperiod == 'quarterly': factor = 4

    s = prices.apply(_cagr, n=years, f=factor, axis=0)
    return s

def annualized_standard_deviation(returns, timeperiod='monthly', years=3,
                                  downside=False):
    ''' returns the annualized standard deviation of entire dataframe
        for the timeframes: daily, weekly, monthly, or quartely
        returns a series of annualized standard deviations
    '''

    if   timeperiod == 'daily':     factor = 252
    elif timeperiod == 'weekly':    factor = 52
    elif timeperiod == 'monthly':   factor = 12
    elif timeperiod == 'quarterly': factor = 4


    # use downside deviation?
    if downside:
        _returns = returns.copy()
        _returns[_returns > 0] = 0
    else:
        _returns = returns

    # calculate annualized std_dev
    dev = np.std(_returns.tail(int(factor*years)), axis=0)
    dev = dev * math.sqrt(factor)
    return dev

