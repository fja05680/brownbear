"""
metrics
---------
functions to compute metrics
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import math
import brownbear as bb

# constants
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 20
TRADING_DAYS_PER_WEEK = 5

def correlation_map(df, method='log', days=None):
    """ return correlation dataframe; show correlation map between symbols """

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
        annual_return = cagr(B, A, n)
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
