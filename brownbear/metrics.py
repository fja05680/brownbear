"""
Compute metrics.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn


def correlation_map(df, method='log', days=None):
    """
    Compute and Plot correlation map between symbols.

    See example use in asset-class-galaxy/asset-classes.ipynb

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries with a column for each symbol.  Each row contains
        the daily closing prices for the symbols.
    method : str, optional {log, price, returns}
        Timeseries can be altered so that correlation is based on
        a price derivative (log or returns) instead of price.  'price'
        does not alter the timeseries.  (default is 'log').
    days : int, optional
        The last number of days over which to compute the correlations.
        (default is None, which implies all days).

    Returns
    -------
    df : pd.DataFrame
        Dataframe representing the correlation map between symbols.
    """ 
    method_choices = ('price', 'log', 'returns')
    assert method in method_choices, f"Invalid method '{method}'"

    if days is None:
        days = 0
    df = df[-days:]

    if method == 'price':
        pass
    elif method == 'log':
        df = np.log(df.pct_change()+1)
    elif method == 'returns':
        df = df.pct_change()

    df = df.corr(method='pearson')

    # Take the bottom triangle since it repeats itself.
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True

    # Generate plot.
    plt.figure(figsize=(16, 12))
    seaborn.heatmap(df, cmap='RdYlGn', vmax=1.0, vmin=-1.0, mask=mask,
                    linewidths=2.5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    return df


def cagr(B, A, n):
    """
    Calculate compound annual growth rate.

    Parameters
    ----------
    B : float
        Ending balance.
    A : float
        Beginning balance.
    n : float
        Number of years over which to calculate cagr.

    Returns
    -------
    float
        Compound annual growth rate.
    """
    if B < 0:
        B = 0
    return (math.pow(B / A, 1 / n) - 1) * 100


def annualized_returns(df, timeperiod='daily', years=5):
    """
    Calculate the annualized returns of entire dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries with a column for each symbol.  Each row contains
        the daily closing prices for the symbols.
    timeperiod : str, optional {'daily', 'weekly', 'monthly',
        'quarterly'}
        Specifies the sample rate of the timeseries 'df'
        (default is 'daily').
    years : float, optional
        Number of years over which to calculate annualized returns
        (default is 5).

    Returns
    -------
    s : pd.Series
        Series of key[value] pairs in which key is the symbol and
        value is the annualized return.
    """
    def _cagr(column, n, f):
        time_units = int(n*f)
        A = column[-(time_units+1)]
        B = column[-1]
        annual_return = cagr(B, A, n)
        return annual_return

    timeperiod_choices = ('daily', 'weekly', 'monthly', 'quarterly')
    assert timeperiod in timeperiod_choices, f"Invalid timeperiod '{timeperiod}'"

    factor = {'daily': 252, 'weekly': 52, 'monthly': 12, 'quarterly': 4}
    factor = factor[timeperiod]

    s = df.apply(_cagr, n=years, f=factor, axis=0)
    return s


def annualized_standard_deviation(returns, timeperiod='monthly', years=3,
                                  downside=False):
    """
    Calculate the annualized standard deviation of entire dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries with a column for each symbol.  Each row contains
        the daily closing prices for the symbols.
    timeperiod : str, optional {'daily', 'weekly', 'monthly',
        'quarterly'}
        Specifies the sample rate of the timeseries 'df'
        (default is 'daily').
    years : float, optional
        Number of years over which to calculate standard deviation
        (default is 3).
    downside : bool, optional
        True to calculate the downside standard deviation, otherwise
        False (default is False).

    Returns
    -------
    s : pd.Series
        Series of key[value] pairs in which key is the symbol and
        value is the annualized standard deviation.
    """
    timeperiod_choices = ('daily', 'weekly', 'monthly', 'quarterly')
    assert timeperiod in timeperiod_choices, f"Invalid timeperiod '{timeperiod}'"

    factor = {'daily': 252, 'weekly': 52, 'monthly': 12, 'quarterly': 4}
    factor = factor[timeperiod]

    # Use downside deviation?
    if downside:
        _returns = returns.copy()
        _returns[_returns > 0] = 0
    else:
        _returns = returns

    # Calculate annualized standard deviation.
    s = np.std(_returns.tail(int(factor*years)), axis=0)
    s = s * math.sqrt(factor)
    return s

