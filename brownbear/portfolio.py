"""
Portfolio analysis and optimization.
"""

import itertools
import math

import numpy

import brownbear as bb
from brownbear.utility import (
    dotdict
)


class Portfolio:
    """
    Portfolio class.
    """
    def __init__(self):
        """
        Initialize instance variables.

        Attributes
        ----------
        investment_universe : str
            List of investment galaxies.
        risk_free_rate : float
            Risk free rate.
        correlation_table : dict
            Correlation table of asset classes within universe.
        asset_class_table : pd.DataFrame
            Asset class returns.
        default_correlation : int
            Default correlation when none is specified.
        vola_column : str
            Column name to use for vola.
        ds_vola_column : str
            Column name to use for ds_vola.
        portfolio_title : str
            Portfolio title.
        """
        self.investment_universe = None
        self.risk_free_rate = None
        self.correlation_table = None
        self.asset_class_table = None
        self.default_correlation = None
        self.vola_column = None
        self.ds_vola_column = None
        self.portfolio_title = 'Portfolio'

PORT = Portfolio()
"""
class : Single instance of Portfolio object to be used globally.
"""


def get_metric_lists(df, portfolio_option):
    """
    Creates lists for investment option, std_dev, asset_class, etc...
    for each investment option for the specified portfolio.
    """

    # Check for valid investment options.
    available_inv_opts = list(df['Investment Option'])
    bb.DBG('available_inv_opts', available_inv_opts)
    bb.DBG('portfolio_option', portfolio_option)
    for key in portfolio_option.keys():
        if key not in available_inv_opts:
            raise Exception("Error: Portfolio option '{}' not in {}!!!"
                            .format(key, PORT.investment_universe))
    ml = dotdict()

    ml.inv_opts = \
        [key for key in portfolio_option.keys()]
    ml.weights = \
        [value for value in portfolio_option.values()]
    ml.asset_classes = \
        [df.loc[df['Investment Option'] == inv_opt,
        'Asset Class'].values[0] for inv_opt in ml.inv_opts]
    ml.volas = \
        [df.loc[df['Investment Option'] == inv_opt,
        PORT.vola_column].values[0] for inv_opt in ml.inv_opts]
    ml.ds_volas = \
        [df.loc[df['Investment Option'] == inv_opt,
        PORT.ds_vola_column].values[0] for inv_opt in ml.inv_opts]
    ml.std_devs = \
        [df.loc[df['Investment Option'] == inv_opt,
        'Std Dev'].values[0] for inv_opt in ml.inv_opts]
    ml.annual_returns = \
        [df.loc[df['Investment Option'] == inv_opt,
        'Annual Returns'].values[0] for inv_opt in ml.inv_opts]
    ml.sharpe_ratios = \
        [df.loc[df['Investment Option'] == inv_opt,
        'Sharpe Ratio'].values[0] for inv_opt in ml.inv_opts]

    return ml


def expected_return(annual_returns, weights):
    """
    Returns expected return given list of investment option returns and
    their corresponding weights.
    """
    return sum(numpy.multiply(annual_returns, weights))


def correlation(correlation_table, a, b):
    """
    Return the correlation between asset a and b using the
    correlation table dict.  Assets are in the form
    a=asset_class:asset_subclass.
    """
    corr = None
    a_asset_class = a.split(':')[0]
    b_asset_class = b.split(':')[0]

    # Compare asset_class:asset_subclass to asset_class:asset_subclass.
    if (a, b) in correlation_table:
        corr = correlation_table[a, b]
    elif (b, a) in correlation_table:
        corr = correlation_table[b, a]
    # Compare asset_class to asset_class:asset_subclass.
    elif (a_asset_class, b) in correlation_table:
        corr = correlation_table[a_asset_class, b]  
    elif (b, a_asset_class) in correlation_table:
        corr = correlation_table[b, a_asset_class]
    # Compare asset_class:asset_subclass to asset_class.
    elif (a, b_asset_class) in correlation_table:
        corr = correlation_table[a, b_asset_class]
    elif (b_asset_class, a) in correlation_table:
        corr = correlation_table[b_asset_class, a]
    # Compare asset_class to asset_class.
    elif (a_asset_class, b_asset_class) in correlation_table:
        corr = correlation_table[a_asset_class, b_asset_class]
    elif (b_asset_class, a_asset_class) in correlation_table:
        corr = correlation_table[b_asset_class, a_asset_class]
    else:
        corr = PORT.default_correlation

    #bb.DBG(f'_correlation: corr({a},{b}) = {corr}')
    return corr


def standard_deviation(weights, std_devs, asset_classes):
    """
    Return std_dev given lists of weights, std_devs, and asset_classes
    Reference: https://en.wikipedia.org/wiki/Modern_portfolio_theory
    """
    weights_sq  = [x*x for x in weights]
    std_devs_sq = [x*x for x in std_devs]
    variance = sum(numpy.multiply(weights_sq, std_devs_sq))

    # Calculate the correlation components, use all combinations of
    # investment pairs.
    if asset_classes:
        for a, b in itertools.combinations(enumerate(asset_classes), 2):
            corr = correlation(PORT.correlation_table, a[1], b[1])
            a_i = a[0]
            b_i = b[0]
            variance += 2*weights[a_i]*weights[b_i]*std_devs[a_i]*std_devs[b_i]*corr
    std_dev = numpy.sqrt(variance)
    return std_dev


def sharpe_ratio(annual_ret, std_dev, risk_free_rate):
    """
    Return the sharpe ratio.

    This is the modified sharpe ratio formulated by Craig L. Israelsen.
    It's the same as the sharpe ratio when the excess return is
    positive.
    """
    if math.isclose(std_dev, 0):
        return 0
    excess_return = annual_ret - risk_free_rate
    divisor = std_dev if excess_return > 0 else 1/std_dev
    return excess_return / divisor