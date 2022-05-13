"""
Portfolio analysis and optimization.
"""

import math
import random

import pandas as pd

import brownbear as bb
from brownbear.portfolio import (
    PORT,
    get_metric_lists,
    expected_return,
    standard_deviation,
    sharpe_ratio
)


########################################################################
# OPTIMIZER

def _sim(ml, min_annual_return, max_worst_typical_down_year, max_black_swan):
    """
    Calculation is done via a Monte Carlo Simulation by trying random
    combinations of weights and checking which combination has the best
    sharpe_ratio.
    """

    o_sharpe_ratio = 0
    o_weights = []

    N = 100000
    i = 0
    while i < N:
        rands = [round(random.random(), 2) for x in range(0, len(ml.annual_returns))]
        s = sum(rands)
        if math.isclose(s, 0):
            continue
        rands = [x/s for x in rands]

        annual_return = expected_return(ml.annual_returns, weights=rands)
        std_dev = standard_deviation(weights=rands, std_devs=ml.std_devs,
                                      asset_classes=ml.asset_classes)
        sr = sharpe_ratio(annual_return, std_dev, PORT.risk_free_rate)
        worst_typical_down_year = annual_return - 2*std_dev
        black_swan = annual_return - 3*std_dev

        if (annual_return < min_annual_return or
            worst_typical_down_year < max_worst_typical_down_year or
            black_swan < max_black_swan):
            pass
        elif sr > o_sharpe_ratio:
            o_sharpe_ratio = sr
            o_weights = rands
            print('.', end='')
            #print(i)
            i = 0
        i += 1
    success = o_sharpe_ratio != 0

    return success, o_weights


def optimizer(df, portfolio_option, constraints=None):
    """
    Optimize the Portfolio based on Sharpe Ratio.

    Optimize sharpe ratio while specifying Annual Rate,
    Worst Typical Down Year, and Black Swan.  Setting a constraint
    to None optimizes absolute Sharpe Ratio without regard to that
    constraint.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of investment options with columns for asset class,
        description, and performace metrics.
    portfolio_option : dict
        Dictionary of investment options along with their weights.  The
        keys are the investment options and the values are the weights.
        The first entry in the dict must be the title of the portfolio.
        `portfolio_option` may be modified if `weight_by` is not None,
        i.e. the weights for each investment option might be adjusted.
    constraints : dict
        Used to specify constraints for the optimization.  Valid
        constraints are: 'Annual Return', 'Worst Typical Down Year',
        and 'Black Swan' (default is None, which implies maximize
        Sharpe Ratio without considering any constraints).

    Returns
    -------
    None
    """
    # Unpack contraints dict.
    min_annual_return = max_worst_typical_down_year = max_black_swan = None
    if constraints:
        min_annual_return           = constraints.get('Annual Return')
        max_worst_typical_down_year = constraints.get('Worst Typical Down Year')
        max_black_swan              = constraints.get('Black Swan')

    if min_annual_return is None:           min_annual_return = 0
    if max_worst_typical_down_year is None: max_worst_typical_down_year = -1000
    if max_black_swan is None:              max_black_swan = -1000

    ml = get_metric_lists(df, portfolio_option)

    print('Running optimizer', end='')
    success, optimal_weights = \
        _sim(ml, min_annual_return, max_worst_typical_down_year, max_black_swan)
    print('\n')

    if not success:
        print('Impossible criteria specified, lower your expectations!!!')
        return

    # Round weights to 2 decimal places.
    optimal_weights = [round(x, 2) for x in optimal_weights]
    # Set any weights less than 0.03 to 0.
    optimal_weights = [0 if x < 0.03 else x for x in optimal_weights]

    # Insure the weights add to 1.
    slots = [i for i, x in enumerate(optimal_weights) if x > 0]
    for i in range(1000):
        if math.isclose(sum(optimal_weights), 1):
            break
        factor = -1 if sum(optimal_weights) > 1 else 1
        optimal_weights[random.choice(slots)] += .01*factor
    if not math.isclose(sum(optimal_weights), 1):
        print('Error: weights don\'t add to 1')

    # Make a local copy of portfolio_options.
    _portfolio_option = portfolio_option.copy()

    # Display results.
    annual_return = expected_return(annual_returns=ml.annual_returns,
                                     weights=optimal_weights)
    std_dev = standard_deviation(weights=optimal_weights, std_devs=ml.std_devs,
                                  asset_classes=ml.asset_classes)
    sr = sharpe_ratio(annual_return, std_dev, PORT.risk_free_rate)
    worst_typical_down_year = annual_return - 2*std_dev
    black_swan = annual_return - 3*std_dev
    _portfolio_option.update(zip(_portfolio_option, optimal_weights))

    s = pd.Series()
    s[PORT.portfolio_title + ' Metrics:'] = ''
    s['    max_sharpe_ratio'] = sr
    s['    annual_return'] = annual_return
    s['    std_dev'] = std_dev
    s['    worst typical down year'] = worst_typical_down_year
    s['    black_swan'] = black_swan
    s = str(s).split('dtype:')[0]
    print(s)
    print()
    bb.print_portfolio(_portfolio_option)
    print()

    # Check to see if one of the inv_options is a risk_free asset.
    risk_free_msg = \
        """
        NOTE: \'{}\' is a risk free asset.
              Although risk free assets don\'t affect the sharpe ratio of a portfolio,
              adding a risk free asset does reduce the worst typical down year
              and black swawn percentages.
        """

    for i, std_dev in enumerate(ml.std_devs):
        if math.isclose(std_dev, 0):
            print(risk_free_msg.format(ml.inv_opts[i]))
            break
