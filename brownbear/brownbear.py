"""
brownbear
---------
portfolio analysis and optimization
"""

# imports
import pandas as pd
import itertools
import math
import numpy
import random
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import brownbear as bb

# this is a pointer to the module object instance itself.
__m = sys.modules[__name__]

# we can explicitly make assignments on it, as follows

# list of investment galaxies, type string
__m.investment_universe = None
# correlation table of asset classes within universe , type dict
__m.correlation_table = None
# default correlation when none is specified
__m.default_correlation = None
# volatility column to use for asset allocation
__m.vola_column = None

#####################################################################
# INVESTMENT OPTIONS

def _cvs_to_df(filepaths):
    ''' read multiple csv files into a dataframe '''
    l = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, skip_blank_lines=True, comment='#')
        l.append(df)
    df = pd.concat(l)
    return df

def _correlation_table_to_dict():
    ''' return a dictionary of the correlation_table '''
    df = __m.correlation_table.set_index(['Asset Class A', 'Asset Class B'])
    # make any na values perfectly correlated=1; convert to float
    df['Correlation'] = df['Correlation'].fillna(1)
    df['Correlation'] = df['Correlation'].astype(float)
    d = df['Correlation'].to_dict()
    return d

def fetch(investment_universe, annual_returns='Annual Returns', vola='Std Dev'):
    ''' Fetch Investment Universe and asset classes
        investment-options.csv format:
          "Investment Option", "Description"(optimal), "Asset Class","Annual Returns","Std Dev"
          annual_returns is the field to use for annualized returns
        asset-classes.csv format:
          "Asset Class A","Asset Class B","Correlation"
        Note: "Description" field is optional.  It is not referenced in code.
        Note: "Annual Returns" column(s) can named anything.
              Recommend "1 Yr", "3 Yr", "5 Yr", or "10 Yr".  Then annual_returns
              parameter can select the column to use.
    '''

    # if user specified a single filename, put it in a list
    if not isinstance(investment_universe, list):
        investment_universe = [investment_universe]

    # create the investment options csv file list, then read into a dataframe
    # there are 2 places to look for investment-options.csv files, under
    # universe/ and /portfolios
    filepaths = []
    for galaxy in investment_universe:
        for subdir in ['/universe/', '/portfolios/']:
            filepath = Path(bb.ROOT + subdir + galaxy + '/investment-options.csv')
            if os.path.isfile(filepath):
                filepaths.append(filepath)

    inv_opts = _cvs_to_df(filepaths)
    # drop duplicate Investment Option's, keep the first, then reset index
    inv_opts.drop_duplicates(subset=['Investment Option'], keep='first', inplace=True)
    inv_opts.reset_index(drop=True, inplace=True)

    # allows the use of different annualized returns,
    # e.g. 1, 3, or 5 year annaulized returns.
    if annual_returns != 'Annual Returns':
        inv_opts['Annual Returns'] = inv_opts[annual_returns]

    # create the asset classes csv file list, then read into a dataframe
    # there are 2 places to look for investment-options.csv files, under
    # universe/ and /portfolios and also the master asset-class.csv file
    # which is directly under universe/
    filepaths = [Path(bb.ROOT + '/universe' + '/asset-classes.csv')]
    for galaxy in investment_universe:
        for subdir in ['/universe/', '/portfolios/']:
            filepath = Path(bb.ROOT + subdir + galaxy + '/asset-classes.csv')
            if os.path.isfile(filepath):
                filepaths.append(filepath)

    # save to module variables
    __m.investment_universe = investment_universe.copy()
    __m.vola_column = vola
    __m.correlation_table = _cvs_to_df(filepaths)
    # convert correlation table to dict for easier faster processing
    __m.correlation_table = _correlation_table_to_dict()

    return inv_opts

def _sharpe_ratio(annual_ret, std_dev, risk_free_rate):
    ''' return the sharpe ratio - this is the modified sharpe ratio
        formulated by Craig L. Israelsen.  It's the same as the sharpe
        ration when the excess return is positive.
    '''
    if math.isclose(std_dev, 0):
        return 0
    else:
        excess_return = annual_ret - risk_free_rate
        divisor = std_dev if excess_return > 0 else 1/std_dev
        return excess_return / divisor

def add_sharpe_ratio_column(df, risk_free_rate):
    ''' Add Sharpe Ratio column to dataframe '''
    def _sharpe(row, risk_free_rate):
        annual_ret = row['Annual Returns']
        std_dev = row['Std Dev']
        return _sharpe_ratio(annual_ret, std_dev, risk_free_rate)

    df['Sharpe Ratio'] = df.apply(_sharpe, risk_free_rate=risk_free_rate, axis=1)
    return df

def rank(df, rank_by, group_by=None, num_per_group=3, ascending=False):
    ''' return dataframe of ranked investment choices optionally
        grouped by asset class or subclass
        group_by = None, Asset Class, or Asset Subclass
    '''

    df = df.copy()

    def _remove_subclass(row):
        class_name = row['Asset Class']
        class_name = class_name.split(':')[0]
        return class_name

    # remove the subclass from 'Asset Class' column
    if group_by == 'Asset Class':
        df['Asset Class'] = df.apply(_remove_subclass, axis=1)

    # sort
    if group_by is None:
        df = df.sort_values(rank_by, ascending=ascending).head(num_per_group)
    elif group_by in ['Asset Class', 'Asset Subclass']:
        df = df.sort_values(['Asset Class', rank_by],
                ascending=False).groupby('Asset Class').head(num_per_group)
    else:
        raise Exception("Error: Invalid value for groupby: '{}'".format(group_by))

    return df

#####################################################################
# METRIC FUNCTIONS

def _expected_return(annual_returns, weights):
    ''' returns expected return given list of investment option returns and
        their weights
    '''
    return sum(numpy.multiply(annual_returns, weights))

def _correlation(correlation_table, a, b):
    ''' return the correlation between asset a and b using the
        correlation table dict.
        assets are in the form a=asset_class:asset_subclass
    '''
    corr = None
    a_asset_class = a.split(':')[0]
    b_asset_class = b.split(':')[0]

    # compare asset_class:asset_subclass to asset_class:asset_subclass
    if (a, b) in correlation_table:
        corr = correlation_table[a, b]
    elif (b, a) in correlation_table:
        corr = correlation_table[b, a]
    # compare asset_class to asset_class:asset_subclass
    elif (a_asset_class, b) in correlation_table:
        corr = correlation_table[a_asset_class, b]  
    elif (b, a_asset_class) in correlation_table:
        corr = correlation_table[b, a_asset_class]
    # compare asset_class:asset_subclass to asset_class
    elif (a, b_asset_class) in correlation_table:
        corr = correlation_table[a, b_asset_class]
    elif (b_asset_class, a) in correlation_table:
        corr = correlation_table[b_asset_class, a]
    # compare asset_class to asset_class
    elif (a_asset_class, b_asset_class) in correlation_table:
        corr = correlation_table[a_asset_class, b_asset_class]
    elif (b_asset_class, a_asset_class) in correlation_table:
        corr = correlation_table[b_asset_class, a_asset_class]
    else:
        corr = __m.default_correlation

    bb.DBG('corr({},{}) = {}'.format(a, b, corr))
    return corr

def _standard_deviation(weights, std_devs, asset_classes):
    ''' return std_dev given lists of weights, std_devs, and asset_classes
        ref: https://en.wikipedia.org/wiki/Modern_portfolio_theory
    '''

    weights_sq  = [x*x for x in weights]
    std_devs_sq = [x*x for x in std_devs]
    variance = sum(numpy.multiply(weights_sq, std_devs_sq))

    # calculate the correlation components, use all combinations of investment pairs
    if asset_classes:
        for a, b in itertools.combinations(enumerate(asset_classes), 2):
            corr = _correlation(correlation_table, a[1], b[1])
            a_i = a[0]; b_i = b[0];
            variance += 2*weights[a_i]*weights[b_i]*std_devs[a_i]*std_devs[b_i]*corr
    std_dev = numpy.sqrt(variance)
    return std_dev

def _metric_lists(df, portfolio_option, refresh=False):
    ''' creates lists for investment option, std_dev, and asset_class
        for each investment option for the specified portfolio.
        returns a cache copy unless refreshset is True
    '''

    _metric_lists.weights  = [value for value in portfolio_option.values()]

    if refresh:
        # check for valid investment options
        available_inv_opts = list(df['Investment Option'])
        for key in portfolio_option.keys():
            if key not in available_inv_opts:
                raise Exception("Error: Portfolio option '{}' not in {}!!!"
                                .format(key, __m.investment_universe))

        _metric_lists.inv_opts = [key for key in portfolio_option.keys()]

        inv_opts = _metric_lists.inv_opts
        _metric_lists.annual_returns = \
                [df.loc[df['Investment Option'] == inv_opt,
                'Annual Returns'].values[0] for inv_opt in inv_opts]
        _metric_lists.std_devs = \
                [df.loc[df['Investment Option'] == inv_opt,
                'Std Dev'].values[0] for inv_opt in inv_opts]
        _metric_lists.asset_classes = \
                [df.loc[df['Investment Option'] == inv_opt,
                'Asset Class'].values[0] for inv_opt in inv_opts]
        _metric_lists.volas = \
                [df.loc[df['Investment Option'] == inv_opt,
                __m.vola_column].values[0] for inv_opt in inv_opts]
        
    else:
        # return cache
        pass

    return (_metric_lists.inv_opts,
            _metric_lists.weights,
            _metric_lists.annual_returns,
            _metric_lists.std_devs,
            _metric_lists.asset_classes,
            _metric_lists.volas)

_metric_lists.inv_opts = None
_metric_lists.weights = None
_metric_lists.annual_returns = None
_metric_lists.std_devs = None
_metric_lists.asset_classes = None
_metric_lists.volas = None

#####################################################################
# ANALYZE

def _process_options(df, portfolio_option, risk_free_rate,
                     use_equal_weights,
                     use_sharpe_ratio_adjusted_weights,
                     use_volatility_adjusted_weights,
                     default_correlation):

    ''' This function may portfolio_option wwights if,...
        use_equal_weights=True will override the precentages specified
          in portfolio_option and use equal weights instead
        use_sharpe_ratio_adjusted_weights=True will override the precentages specified
          in portfolio_option and use proportionally weighted
          allocations based on the percent of an investment option's sharpe ratio
          to the sum of all the sharpe ratios in the portfolio.  For this
          operation, any negative sharpe ratios are set to zero.
        use_volatility_adjusted_weights=True will override the precentages specified
          in portfolio_option and use volatility adjusted weights instead

        default_correlation - correlation to use when no correlation has been
          specified between two asset classes.  If you use only the Asset Classes
          defined in universe/asset-classes.csv, then this will never happen.
    '''

    __m.default_correlation = default_correlation

    inv_opts, weights, annual_returns, std_devs, asset_classes, volas = \
        _metric_lists(df, portfolio_option, refresh=True)

    n = len(portfolio_option)

    if use_equal_weights:
        portfolio_option = dict.fromkeys(portfolio_option, 1/n)
    elif use_sharpe_ratio_adjusted_weights:
        sharpe_ratios = []
        for i, inv_opt in enumerate(inv_opts):
            sharpe_ratios.append(_sharpe_ratio(annual_returns[i], std_devs[i],
                                               risk_free_rate))
        # investment options with negative sharpe ratios will have weight=0
        sharpe_ratios = [sr if sr >= 0 else 0 for sr in sharpe_ratios]
        sharpe_ratio_sum = sum(sharpe_ratios)
        if math.isclose(sharpe_ratio_sum, 0):
            raise Exception('Error: Can\'t allocate Portfolio.\n'
                            'All investment options have negative sharpe ratios')
        weights = [sharpe_ratio/sharpe_ratio_sum for sharpe_ratio in sharpe_ratios]
        portfolio_option.update(zip(portfolio_option, weights))
    elif use_volatility_adjusted_weights:
        inverse_volas = [1/0.001 if math.isclose(vola, 0) else 1/vola \
                         for vola in volas]
        inverse_vola_sum = sum(inverse_volas)
        weights = [inverse_vola/inverse_vola_sum for inverse_vola in inverse_volas]
        portfolio_option.update(zip(portfolio_option, weights))

def analyze(df, portfolio_option, risk_free_rate,
            use_equal_weights=False,
            use_sharpe_ratio_adjusted_weights=False,
            use_volatility_adjusted_weights=False,
            default_correlation=1):
    ''' analyze portfolio_option and return the annual_ret, std_dev,
        sharpe_ratio
    '''

    # make a local copy of portfolio_option and pop the title
    __m.portfolio_title = portfolio_option.pop('Title', 'Portfolio')

    # process options
    _process_options(df, portfolio_option, risk_free_rate,
                     use_equal_weights,
                     use_sharpe_ratio_adjusted_weights,
                     use_volatility_adjusted_weights,
                     default_correlation)

    # make sure total adds to 100
    s = sum(portfolio_option.values())
    if not math.isclose(s, 1, rel_tol=1e-09, abs_tol=0.0):
        raise Exception('Error: Portfolio allocation of \'{}\' is not 100%!!!'
                        .format(s))

    inv_opts, weights, annual_returns, std_devs, asset_classes, volas = \
        _metric_lists(df, portfolio_option)

    # compute metrics
    annual_return = _expected_return(annual_returns=annual_returns,
                                     weights=weights)
    std_dev = _standard_deviation(weights=weights, std_devs=std_devs,
                                  asset_classes=asset_classes)
    sharpe_ratio = _sharpe_ratio(annual_return, std_dev, risk_free_rate)

    return annual_return, std_dev, sharpe_ratio

#####################################################################
# SUMMARY

def summary(df, portfolio_option, annual_ret, std_dev, sharpe_ratio):
    ''' return a dataframe with summary results
        Note: analyze must be called before calling summary
    '''

    def _plot_returns(summary, columns):
        ''' Bar Plot of returns with 1, 2, and 3 standard deviations '''
        means = list(summary.loc['Annual Returns'])
        Xs = list(range(0,len(means)))
        # plot 1 std dev
        maxs =  list(summary.loc['Std Dev'])
        plt.errorbar(Xs, means, maxs, fmt='.k', lw=20)
        # plot 2 std dev
        maxs_2 =  [x * 2 for x in maxs]
        plt.errorbar(Xs, means, maxs_2, fmt='.k', lw=5)
        # plot 3 std dev
        maxs_3 =  [x * 3 for x in maxs]
        plt.errorbar(Xs, means, maxs_3, fmt='.k', lw=1)
        # plot horizontal line for median
        max_std_dev = max(maxs)
        maxs_median =  [max_std_dev*.02 for x in means]
        plt.errorbar(Xs, means, maxs_median, fmt='.k', lw=50)
        plt.xlim(-1, len(means))
        plt.xticks(range(len(columns)), columns, rotation=60)

    inv_opts, weights, annual_returns, std_devs, asset_classes, volas = \
        _metric_lists(df, portfolio_option)

    metrics = ['Annual Returns', 'Std Dev', 'Sharpe Ratio']
    index = []
    columns = [inv_opt for inv_opt in inv_opts]
    data = []

    # add metrics
    for metric in metrics:
        index.append(metric)
        data.append([df.loc[df['Investment Option'] == inv_opt, metric].values[0]
                    for inv_opt in inv_opts])

    # add weight
    index.append('Weight')
    data.append([portfolio_option[inv_opt] for inv_opt in inv_opts])

    # add Black Swan
    index.append('Black Swan')
    data.append([df.loc[df['Investment Option'] == inv_opt, 'Annual Returns'].values[0] +
              -3*df.loc[df['Investment Option'] == inv_opt, 'Std Dev'].values[0]
                for inv_opt in inv_opts])

    # create dataframe
    summary = pd.DataFrame(data, columns=columns, index=index)

    # set portfolio values
    summary[__m.portfolio_title] = \
        [annual_ret, std_dev, sharpe_ratio, sum(weights), annual_ret + -3*std_dev]

    # plot returns
    column_names = columns.copy()
    column_names.append(__m.portfolio_title)
    _plot_returns(summary, column_names)

    return summary

#####################################################################
# OPTIMIZER

def _sim(annual_returns, std_devs, asset_classes,
         risk_free_rate, min_annual_return, max_black_swan):
    ''' Calculation is done via a Monte Carlo Simulation by trying random
        combinations of weights and checking which combination has the best
        sharpe_ratio
    '''

    o_sharpe_ratio = 0
    o_weights = []

    N = 100000
    i = 0
    while i < N:
        rands = [round(random.random(), 2) for x in range(0, len(annual_returns))]
        s = sum(rands)
        if math.isclose(s, 0): continue
        rands = [x/s for x in rands]

        annual_return = _expected_return(annual_returns=annual_returns, weights=rands)
        std_dev = _standard_deviation(weights=rands, std_devs=std_devs,
                                      asset_classes=asset_classes)
        sharpe_ratio = _sharpe_ratio(annual_return, std_dev, risk_free_rate)
        black_swan = annual_return - 3*std_dev

        if (annual_return < min_annual_return or 
            black_swan < max_black_swan):
            pass
        elif sharpe_ratio > o_sharpe_ratio:
            o_sharpe_ratio = sharpe_ratio
            o_weights = rands
            print('.', end='')
            #print(i)
            i = 0
        i += 1
    success = o_sharpe_ratio != 0

    return success, o_weights

def optimizer(df, portfolio_option, risk_free_rate, min_annual_return=None,
              max_black_swan=None):
    ''' Optimize sharpe ratio while allowing a min_annual_rate and/or
        max_black_swan.  Setting min_annual_return and max_black_swan to None
        optimizes absolute sharpe_ratio without regard to these quantities.
    '''

    if min_annual_return is None: min_annual_return = 0
    if max_black_swan is None:    max_black_swan = -1000

    inv_opts, weights, annual_returns, std_devs, asset_classes, volas = \
        _metric_lists(df, portfolio_option)

    print('Running optimizer', end='')
    success, optimal_weights = \
        _sim(annual_returns, std_devs, asset_classes, risk_free_rate,
             min_annual_return, max_black_swan)
    print('\n')

    if not success:
        print('Impossible criteria specified, lower your expectations!!!')
        return

    # round weights to 2 decimal places
    optimal_weights = [round(x, 2) for x in optimal_weights]
    # set any weights less than 0.03 to 0
    optimal_weights = [0 if x < 0.03 else x for x in optimal_weights]

    # insure the weights add to 1
    slots = [i for i, x in enumerate(optimal_weights) if x > 0]
    for i in range(1000):
        if math.isclose(sum(optimal_weights), 1):
            break
        factor = -1 if sum(optimal_weights) > 1 else 1
        optimal_weights[random.choice(slots)] += .01*factor
    if not math.isclose(sum(optimal_weights), 1):
        print('Error: weights don\'t add to 1')

    # make a local copy of portfolio_options
    _portfolio_option = portfolio_option.copy()

    # display results
    annual_return = _expected_return(annual_returns=annual_returns,
                                     weights=optimal_weights)
    std_dev = _standard_deviation(weights=optimal_weights, std_devs=std_devs,
                                  asset_classes=asset_classes)
    sharpe_ratio = _sharpe_ratio(annual_return, std_dev, risk_free_rate)
    black_swan = annual_return - 3*std_dev

    print('{} Metrics:'.format(__m.portfolio_title))
    print('    max_sharpe_ratio: {:0.2f}'.format(sharpe_ratio))
    print('    annual_return: {:0.2f}'.format(annual_return))
    print('    std_dev: {:0.2f}'.format(std_dev))
    print('    black_swan: {:0.2f}'.format(annual_return - 3*std_dev))
    print()
    print('{} Weights:'.format(__m.portfolio_title))
    _portfolio_option.update(zip(_portfolio_option, optimal_weights))
    for key, value in _portfolio_option.items():
        print('    {}: {:0.2f}'.format(key, value))
    print()

    # check to see if one of the inv_options is a risk_free asset
    for i, std_dev in enumerate(std_devs):
        if math.isclose(std_dev, 0):
            print('NOTE: \'{}\' is a risk free asset.\n'
                  '      Although risk free assets don\'t affect the sharpe ratio of a portfolio,\n'
                  '      adding a risk free asset does reduce the black swawn percent.'
                  .format(inv_opts[i]))
            break

