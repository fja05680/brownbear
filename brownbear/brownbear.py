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
from brownbear.utility import dotdict
import brownbear as bb

# this is a pointer to the module object instance itself.
__m = sys.modules[__name__]

# we can explicitly make assignments on it, as follows

# list of investment galaxies, type string
__m.investment_universe = None
# risk_free_rate, type float
__m.risk_free_rate = None
# correlation table of asset classes within universe , type dict
__m.correlation_table = None
# asset class returns, type dataframe
__m.asset_class_table = None
# default correlation when none is specified, type int
__m.default_correlation = None
# columns to use for vola and ds_vola, type str
__m.vola_column = None
__m.ds_vola_column = None
# portfolio title, type str
__m.portfolio_title = 'Portfolio'

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

def _add_sharpe_ratio_column(df, risk_free_rate):
    ''' Add Sharpe Ratio column to dataframe '''
    def _sharpe(row, risk_free_rate):
        annual_ret = row['Annual Returns']
        std_dev = row['Std Dev']
        return _sharpe_ratio(annual_ret, std_dev, risk_free_rate)

    df['Sharpe Ratio'] = df.apply(_sharpe, risk_free_rate=risk_free_rate, axis=1)
    return df

def fetch(investment_universe, risk_free_rate=0, annual_returns='Annual Returns',
          vola='Std Dev', ds_vola='Std Dev', clean=True):
    ''' Fetch Investment Universe and asset classes
        investment-options.csv format:
          "Investment Option", "Description"(optimal), "Asset Class","Annual Returns","Std Dev"
          annual_returns is the field to use for annualized returns
        asset-classes.csv format:
          "Asset Class A","Asset Class B","Correlation"
          "Description" field is optional.  It is not referenced in code.
          "Annual Returns" column(s) can named anything.
              Recommend "1 Yr", "3 Yr", "5 Yr", or "10 Yr".  Then annual_returns
              parameter can select the column to use.
        "vola" is used to specify the volatility column.
        "ds_vola" is used to specify the downside volatility column
        "clean" is used to remove rows that have a nan as a column value
    '''

    # if user specified a single filename, put it in a list
    if not isinstance(investment_universe, list):
        investment_universe = [investment_universe]

    # create the investment options csv file list, then read into a dataframe
    # there are 2 places to look for investment-options.csv files, under
    # /universe and /portfolios
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
    # Add Sharpe Ratio column
    inv_opts = _add_sharpe_ratio_column(inv_opts, risk_free_rate)

    # create the asset classes csv file list, then read into a dataframe
    # there are 2 places to look for asset-classes.csv files, under
    # /universe/ and /portfolios and also the master asset-class.csv file
    # which is under /universe/asset-class-galaxy
    filepaths = [Path(bb.ROOT + '/universe/asset-class-galaxy/asset-classes.csv')]
    for galaxy in investment_universe:
        for subdir in ['/universe/', '/portfolios/']:
            filepath = Path(bb.ROOT + subdir + galaxy + '/asset-classes.csv')
            if (filepath not in filepaths) and os.path.isfile(filepath):
                filepaths.append(filepath)
    __m.correlation_table = _cvs_to_df(filepaths)

    # save to module variables
    __m.investment_universe = investment_universe.copy()
    __m.risk_free_rate = risk_free_rate
    __m.vola_column = vola
    __m.ds_vola_column = ds_vola
    # convert correlation table to dict for easier faster processing
    __m.correlation_table = _correlation_table_to_dict()
    __m.asset_class_table = _cvs_to_df(
        [Path(bb.ROOT + '/universe/asset-class-galaxy/investment-options.csv')])
    # add Annual Returns column to class-assets
    if annual_returns in __m.asset_class_table.columns:
        __m.asset_class_table['Annual Returns'] = __m.asset_class_table[annual_returns]
    else:
        __m.asset_class_table['Annual Returns'] = __m.asset_class_table['5 Yr']
    # add Sharpe Ratio column to class-assets
    __m.asset_class_table = _add_sharpe_ratio_column(__m.asset_class_table, risk_free_rate)
    
    if clean:
        # remove any rows that have nan for column values
        inv_opts = inv_opts.dropna()
        inv_opts.reset_index(drop=True, inplace=True)

    return inv_opts

def rank(df, rank_by, group_by=None, num_per_group=None, ascending=False):
    ''' return dataframe of ranked investment choices optionally
        grouped by asset class or subclass
        group_by = None, Asset Class, or Asset Subclass
    '''
 
    group_by_choices = (None, 'Asset Class', 'Asset Subclass')
    assert group_by in group_by_choices, \
        "Invalid group_by '{}'".format(group_by)

    df = df.copy()

    # temporarily add __asset_class__ and  __asset_subclass__ for convenience;
    # drop it later
    df['__asset_subclass__'] = df['Asset Class']

    def _add_asset_class(row):
        class_name = row['__asset_subclass__']
        class_name = class_name.split(':')[0]
        return class_name

    # remove the subclass from '__asset_class__' column
    df['__asset_class__'] = df.apply(_add_asset_class, axis=1)

    # sort
    if group_by is None:
        if num_per_group is None:  num_per_group = 10000
        df = df.sort_values(rank_by, ascending=ascending) \
                            .head(num_per_group)
    elif group_by == 'Asset Class':
        if num_per_group is None:  num_per_group = 5
        df = df.sort_values(['__asset_class__', rank_by], ascending=ascending) \
                            .groupby('__asset_class__').head(num_per_group)
    elif group_by == 'Asset Subclass':
        if num_per_group is None:  num_per_group = 5
        df = df.sort_values(['__asset_subclass__', rank_by], ascending=ascending) \
                            .groupby('__asset_subclass__').head(num_per_group)
    else:
        raise Exception("Error: Invalid value for groupby: '{}'".format(group_by))
        
    # Drop temporary column
    df.drop(columns=['__asset_class__', '__asset_subclass__'], inplace=True)

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

    #bb.DBG('_correlation: corr({},{}) = {}'.format(a, b, corr))
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

def _get_metric_lists(df, portfolio_option):
    ''' creates lists for investment option, std_dev, asset_class, etc...
        for each investment option for the specified portfolio.
        returns a cache copy unless refreshset is True
    '''

    # check for valid investment options
    available_inv_opts = list(df['Investment Option'])
    bb.DBG('available_inv_opts', available_inv_opts)
    bb.DBG('portfolio_option', portfolio_option)
    for key in portfolio_option.keys():
        if key not in available_inv_opts:
            raise Exception("Error: Portfolio option '{}' not in {}!!!"
                            .format(key, __m.investment_universe))
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
        __m.vola_column].values[0] for inv_opt in ml.inv_opts]
    ml.ds_volas = \
        [df.loc[df['Investment Option'] == inv_opt,
        __m.ds_vola_column].values[0] for inv_opt in ml.inv_opts]
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

#####################################################################
# ANALYZE

def _calc_weights(df, asset_dict, weight_by):
    """ calculate weights for assets in asset_dict using weight_by method """

    weight_by_choices = ('Equal', 'Sharpe Ratio', 'Annual Returns',
                         'Std Dev', 'Vola', 'DS Vola')
    assert weight_by in weight_by_choices, \
        "Invalid weight_by '{}'".format(weight_by)

    ml = _get_metric_lists(df, asset_dict)
    bb.DBG('_calc_weights: asset_dict = {}'.format(asset_dict))
    bb.DBG('_calc_weights: asset_dict_ml = {}'.format(ml))

    if weight_by == 'Equal':
        n = len(asset_dict)
        weights = [1/n] * n
        asset_dict.update(zip(asset_dict, weights))

    elif weight_by in ('Sharpe Ratio', 'Annual Returns'):
        if weight_by == 'Sharpe Ratio':  metric = ml.sharpe_ratios
        else:                            metric = ml.annual_returns
        # investment options with negative metrics will have weight=0
        metric = [m if m >= 0 else 0 for m in metric]
        metric_sum = sum(metric)
        if not math.isclose(metric_sum, 0):
            weights = [m/metric_sum for m in metric]
        else:
            print('ZeroMetricWarning: All investment options within this group'
                  ' have zero {} metric.  Defaulting to Equal Weighting for {}'
                  .format(weight_by, asset_dict))
            n = len(asset_dict)
            weights = [1/n] * n
        asset_dict.update(zip(asset_dict, weights))

    elif weight_by in ('Std Dev', 'Vola', 'DS Vola'):
        if weight_by == 'Std Dev':  metric = ml.std_devs
        elif weight_by == 'Vola':   metric = ml.volas
        else:                       metric = ml.ds_volas
        inverse_metric = [1/0.001 if math.isclose(m, 0) else 1/m \
                          for m in metric]
        inverse_metric_sum = sum(inverse_metric)
        weights = [m/inverse_metric_sum for m in inverse_metric]
        asset_dict.update(zip(asset_dict, weights))

    else:
        raise Exception('Error: Invalid weight_by {}'.format(weight_by))

def _get_cmpt_weights(df, d, user_weights, user_weight_by):
    """ calculate the weights not specified by user, we need to compute them """

    w = user_weights.copy()
    d = {k : 0 for k in set(d) - set(user_weights)}
    if (d):
        _calc_weights(df, d, user_weight_by)
        multi = 1 - sum(user_weights.values())
        if multi < 0: multi = 0
        for key in d: d[key] *= multi
        w.update(d)
    return w


def _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user):
    """ calculate portfolio option weights using asset class. asset subclass,
        and inv_opt weights
    """
    for i, inv_opt in enumerate(ml.inv_opts):
        asset_class = ml.asset_classes[i].split(':')[0]
        asset_subclass = ml.asset_classes[i]
        asset_class_weight = 1 if user.asset_class_weight_by is None \
            else cmpt.asset_class_weights[asset_class]
        asset_subclass_weight = 1 if user.asset_subclass_weight_by is None \
            else cmpt.asset_subclass_weights[asset_subclass]
        weight = (asset_class_weight *
                  asset_subclass_weight *
                  cmpt.inv_opt_weights[inv_opt])
        portfolio_option[inv_opt] = weight

def _check_allocation(weights, asset_class_name):
    """ make sure total adds to 100 """
    s = sum(weights.values())
    if not math.isclose(s, 1, rel_tol=1e-09, abs_tol=0.0):
        raise Exception('Error: {} allocation of \'{}\' is not 100%!!!'
                        .format(asset_class_name, s))

def _assign_weights(df, ml, portfolio_option, weight_by):

    ''' Specify the weighting scheme.  It will replace the weights specified
        in the portfolio.  You can also fix the weights on some
        Investent Options, Asset Classes, and Asset Subclasses while the others
        are automatically calculated.

        'Equal' - will use equal weights.

        'Sharpe Ratio' - will use proportionally weighted allocations based on
        the percent of an investment option's sharpe ratio to the sum of all
        the sharpe ratios in the portfolio.

        'Std Dev' - will use standard deviation adjusted weights

        'Annual Returns' - will use return adjusted weights

        'Vola' - will use volatility adjusted weights

        'DS Vola' - will use downside volatility adjusted weights

        None:   'Investment Option' means use use specified weights
                'Asset Class' means do not group by Asset Class
                'Asset Subclass means do not group by Asset Subclass 
    '''

    # weight by user specified portfolio weights ###############################
    if weight_by is None:
        return

    # unpack weight_by dictionary
    asset_class_weight_by = asset_subclass_weight_by = inv_opt_weight_by = None

    asset_class_weights     = weight_by.get('Asset Class')
    asset_subclass_weights  = weight_by.get('Asset Subclass')
    inv_opt_weights         = weight_by.get('Investment Option')

    if asset_class_weights:
       asset_class_weight_by = asset_class_weights.pop('weight_by', None)
    if asset_subclass_weights:
        asset_subclass_weight_by = asset_subclass_weights.pop('weight_by', None)
    if inv_opt_weights:
        inv_opt_weight_by = inv_opt_weights.pop('weight_by', None)

    # user dict is the user_specified weights; cpmt is the computed weights
    user = dotdict()
    cmpt = dotdict()
    # user initialization
    user.asset_class_weights = asset_class_weights
    user.asset_subclass_weights = asset_subclass_weights
    user.inv_opt_weights = inv_opt_weights
    user.asset_class_weight_by = asset_class_weight_by
    user.asset_subclass_weight_by = asset_subclass_weight_by
    user.inv_opt_weight_by = inv_opt_weight_by
    # cmpt initialization
    cmpt.asset_class_weights = {key.split(':')[0] : 0 for key in ml.asset_classes}
    cmpt.asset_subclass_weights = {key : 0 for key in ml.asset_classes}
    cmpt.inv_opt_weights = {key : 0 for key in ml.inv_opts}

    # handle invalid weight_by combinations ####################################
    msg = ('WeightByWarning: A value is set on Asset Class weight_by or'
           ' Asset Subclass weight_by, even though Investment Option weight_by'
           ' is None.  These setting are disabled when Investment Option'
           ' weight_by is None')
    if (user.inv_opt_weight_by is None and
       (user.asset_class_weight_by or user.asset_subclass_weight_by)):
        print(msg)
        return;

    # weight by user specified portfolio weights ###############################
    if user.inv_opt_weight_by is None:
        return

    # weight_by inv_opts only ##################################################
    if (user.inv_opt_weight_by and 
        user.asset_class_weight_by is None and
        user.asset_subclass_weight_by is None):

        bb.DBG(user.inv_opt_weights, user.inv_opt_weight_by)
        # use the weights in the dictionary, then the weight_by method for the
        # remaining inv_opts
        assert(set(user.inv_opt_weights).issubset(set(cmpt.inv_opt_weights))), \
               "Invalid Investment Option in weight_by!"
        d = cmpt.inv_opt_weights
        w = _get_cmpt_weights(df, d, user.inv_opt_weights, user.inv_opt_weight_by)
        _check_allocation(w, 'Investment Option')
        cmpt.inv_opt_weights.update(w)
        bb.DBG('cmpt.inv_opt_weights', cmpt.inv_opt_weights)

        _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user)
        bb.DBG('portfolio_option', portfolio_option)
        return

    # weight by all ############################################################
    if (user.inv_opt_weight_by and
        user.asset_class_weight_by and
        user.asset_subclass_weight_by):

        bb.DBG(user.inv_opt_weights, user.inv_opt_weight_by)
        bb.DBG(user.asset_class_weights, user.asset_class_weight_by)
        bb.DBG(user.asset_subclass_weights, user.asset_subclass_weight_by)

        # compute asset class weights within portfolio
        assert(set(user.asset_class_weights).issubset(set(cmpt.asset_class_weights))), \
               "Invalid Asset Class in weight_by!"
        d = cmpt.asset_class_weights
        w = _get_cmpt_weights(__m.asset_class_table, d, user.asset_class_weights,
                              user.asset_class_weight_by)
        _check_allocation(w, 'Asset Class')
        cmpt.asset_class_weights.update(w)
        bb.DBG('cmpt.asset_class_weights', cmpt.asset_class_weights)

         # compute asset subclass weights within each asset class
        assert(set(user.asset_subclass_weights).issubset(set(user.asset_subclass_weights))), \
               "Invalid Asset Sublass in weight_by!"
        for asset_class in cmpt.asset_class_weights.copy():
            # d: get asset subclasses for this asset_class
            d = {k: v for k, v in cmpt.asset_subclass_weights.items() if k.startswith(asset_class)}
            # i: get the intersection of d and user specified asset_subclasses
            i = d.keys() & user.asset_subclass_weights.keys()
            user._asset_subclass_weights = {k: user.asset_subclass_weights[k] for k in i}
            w = _get_cmpt_weights(__m.asset_class_table, d, user._asset_subclass_weights,
                                  user.asset_subclass_weight_by)
            _check_allocation(w, 'Asset Sublass')
            cmpt.asset_subclass_weights.update(w)
        bb.DBG('cmpt.asset_subclass_weights', cmpt.asset_subclass_weights)

        # compute investment option weights within each asset subclass
        assert(set(user.inv_opt_weights).issubset(set(user.inv_opt_weights))), \
               "Invalid Investment Option in weight_by!"
        for asset_subclass in cmpt.asset_subclass_weights.copy():
            # d: get investment options for this asset_subclass
            d = {k: v for i, (k, v) in enumerate(cmpt.inv_opt_weights.items()) \
                      if ml.asset_classes[i] == asset_subclass}
            # i: get the intersection of d and user specified inv_opts
            i = d.keys() & user.inv_opt_weights.keys()
            user._inv_opt_weights = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user._inv_opt_weights, user.inv_opt_weight_by)
            _check_allocation(w, 'Investment Option')
            cmpt.inv_opt_weights.update(w)
        bb.DBG('cmpt.inv_opt_weights', cmpt.inv_opt_weights)

        _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user)
        bb.DBG('portfolio_option', portfolio_option)
        return

    # weight by inv_opt and asset_class ########################################
    if (user.inv_opt_weight_by and
        user.asset_class_weight_by and
        user.asset_subclass_weight_by is None):

        bb.DBG(user.inv_opt_weights, user.inv_opt_weight_by)
        bb.DBG(user.asset_class_weights, user.asset_class_weight_by)

        # compute asset class weights within portfolio
        assert(set(user.asset_class_weights).issubset(set(cmpt.asset_class_weights))), \
               "Invalid Asset Class in weight_by!"
        d = cmpt.asset_class_weights
        w = _get_cmpt_weights(__m.asset_class_table, d, user.asset_class_weights, user.asset_class_weight_by)
        _check_allocation(w, 'Asset Class')
        cmpt.asset_class_weights.update(w)
        bb.DBG('cmpt.asset_class_weights', cmpt.asset_class_weights)

        # compute investment option weights within each asset class
        assert(set(user.inv_opt_weights).issubset(set(user.inv_opt_weights))), \
               "Invalid Investment Option in weight_by!"
        for asset_class in cmpt.asset_class_weights.copy():
            # d: get investment options for this asset_class
            d = {k: v for i, (k, v) in enumerate(cmpt.inv_opt_weights.items()) \
                      if ml.asset_classes[i].split(':')[0] == asset_class}
            # i: get the intersection of d and user specified inv_opts
            i = d.keys() & user.inv_opt_weights.keys()
            user._inv_opt_weights = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user._inv_opt_weights, user.inv_opt_weight_by)
            _check_allocation(w, 'Investment Option')
            cmpt.inv_opt_weights.update(w)
        bb.DBG('cmpt.inv_opt_weights', cmpt.inv_opt_weights)

        _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user)
        bb.DBG('portfolio_option', portfolio_option)
        return

    # weight by inv_opt and asset_subclass #####################################
    if (user.inv_opt_weight_by and
        user.asset_class_weight_by is None and
        user.asset_subclass_weight_by):

        bb.DBG(user.inv_opt_weights, user.inv_opt_weight_by)
        bb.DBG(user.asset_subclass_weights, user.asset_subclass_weight_by)

        # compute asset subclass weights within portfolio
        assert(set(user.asset_subclass_weights).issubset(set(cmpt.asset_subclass_weights))), \
               "Invalid Asset SubClass in weight_by!"
        d = cmpt.asset_subclass_weights
        w = _get_cmpt_weights(__m.asset_class_table, d, user.asset_subclass_weights,
                              user.asset_subclass_weight_by)
        _check_allocation(w, 'Asset SubClass')
        cmpt.asset_subclass_weights.update(w)
        bb.DBG('cmpt.asset_subclass_weights', cmpt.asset_subclass_weights)

        # compute investment option weights within each asset subclass
        assert(set(user.inv_opt_weights).issubset(set(user.inv_opt_weights))), \
               "Invalid Investment Option in weight_by!"
        for asset_subclass in cmpt.asset_subclass_weights.copy():
            # d: get investment options for this asset_subclass
            d = {k: v for i, (k, v) in enumerate(cmpt.inv_opt_weights.items()) \
                      if ml.asset_classes[i] == asset_subclass}
            # i: get the intersection of d and user specified inv_opts
            i = d.keys() & user.inv_opt_weights.keys()
            user._inv_opt_weights = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user._inv_opt_weights, user.inv_opt_weight_by)
            _check_allocation(w, 'Investment Option')
            cmpt.inv_opt_weights.update(w)
        bb.DBG('cmpt.inv_opt_weights', cmpt.inv_opt_weights)

        _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user)
        bb.DBG('portfolio_option', portfolio_option)
        return

def print_portfolio(portfolio_option):
    print('{} Weights:'.format(__m.portfolio_title))
    for k, v in portfolio_option.items():
        print('    {:30} {:0.4f}'.format(k, v))

def analyze(df, portfolio_option, weight_by=None, default_correlation=1):
    ''' analyze portfolio_option and return the annual_ret, std_dev, sharpe_ratio
        portfolio_option may also be modified if weight_by is not None
    
        default_correlation - correlation to use when no correlation has been
        specified between two asset classes.  If you use only the Asset Classes
        defined in universe/asset-classes.csv, then this will never happen.
    '''

    # pop the title
    __m.portfolio_title = portfolio_option.pop('Title', __m.portfolio_title)

    # set default correlation
    __m.default_correlation = default_correlation
    
    # get metric_lists
    ml = _get_metric_lists(df, portfolio_option)
    
    bb.DBG('ml = {}'.format(ml))

    # assign weights
    _assign_weights(df, ml, portfolio_option, weight_by)

    # make sure total adds to 100
    _check_allocation(portfolio_option, __m.portfolio_title)

    # compute metrics
    annual_return = _expected_return(ml.annual_returns, ml.weights)
    std_dev = _standard_deviation(ml.weights, ml.std_devs, ml.asset_classes)
    sharpe_ratio = _sharpe_ratio(annual_return, std_dev, __m.risk_free_rate)

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

    ml = _get_metric_lists(df, portfolio_option)

    metrics = ['Annual Returns', 'Std Dev', 'Sharpe Ratio']
    index = []
    columns = [inv_opt for inv_opt in ml.inv_opts]
    data = []

    # add metrics
    for metric in metrics:
        index.append(metric)
        data.append([df.loc[df['Investment Option'] == inv_opt, metric].values[0]
                    for inv_opt in ml.inv_opts])

    # add weight
    index.append('Weight')
    data.append([portfolio_option[inv_opt] for inv_opt in ml.inv_opts])

    # Worst Typical Down Year
    index.append('Worst Typical Down Year')
    data.append([df.loc[df['Investment Option'] == inv_opt, 'Annual Returns'].values[0] +
              -2*df.loc[df['Investment Option'] == inv_opt, 'Std Dev'].values[0]
                for inv_opt in ml.inv_opts])

    # add Black Swan
    index.append('Black Swan')
    data.append([df.loc[df['Investment Option'] == inv_opt, 'Annual Returns'].values[0] +
              -3*df.loc[df['Investment Option'] == inv_opt, 'Std Dev'].values[0]
                for inv_opt in ml.inv_opts])

    # create dataframe
    summary = pd.DataFrame(data, columns=columns, index=index)

    # set portfolio values
    summary[__m.portfolio_title] = \
        [annual_ret, std_dev, sharpe_ratio, sum(ml.weights),
         annual_ret + -2*std_dev, annual_ret + -3*std_dev]

    # plot returns
    column_names = columns.copy()
    column_names.append(__m.portfolio_title)
    _plot_returns(summary, column_names)

    return summary

#####################################################################
# SHOW PIE CHARTS

def _add_weight_column(row, portfolio_option):
    inv_opt = annual_ret = row['Investment Option']
    return portfolio_option[inv_opt]

def _show_pie_chart(df, portfolio_option, chart):

    title = '{} - by {}'.format(__m.portfolio_title, chart)

    if chart == 'Investment Option':
        weights = df['Weight']
        labels = df['Investment Option']
        plt.title(title)
    else:
        asset_classes = df['Asset Class']
        asset_classes = list(asset_classes)
        if chart == 'Asset Class':
            asset_classes = [asset_class.split(':')[0] for asset_class in asset_classes]
        else:
            asset_classes = [asset_class for asset_class in asset_classes]
        asset_classes = list(set(asset_classes))
        asset_classes.sort()
        weights = []
        labels = asset_classes
        for asset_class in asset_classes:
            if chart == 'Asset Class':
                weight = df.loc[df['Asset Class'].str.startswith(asset_class), 'Weight'].sum()
            else:
                weight = df.loc[df['Asset Class'].eq(asset_class), 'Weight'].sum()
            weights.append(weight)

    plt.pie(weights, labels=labels, counterclock=False, startangle=90,
            autopct='%1.1f%%', normalize=True)
    plt.title(title)
    plt.axis('equal')
    plt.show()

    s = pd.Series()
    for i, label in enumerate(labels):
        s[label] = weights[i]
    s = str(s).split('dtype:')[0]
    return s

def show_pie_charts(df, portfolio_option,
                    charts=['Investment Option', 'Asset Class', 'Asset Subclass']):

    # if user specified a single chart, put it in a list
    if not isinstance(charts, list):  charts = [charts]
    chart_choices = ('All', 'Investment Option', 'Asset Class', 'Asset Subclass')
    assert(set(charts).issubset(set(chart_choices))), \
           "Invalid Chart type in charts!"

    df = df.copy()
    df = df[df['Investment Option'].isin(list(portfolio_option))]
    df.reset_index(drop=True, inplace=True)
    df['Weight'] = df.apply(_add_weight_column, portfolio_option=portfolio_option, axis=1)
    df.sort_values('Asset Class', inplace=True)

    for chart in charts:
        s = _show_pie_chart(df, portfolio_option, chart)
        print(s)

#####################################################################
# OPTIMIZER

def _sim(ml, min_annual_return, max_worst_typical_down_year, max_black_swan):
    ''' Calculation is done via a Monte Carlo Simulation by trying random
        combinations of weights and checking which combination has the best
        sharpe_ratio
    '''

    o_sharpe_ratio = 0
    o_weights = []

    N = 100000
    i = 0
    while i < N:
        rands = [round(random.random(), 2) for x in range(0, len(ml.annual_returns))]
        s = sum(rands)
        if math.isclose(s, 0): continue
        rands = [x/s for x in rands]

        annual_return = _expected_return(ml.annual_returns, weights=rands)
        std_dev = _standard_deviation(weights=rands, std_devs=ml.std_devs,
                                      asset_classes=ml.asset_classes)
        sharpe_ratio = _sharpe_ratio(annual_return, std_dev, __m.risk_free_rate)
        worst_typical_down_year = annual_return - 2*std_dev
        black_swan = annual_return - 3*std_dev

        if (annual_return < min_annual_return or
            worst_typical_down_year < max_worst_typical_down_year or
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

def optimizer(df, portfolio_option, constraints=None):
    ''' Optimize sharpe ratio while specifying Annual Rate,
        Worst Typical Down Year, and Black Swan.  Setting a constraint to None
        optimizes absolute Sharpe Ratio without regard to that constraint.
    '''

    # unpack contraints dict
    min_annual_return = max_worst_typical_down_year = max_black_swan = None
    if constraints:
        min_annual_return           = constraints.get('Annual Return')
        max_worst_typical_down_year = constraints.get('Worst Typical Down Year')
        max_black_swan              = constraints.get('Black Swan')

    if min_annual_return is None:           min_annual_return = 0
    if max_worst_typical_down_year is None: max_worst_typical_down_year = -1000
    if max_black_swan is None:              max_black_swan = -1000

    ml = _get_metric_lists(df, portfolio_option)

    print('Running optimizer', end='')
    success, optimal_weights = \
        _sim(ml, min_annual_return, max_worst_typical_down_year, max_black_swan)
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
    annual_return = _expected_return(annual_returns=ml.annual_returns,
                                     weights=optimal_weights)
    std_dev = _standard_deviation(weights=optimal_weights, std_devs=ml.std_devs,
                                  asset_classes=ml.asset_classes)
    sharpe_ratio = _sharpe_ratio(annual_return, std_dev, __m.risk_free_rate)
    worst_typical_down_year = annual_return - 2*std_dev
    black_swan = annual_return - 3*std_dev
    _portfolio_option.update(zip(_portfolio_option, optimal_weights))

    '''
    print('{} Metrics:'.format(__m.portfolio_title))
    print('    max_sharpe_ratio         {:>0.2f}'.format(sharpe_ratio))
    print('    annual_return            {:>0.2f}'.format(annual_return))
    print('    std_dev                  {:>0.2f}'.format(std_dev))
    print('    worst typical down year  {:>0.2f}'.format(worst_typical_down_year))
    print('    black_swan               {:>0.2f}'.format(black_swan))
    print()
    '''

    s = pd.Series()
    s[__m.portfolio_title + ' Metrics:'] = ''
    s['    max_sharpe_ratio'] = sharpe_ratio
    s['    annual_return'] = annual_return
    s['    std_dev'] = std_dev
    s['    worst typical down year'] = worst_typical_down_year
    s['    black_swan'] = black_swan
    s = str(s).split('dtype:')[0]
    print(s)
    print()
    print_portfolio(_portfolio_option)
    print()

    # check to see if one of the inv_options is a risk_free asset
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

