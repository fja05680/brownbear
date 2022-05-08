"""
Portfolio analysis and optimization.
"""

import itertools
import math
import random

import matplotlib.pyplot as plt
import numpy
import pandas as pd

import brownbear as bb
from brownbear.utility import dotdict


########################################################################
# PORTFOLIO

class Portfolio:

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
class : Single instance of Portfolio object to be used glabally.
"""

########################################################################
# FETCH

def _correlation_table_to_dict():
    """
    Return a dictionary of the correlation_table.
    """
    df = PORT.correlation_table.set_index(['Asset Class A', 'Asset Class B'])
    # Make any na values perfectly correlated=1; convert to float.
    df['Correlation'] = df['Correlation'].fillna(1)
    df['Correlation'] = df['Correlation'].astype(float)
    d = df['Correlation'].to_dict()
    return d


def _sharpe_ratio(annual_ret, std_dev, risk_free_rate):
    """
    Return the sharpe ratio.

    This is the modified sharpe ratio formulated by Craig L. Israelsen.
    It's the same as the sharpe ration when the excess return is
    positive.
    """
    if math.isclose(std_dev, 0):
        return 0
    excess_return = annual_ret - risk_free_rate
    divisor = std_dev if excess_return > 0 else 1/std_dev
    return excess_return / divisor


def _add_sharpe_ratio_column(inv_opts, risk_free_rate):
    """
    Add Sharpe Ratio column to the inv_opts dataframe.
    """
    def _sharpe(row, risk_free_rate):
        annual_ret = row['Annual Returns']
        std_dev = row['Std Dev']
        return _sharpe_ratio(annual_ret, std_dev, risk_free_rate)

    inv_opts['Sharpe Ratio'] = inv_opts.apply(_sharpe, risk_free_rate=risk_free_rate, axis=1)
    return inv_opts


def fetch(investment_universe, risk_free_rate=0, annual_returns='Annual Returns',
          vola='Std Dev', ds_vola='Std Dev', clean=True):
    """
    Fetch Investment Universe and asset classes
    
    investment-options.csv format:
        "Investment Option", "Description"(optimal), "Asset Class",
        "Annual Returns", "Std Dev"

    asset-classes.csv format:
        "Asset Class A", "Asset Class B","Correlation"
        "Description" field is optional.  It is not referenced in code.
        "Annual Returns" column(s) can named anything.
            Recommend "1 Yr", "3 Yr", "5 Yr", or "10 Yr".  Then
            annual_returns parameter can select the column to use.

    Parameters
    ----------
    investment_universe : list of str
        List of investment galaxies.  These are the dirs within
        universe/, for example ['dow30-galaxy', 'alabama-galaxy'].
    risk_free_rate : float, optional
        Risk free rate (default is 0).
    annual_returns : str, optional
        Specifies which column to use for annualized returns
        (default is 'Annual Returns').
    vola : str, optional
        Specifies which column to use for volatility
        (default is 'Std Dev').
    ds_vola : str, optional
        Specifies which column to use for downside volatility
        (default is 'Std Dev').
    clean : bool, optional
        True to remove rows that have a 'nan' as a column value
        (default is True).
    
    Returns
    -------
    inv_opts : pd.DataFrame
        Dataframe of investment options with columns for asset class,
        description, and performace metrics.
    """
    # If caller specified a single filename, put it in a list.
    if not isinstance(investment_universe, list):
        investment_universe = [investment_universe]

    # Create the investment options csv file list, then read into
    # a dataframe.  There are 2 places to look for
    # investment-options.csv files: under universe/ and portfolios/.
    filepaths = []
    for galaxy in investment_universe:
        for subdir in ['universe', 'portfolios']:
            filepath = bb.ROOT / subdir / galaxy / 'investment-options.csv'
            if filepath.is_file():
                filepaths.append(filepath)

    inv_opts = bb.cvs_to_df(filepaths)

    # Drop duplicate Investment Option's, keep the first,
    # then reset index.
    inv_opts.drop_duplicates(subset=['Investment Option'], keep='first', inplace=True)
    inv_opts.reset_index(drop=True, inplace=True)

    # Allows the use of different annualized returns,
    # e.g. 1, 3, or 5 year annaulized returns.
    if annual_returns != 'Annual Returns':
        inv_opts['Annual Returns'] = inv_opts[annual_returns]

    # Add Sharpe Ratio column.
    inv_opts = _add_sharpe_ratio_column(inv_opts, risk_free_rate)

    # Asset class table.
    PORT.asset_class_table = bb.cvs_to_df(
        [bb.ROOT / 'universe' / 'asset-class-galaxy' / 'investment-options.csv'])

    # Add Annual Returns column to asset class table.
    if annual_returns in PORT.asset_class_table.columns:
        PORT.asset_class_table['Annual Returns'] = PORT.asset_class_table[annual_returns]
    else:
        PORT.asset_class_table['Annual Returns'] = PORT.asset_class_table['5 Yr']

    # Add Sharpe Ratio column to asset class table.
    PORT.asset_class_table = _add_sharpe_ratio_column(PORT.asset_class_table, risk_free_rate)

    # Correlation table.
    PORT.correlation_table = bb.cvs_to_df(
        [bb.ROOT / 'universe' / 'asset-class-galaxy' / 'asset-classes.csv'])

    # Convert correlation table to dict for easier faster processing.
    PORT.correlation_table = _correlation_table_to_dict()

    # Set other module variables.
    PORT.investment_universe = investment_universe.copy()
    PORT.risk_free_rate = risk_free_rate
    PORT.vola_column = vola
    PORT.ds_vola_column = ds_vola

    if clean:
        # Remove any rows that have nan for column values.
        inv_opts = inv_opts.dropna()
        inv_opts.reset_index(drop=True, inplace=True)

    return inv_opts


########################################################################
# FUNDAMENTALS

def add_fundamental_columns(inv_opts, clean=True):
    """
    Add fundamental data columns to inv_opts dataframe.

    Columns added:
      'Previous Close',
      'Trailing PE',
      'Dividend Yield',
      'Market Cap'

    Parameters
    ----------
    inv_opts : pd.DataFrame
        Dataframe of investment options with columns for asset class,
        description, and performace metrics.
    clean : bool, optional
        True to remove rows that have a 'nan' as a column value
        (default is True).

    Returns
    -------
    inv_opts : pd.DataFrame
        Dataframe of investment options with fundamental data columns.
    """
    filepath = bb.ROOT / 'tools' / 'symbol-cache' / 'fundamentals.csv'
    df = pd.read_csv(filepath)
    df.rename(columns={'symbol': 'Investment Option',
                        'previousClose': 'Previous Close',
                        'trailingPE': 'Trailing PE',
                        'dividendYield': 'Dividend Yield',
                        'marketCap': 'Market Cap'}, inplace=True)
    inv_opts = inv_opts.merge(df, how='left')
    if clean:
        # Remove any rows that have nan for column values.
        inv_opts = inv_opts.dropna()
        inv_opts.reset_index(drop=True, inplace=True)
    return inv_opts


########################################################################
# RANK

def rank(inv_opts, rank_by, group_by=None, num_per_group=None, ascending=False):
    """
    Rank investment options.

    Parameters
    ----------
    inv_opts : pd.DataFrame
        Dataframe of investment options with columns for asset class,
        description, and performace metrics.
    rank_by : str
        The performance or fundamental metric used to sort the
        investment options.
    group_by : str, optional {None, 'Asset Class', 'Asset Subclass'}
        How to group investment options (default is None, which imples
        no grouping)
    num_per_group : int, optional
        The number of investment options for each group
        (default is None, which imples 5 if group_by is specified, 
         otherwise 1000).
    ascending : bool, optional
        True to sort in ascending order (default is False, which imples
        sorting in descending order).

    Returns
    -------
    df : pd.DataFrame
        Dataframe of investment options with ranking.
    """
    group_by_choices = (None, 'Asset Class', 'Asset Subclass')
    assert group_by in group_by_choices, "Invalid group_by f'{group_by}'"

    df = inv_opts.copy()

    # Temporarily add __asset_class__ and  __asset_subclass__ for
    # convenience; drop it later.S
    df['__asset_subclass__'] = df['Asset Class']

    def _add_asset_class(row):
        # Extract the class_name from '__asset_subclass__' column.
        class_name = row['__asset_subclass__']
        class_name = class_name.split(':')[0]
        return class_name

    # Add '__asset_class__' column.
    df['__asset_class__'] = df.apply(_add_asset_class, axis=1)

    # Sort.
    if group_by is None:
        if num_per_group is None:
            num_per_group = 10000
        df = df.sort_values(rank_by, ascending=ascending) \
                            .head(num_per_group)
    elif group_by == 'Asset Class':
        if num_per_group is None:
            num_per_group = 5
        df = df.sort_values(['__asset_class__', rank_by], ascending=ascending) \
                            .groupby('__asset_class__').head(num_per_group)
    elif group_by == 'Asset Subclass':
        if num_per_group is None:
            num_per_group = 5
        df = df.sort_values(['__asset_subclass__', rank_by], ascending=ascending) \
                            .groupby('__asset_subclass__').head(num_per_group)
    else:
        raise Exception(f"Error: Invalid value for groupby: '{group_by}'")

    # Drop temporary column.
    df.drop(columns=['__asset_class__', '__asset_subclass__'], inplace=True)

    return df


########################################################################
# METRIC FUNCTIONS

def _expected_return(annual_returns, weights):
    """
    Returns expected return given list of investment option returns and
    their corresponding weights.
    """
    return sum(numpy.multiply(annual_returns, weights))


def _correlation(correlation_table, a, b):
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


def _standard_deviation(weights, std_devs, asset_classes):
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
            corr = _correlation(PORT.correlation_table, a[1], b[1])
            a_i = a[0]
            b_i = b[0]
            variance += 2*weights[a_i]*weights[b_i]*std_devs[a_i]*std_devs[b_i]*corr
    std_dev = numpy.sqrt(variance)
    return std_dev


def _get_metric_lists(df, portfolio_option):
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


########################################################################
# ASSIGN WEIGHTS

def _calc_weights(df, asset_dict, weight_by):
    """
    Calculate weights for assets in asset_dict using weight_by method.
    """
    weight_by_choices = ('Equal', 'Sharpe Ratio', 'Annual Returns',
                         'Std Dev', 'Vola', 'DS Vola')
    assert weight_by in weight_by_choices, f"Invalid weight_by '{weight_by}'"

    ml = _get_metric_lists(df, asset_dict)
    bb.DBG(f'asset_dict = {asset_dict}')
    bb.DBG(f'asset_dict_ml = {ml}')

    if weight_by == 'Equal':
        n = len(asset_dict)
        weights = [1/n] * n
        asset_dict.update(zip(asset_dict, weights))

    elif weight_by in ('Sharpe Ratio', 'Annual Returns'):
        # If there are any negative returns, apply unity-based normalization.
        # if a return is negative, then sharpe_ratio will also be negative.
        numpy.seterr('raise')
        xmin = min(ml.annual_returns)
        if xmin < 0:
            a = 1; b = 10
            if len(ml.annual_returns) == 1:
                ml.annual_returns[0] = ml.sharpe_ratios[0] = a
            else:
                # Z = a + (x − xmin)*(b − a) (xmax − xmin)
                xmax = max(ml.annual_returns)
                z = [a + (x-xmin)*(b-a)/(xmax-xmin) for x in ml.annual_returns]
                ml.annual_returns = z
                # Recalculate sharpe_ratios besed on normalized annual_returns.
                ml.sharpe_ratios = [_sharpe_ratio(annual_ret, std_dev, PORT.risk_free_rate)
                    for annual_ret, std_dev in zip(ml.annual_returns, ml.std_devs)]

        if weight_by == 'Sharpe Ratio':
            metric = ml.sharpe_ratios
        else:
            metric = ml.annual_returns

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
        raise Exception(f'Error: Invalid weight_by {weight_by}')


def _get_cmpt_weights(df, d, user_weights, user_weight_by):
    """
    Calculate the weights not specified by user.  We need to compute
    them.
    """
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
    """
    Calculate portfolio option weights using asset class,
    asset subclass, and inv_opt weights
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
    """
    Make sure total adds to 100.
    """
    s = sum(weights.values())
    if not math.isclose(s, 1, rel_tol=1e-09, abs_tol=0.0):
        raise Exception(f"Error: {asset_class_name} allocation of '{s}' is not 100%!!!")

def _assign_weights(df, ml, portfolio_option, weight_by):

    """ 
    Specify the weighting scheme.  It will replace the weights specified
    in the portfolio.  You can also fix the weights on some
    Investent Options, Asset Classes, and Asset Subclasses while the
    others are automatically calculated.

        'Equal' - will use equal weights.

        'Sharpe Ratio' - will use proportionally weighted allocations
        based on the percent of an investment option's sharpe ratio to
        the sum of all the sharpe ratios in the portfolio.

        'Std Dev' - will use standard deviation adjusted weights.

        'Annual Returns' - will use return adjusted weights.

        'Vola' - will use volatility adjusted weights.

        'DS Vola' - will use downside volatility adjusted weights.

        None:   'Investment Option' means use use specified weights.
                'Asset Class' means do not group by Asset Class.
                'Asset Subclass means do not group by Asset Subclass.
    """

    # `weight_by` user specified portfolio weights #####################
    if weight_by is None:
        return

    # Unpack `weight_by` dictionary.
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

    # `user` dict is the user_specified weights.
    # `cpmt` is the computed weights.
    user = dotdict()
    cmpt = dotdict()

    # `user` initialization.
    user.asset_class_weights = asset_class_weights
    user.asset_subclass_weights = asset_subclass_weights
    user.inv_opt_weights = inv_opt_weights
    user.asset_class_weight_by = asset_class_weight_by
    user.asset_subclass_weight_by = asset_subclass_weight_by
    user.inv_opt_weight_by = inv_opt_weight_by

    # `cmpt` initialization.
    cmpt.asset_class_weights = {key.split(':')[0] : 0 for key in ml.asset_classes}
    cmpt.asset_subclass_weights = {key : 0 for key in ml.asset_classes}
    cmpt.inv_opt_weights = {key : 0 for key in ml.inv_opts}

    # Handle invalid weight_by combinations.
    msg = ( 'WeightByWarning: A value is set on Asset Class weight_by or'
            ' Asset Subclass weight_by, even though Investment Option weight_by'
            ' is None.  These setting are disabled when Investment Option'
            ' weight_by is None')

    if (user.inv_opt_weight_by is None and
       (user.asset_class_weight_by or user.asset_subclass_weight_by)):
        print(msg)
        return

    # `weight_by` user specified portfolio weights.
    if user.inv_opt_weight_by is None:
        return

    # `weight_by` inv_opts only.
    if (user.inv_opt_weight_by and 
        user.asset_class_weight_by is None and
        user.asset_subclass_weight_by is None):

        bb.DBG(user.inv_opt_weights, user.inv_opt_weight_by)

        # Use the weights in the dictionary, then the `weight_by` method
        # for the remaining `inv_opts`.
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

    # `weight_by` all.
    if (user.inv_opt_weight_by and
        user.asset_class_weight_by and
        user.asset_subclass_weight_by):

        bb.DBG(user.inv_opt_weights, user.inv_opt_weight_by)
        bb.DBG(user.asset_class_weights, user.asset_class_weight_by)
        bb.DBG(user.asset_subclass_weights, user.asset_subclass_weight_by)

        # Compute asset class weights within portfolio.
        assert(set(user.asset_class_weights).issubset(set(cmpt.asset_class_weights))), \
               "Invalid Asset Class in weight_by!"
        d = cmpt.asset_class_weights
        w = _get_cmpt_weights(PORT.asset_class_table, d, user.asset_class_weights,
                              user.asset_class_weight_by)
        _check_allocation(w, 'Asset Class')
        cmpt.asset_class_weights.update(w)
        bb.DBG('cmpt.asset_class_weights', cmpt.asset_class_weights)

         # Compute asset subclass weights within each asset class.
        assert(set(user.asset_subclass_weights).issubset(set(cmpt.asset_subclass_weights))), \
               "Invalid Asset Sublass in weight_by!"
        for asset_class in cmpt.asset_class_weights.copy():
            # d: get asset subclasses for this asset_class.
            d = {k: v for k, v in cmpt.asset_subclass_weights.items() if k.startswith(asset_class)}
            # i: get the intersection of d and user specified asset_subclasses.
            i = d.keys() & user.asset_subclass_weights.keys()
            user._asset_subclass_weights = {k: user.asset_subclass_weights[k] for k in i}
            w = _get_cmpt_weights(PORT.asset_class_table, d, user._asset_subclass_weights,
                                  user.asset_subclass_weight_by)
            _check_allocation(w, 'Asset Sublass')
            cmpt.asset_subclass_weights.update(w)
        bb.DBG('cmpt.asset_subclass_weights', cmpt.asset_subclass_weights)

        # Compute investment option weights within each asset subclass.
        assert(set(user.inv_opt_weights).issubset(set(cmpt.inv_opt_weights))), \
               "Invalid Investment Option in weight_by!"
        for asset_subclass in cmpt.asset_subclass_weights.copy():
            # d: get investment options for this asset_subclass.
            d = {k: v for i, (k, v) in enumerate(cmpt.inv_opt_weights.items()) \
                      if ml.asset_classes[i] == asset_subclass}
            # i: get the intersection of d and user specified inv_opts.
            i = d.keys() & user.inv_opt_weights.keys()
            user._inv_opt_weights = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user._inv_opt_weights, user.inv_opt_weight_by)
            _check_allocation(w, 'Investment Option')
            cmpt.inv_opt_weights.update(w)
        bb.DBG('cmpt.inv_opt_weights', cmpt.inv_opt_weights)

        _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user)
        bb.DBG('portfolio_option', portfolio_option)
        return

    # `weight_by` `inv_opt`` and asset_class.
    if (user.inv_opt_weight_by and
        user.asset_class_weight_by and
        user.asset_subclass_weight_by is None):

        bb.DBG(user.inv_opt_weights, user.inv_opt_weight_by)
        bb.DBG(user.asset_class_weights, user.asset_class_weight_by)

        # Compute asset class weights within portfolio.
        assert(set(user.asset_class_weights).issubset(set(cmpt.asset_class_weights))), \
               "Invalid Asset Class in weight_by!"
        d = cmpt.asset_class_weights
        w = _get_cmpt_weights(PORT.asset_class_table, d, user.asset_class_weights, user.asset_class_weight_by)
        _check_allocation(w, 'Asset Class')
        cmpt.asset_class_weights.update(w)
        bb.DBG('cmpt.asset_class_weights', cmpt.asset_class_weights)

        # Compute investment option weights within each asset class.
        assert(set(user.inv_opt_weights).issubset(set(cmpt.inv_opt_weights))), \
               "Invalid Investment Option in weight_by!"
        for asset_class in cmpt.asset_class_weights.copy():
            # d: get investment options for this asset_class.
            d = {k: v for i, (k, v) in enumerate(cmpt.inv_opt_weights.items()) \
                      if ml.asset_classes[i].split(':')[0] == asset_class}
            # i: get the intersection of d and user specified `inv_opts`.
            i = d.keys() & user.inv_opt_weights.keys()
            user._inv_opt_weights = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user._inv_opt_weights, user.inv_opt_weight_by)
            _check_allocation(w, 'Investment Option')
            cmpt.inv_opt_weights.update(w)
        bb.DBG('cmpt.inv_opt_weights', cmpt.inv_opt_weights)

        _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user)
        bb.DBG('portfolio_option', portfolio_option)
        return

    # `weight_by` `inv_opt` and asset_subclass.
    if (user.inv_opt_weight_by and
        user.asset_class_weight_by is None and
        user.asset_subclass_weight_by):

        bb.DBG(user.inv_opt_weights, user.inv_opt_weight_by)
        bb.DBG(user.asset_subclass_weights, user.asset_subclass_weight_by)

        # Compute asset subclass weights within portfolio.
        assert(set(user.asset_subclass_weights).issubset(set(cmpt.asset_subclass_weights))), \
               "Invalid Asset SubClass in weight_by!"
        d = cmpt.asset_subclass_weights
        w = _get_cmpt_weights(PORT.asset_class_table, d, user.asset_subclass_weights,
                              user.asset_subclass_weight_by)
        _check_allocation(w, 'Asset SubClass')
        cmpt.asset_subclass_weights.update(w)
        bb.DBG('cmpt.asset_subclass_weights', cmpt.asset_subclass_weights)

        # Compute investment option weights within each asset subclass.
        assert(set(user.inv_opt_weights).issubset(set(cmpt.inv_opt_weights))), \
               "Invalid Investment Option in weight_by!"
        for asset_subclass in cmpt.asset_subclass_weights.copy():
            # d: get investment options for this asset_subclass.
            d = {k: v for i, (k, v) in enumerate(cmpt.inv_opt_weights.items()) \
                      if ml.asset_classes[i] == asset_subclass}
            # i: get the intersection of d and user specified inv_opts.
            i = d.keys() & user.inv_opt_weights.keys()
            user._inv_opt_weights = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user._inv_opt_weights, user.inv_opt_weight_by)
            _check_allocation(w, 'Investment Option')
            cmpt.inv_opt_weights.update(w)
        bb.DBG('cmpt.inv_opt_weights', cmpt.inv_opt_weights)

        _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user)
        bb.DBG('portfolio_option', portfolio_option)
        return


#####################################################################
# ANALYZE

def analyze(df, portfolio_option, weight_by=None, default_correlation=1):
    """
    Analyze Portfolio.

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
    weight_by : dict of dicts, optional
        Specify the weighting scheme.  If not None, it will replace the
        weights specified in the portfolio.  You can also fix the
        weights on some Investent Options, Asset Classes, and Asset
        Subclasses, while the others are automatically calculated
        (default is None, which implies use the user specified weights
        specified in `portfolio_option`).

        'Equal' - use equal weights.

        'Sharpe Ratio' - use proportionally weighted allocations
        based on the percent of an investment option's sharpe ratio to
        the sum of all the sharpe ratios in the portfolio.

        'Std Dev' - use standard deviation adjusted weights.

        'Annual Returns' - use return adjusted weights.

        'Vola' - use volatility adjusted weights.

        'DS Vola' - use downside volatility adjusted weights.

        None:   'Investment Option' means use user specified weights
                'Asset Class' means do not group by Asset Class
                'Asset Subclass means do not group by Asset Subclass 

        Example:

        # At the Asset Class level, explicitly specify
        # US Stock, US Bonds, and Risk-Free Asset weights, then equally
        # allocate among any remaining asset classes.  Next, do not
        # consider the Asset Subclass within an Asset Class.  Finally,
        # weight the Investment Options within each Asset Class by
        # Annual Return.
        weight_by = {
            'Asset Class':       {'weight_by': 'Equal',
                                  'US Stocks': 0.40,
                                  'US Bonds': 0.40,
                                  'Risk-Free Asset': 0.10},
            'Asset Subclass':    {'weight_by': None},
            'Investment Option': {'weight_by': 'Annual Returns'},
        }

        default_correlation : int, optional
            Correlation to use when no correlation has been specified
            between two asset classes.  If you use only the Asset
            Classes defined in universe/asset-classes.csv, then this
            will never happen. (default is 1, which assumes that the
            assets are perfectly coorelated, i.e. worst case for
            asset diversification).

    Returns
    -------
    annual_return : float
        The expected annualized return of the portfolio.

    std_dev : float
        The standard deviation of the portfolio.

    sharpe_ratio : float
        The overall sharpe ratio of the portfolio.
    """

    # Pop the title.
    PORT.portfolio_title = portfolio_option.pop('Title', PORT.portfolio_title)

    # Set default correlation.
    PORT.default_correlation = default_correlation

    # Get metric_lists.
    ml = _get_metric_lists(df, portfolio_option)
    bb.DBG(f'ml = {ml}')

    # Assign weights.
    _assign_weights(df, ml, portfolio_option, weight_by)

    # Make sure total adds to 100 percent.
    _check_allocation(portfolio_option, PORT.portfolio_title)

    # Update metric_lists.
    ml = _get_metric_lists(df, portfolio_option)

    # Compute metrics.
    annual_return = _expected_return(ml.annual_returns, ml.weights)
    std_dev = _standard_deviation(ml.weights, ml.std_devs, ml.asset_classes)
    sharpe_ratio = _sharpe_ratio(annual_return, std_dev, PORT.risk_free_rate)

    return annual_return, std_dev, sharpe_ratio


########################################################################
# SUMMARY

def _plot_returns(summary, columns):
    """
    Bar Plot of returns with 1, 2, and 3 standard deviations.
    """
    means = list(summary.loc['Annual Returns'])
    Xs = list(range(0, len(means)))
    # Plot 1 std dev.
    maxs = list(summary.loc['Std Dev'])
    plt.errorbar(Xs, means, maxs, fmt='.k', lw=20)
    # Plot 2 std dev.
    maxs_2 = [x * 2 for x in maxs]
    plt.errorbar(Xs, means, maxs_2, fmt='.k', lw=5)
    # Plot 3 std dev.
    maxs_3 = [x * 3 for x in maxs]
    plt.errorbar(Xs, means, maxs_3, fmt='.k', lw=1)
    # Plot horizontal line for median.
    max_std_dev = max(maxs)
    maxs_median = [max_std_dev*.02 for x in means]
    plt.errorbar(Xs, means, maxs_median, fmt='.k', lw=50)
    plt.xlim(-1, len(means))
    plt.xticks(range(len(columns)), columns, rotation=60)


def summary(df, portfolio_option, annual_ret, std_dev, sharpe_ratio):
    """
    Generate summary results.

    Note: analyze() must be called before calling summary().

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
    annual_return : float
        The expected annualized return of the portfolio.
    std_dev : float
        The standard deviation of the portfolio.
    sharpe_ratio : float
        The overall sharpe ratio of the portfolio.

    Returns
    -------
    summary : pd.DataFrame
        Summary results.
    """
    ml = _get_metric_lists(df, portfolio_option)

    metrics = ['Annual Returns', 'Std Dev', 'Sharpe Ratio']
    index = []
    columns = [inv_opt for inv_opt in ml.inv_opts]
    data = []

    # Add metrics.
    for metric in metrics:
        index.append(metric)
        data.append([df.loc[df['Investment Option'] == inv_opt, metric].values[0]
                    for inv_opt in ml.inv_opts])

    # Add weight.
    index.append('Weight')
    data.append([portfolio_option[inv_opt] for inv_opt in ml.inv_opts])

    # Worst Typical Down Year.
    index.append('Worst Typical Down Year')
    data.append([df.loc[df['Investment Option'] == inv_opt, 'Annual Returns'].values[0] +
              -2*df.loc[df['Investment Option'] == inv_opt, 'Std Dev'].values[0]
                for inv_opt in ml.inv_opts])

    # Add Black Swan.
    index.append('Black Swan')
    data.append([df.loc[df['Investment Option'] == inv_opt, 'Annual Returns'].values[0] +
              -3*df.loc[df['Investment Option'] == inv_opt, 'Std Dev'].values[0]
                for inv_opt in ml.inv_opts])

    # Create dataframe.
    summary = pd.DataFrame(data, columns=columns, index=index)

    # Set portfolio values.
    summary[PORT.portfolio_title] = \
        [annual_ret, std_dev, sharpe_ratio, sum(ml.weights),
         annual_ret + -2*std_dev, annual_ret + -3*std_dev]

    # Plot returns.
    column_names = columns.copy()
    column_names.append(PORT.portfolio_title)
    _plot_returns(summary, column_names)

    return summary


def print_portfolio(portfolio_option):
    """
    Print portfolio options with their weights.
    """
    print(f'{PORT.portfolio_title} Weights:')
    for k, v in portfolio_option.items():
        print(f'    {k:30} {v:0.4f}')


#####################################################################
# SHOW PIE CHARTS

def _show_pie_chart(df, chart):
    """
    Show a single investment pie chart.
    """
    title = f'{PORT.portfolio_title} - by {chart}'

    if chart == 'Investment Option':
        weights = list(df['Weight'])
        labels = list(df['Investment Option'])
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


def show_pie_charts(df, portfolio_option, charts=None):
    """
    Show pie chart(s) of investment allocations by percent.

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
    charts : list of str, optional
        {'Investment Option', 'Asset Class', 'Asset Subclass'}.
        The charts to display (default is None, which implies all
        charts).

    Returns
    -------
    None
    """

    def _add_weight_column(row, portfolio_option):
        inv_opt = row['Investment Option']
        return portfolio_option[inv_opt]

    # If charts is None, that imples all charts.
    if (charts is None):
        charts =['Investment Option', 'Asset Class', 'Asset Subclass']
    # If user specified a single chart, put it in a list.
    if not isinstance(charts, list):
        charts = [charts]
    # Check `charts` against valid chart choices.
    chart_choices = ('Investment Option', 'Asset Class', 'Asset Subclass')
    assert(set(charts).issubset(set(chart_choices))), \
           "Invalid Chart type in charts!"

    df = df.copy()
    df = df[df['Investment Option'].isin(list(portfolio_option))]
    df.reset_index(drop=True, inplace=True)
    df['Weight'] = df.apply(_add_weight_column, portfolio_option=portfolio_option, axis=1)
    df.sort_values('Asset Class', inplace=True)

    for chart in charts:
        s = _show_pie_chart(df, chart)
        print(s)


#####################################################################
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
        if math.isclose(s, 0): continue
        rands = [x/s for x in rands]

        annual_return = _expected_return(ml.annual_returns, weights=rands)
        std_dev = _standard_deviation(weights=rands, std_devs=ml.std_devs,
                                      asset_classes=ml.asset_classes)
        sharpe_ratio = _sharpe_ratio(annual_return, std_dev, PORT.risk_free_rate)
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

    ml = _get_metric_lists(df, portfolio_option)

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
    annual_return = _expected_return(annual_returns=ml.annual_returns,
                                     weights=optimal_weights)
    std_dev = _standard_deviation(weights=optimal_weights, std_devs=ml.std_devs,
                                  asset_classes=ml.asset_classes)
    sharpe_ratio = _sharpe_ratio(annual_return, std_dev, PORT.risk_free_rate)
    worst_typical_down_year = annual_return - 2*std_dev
    black_swan = annual_return - 3*std_dev
    _portfolio_option.update(zip(_portfolio_option, optimal_weights))

    s = pd.Series()
    s[PORT.portfolio_title + ' Metrics:'] = ''
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
