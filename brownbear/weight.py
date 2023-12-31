"""
Portfolio analysis and optimization.
"""

import math

import numpy

import brownbear as bb
from brownbear.portfolio import (
    PORT,
    get_metric_lists,
    sharpe_ratio
)
from brownbear.utility import (
    dotdict
)


def _calc_weights(df, asset_dict, weight_by):
    """
    Calculate weights for assets in asset_dict using weight_by method.
    """
    weight_by_choices = ('Equal', 'Sharpe Ratio', 'Annual Returns',
                         'Std Dev', 'Vola', 'DS Vola')
    assert weight_by in weight_by_choices, f"Invalid weight_by '{weight_by}'"

    ml = get_metric_lists(df, asset_dict)
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
                ml.sharpe_ratios = [sharpe_ratio(annual_ret, std_dev, PORT.risk_free_rate)
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


def check_allocation(weights, asset_class_name):
    """
    Make sure total adds to 100.

    weights : dict
        Dictionary of investment options along with their weights.  The
        keys are the investment options and the values are the weights.
    asset_class_name : str
        Description of the asset class.
    """
    s = sum(weights.values())
    if not math.isclose(s, 1, rel_tol=1e-09, abs_tol=0.0):
        raise Exception(f"Error: {asset_class_name} allocation of '{s}' is not 100%!!!")


def assign_weights(df, ml, portfolio_option, weight_by):

    """
    Specify the weighting scheme.  It will replace the weights specified
    in the portfolio.  You can also fix the weights on some
    Investent Options, Asset Classes, and Asset Subclasses while the
    others are automatically calculated.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of investment options with columns for asset class,
        description, and performace metrics.
    ml: bb.dotdict of lists
        Creates dict of lists for investment option, std_dev,
        asset_class, etc... for each investment option for the
        specified portfolio.
    portfolio_option : dict
        Dictionary of investment options along with their weights.  The
        keys are the investment options and the values are the weights.
        The first entry in the dict must be the title of the portfolio.
    weight_by : dict of dicts, optional
        Specify the weighting scheme.  If not None, it will replace the
        weights specified in the portfolio.  You can also fix the
        weights on some Investent Options, Asset Classes, and Asset
        Subclasses, while the others are automatically calculated
        (default is None, which implies use the user specified weights
        specified in `portfolio_option`).  See bb.analyze.analyze()
        for more information about this parameter.
    
    Returns
    -------
    None
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
        check_allocation(w, 'Investment Option')
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
        check_allocation(w, 'Asset Class')
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
            user.asset_subclass_weights_ = {k: user.asset_subclass_weights[k] for k in i}
            w = _get_cmpt_weights(PORT.asset_class_table, d, user.asset_subclass_weights_,
                                  user.asset_subclass_weight_by)
            check_allocation(w, 'Asset Sublass')
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
            user.inv_opt_weights_ = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user.inv_opt_weights_, user.inv_opt_weight_by)
            check_allocation(w, 'Investment Option')
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
        check_allocation(w, 'Asset Class')
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
            user.inv_opt_weights_ = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user.inv_opt_weights_, user.inv_opt_weight_by)
            check_allocation(w, 'Investment Option')
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
        check_allocation(w, 'Asset SubClass')
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
            user.inv_opt_weights_ = {k: user.inv_opt_weights[k] for k in i}
            w = _get_cmpt_weights(df, d, user.inv_opt_weights_, user.inv_opt_weight_by)
            check_allocation(w, 'Investment Option')
            cmpt.inv_opt_weights.update(w)
        bb.DBG('cmpt.inv_opt_weights', cmpt.inv_opt_weights)

        _calc_portfolio_option_weights(portfolio_option, ml, cmpt, user)
        bb.DBG('portfolio_option', portfolio_option)
        return
