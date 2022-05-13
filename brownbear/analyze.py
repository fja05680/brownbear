"""
Portfolio analysis and optimization.
"""

import matplotlib.pyplot as plt
import pandas as pd

import brownbear as bb

from brownbear.portfolio import (
    PORT,
    get_metric_lists,
    expected_return,
    standard_deviation,
    sharpe_ratio
)

from brownbear.weight import (
    check_allocation,
    assign_weights
)


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
                'Asset Subclass' means do not group by Asset Subclass

        Example:

        At the Asset Class level, explicitly specify
        US Stock, US Bonds, and Risk-Free Asset weights, then equally
        allocate among any remaining asset classes.  Next, do not
        consider the Asset Subclass within an Asset Class.  Finally,
        weight the Investment Options within each Asset Class by
        Annual Return.

        >>> weight_by = {  
        >>>     'Asset Class':       {'weight_by': 'Equal',  
        >>>                           'US Stocks': 0.40,  
        >>>                           'US Bonds': 0.40,  
        >>>                           'Risk-Free Asset': 0.10},  
        >>>     'Asset Subclass':    {'weight_by': None},  
        >>>     'Investment Option': {'weight_by': 'Annual Returns'},  
        >>> }  

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

    sr : float
        The overall sharpe ratio of the portfolio.
    """

    # Pop the title.
    PORT.portfolio_title = \
        portfolio_option.pop('Title', PORT.portfolio_title)

    # Set default correlation.
    PORT.default_correlation = default_correlation

    # Get metric_lists.
    ml = get_metric_lists(df, portfolio_option)
    bb.DBG(f'ml = {ml}')

    # Assign weights.
    assign_weights(df, ml, portfolio_option, weight_by)

    # Make sure total adds to 100 percent.
    check_allocation(portfolio_option, PORT.portfolio_title)

    # Update metric_lists.
    ml = get_metric_lists(df, portfolio_option)

    # Compute metrics.
    annual_return = expected_return(ml.annual_returns, ml.weights)
    std_dev = standard_deviation(ml.weights, ml.std_devs, ml.asset_classes)
    sr = sharpe_ratio(annual_return, std_dev, PORT.risk_free_rate)

    return annual_return, std_dev, sr


########################################################################
# SUMMARY

def _plot_returns(summary_results, columns):
    """
    Bar Plot of returns with 1, 2, and 3 standard deviations.
    """
    means = list(summary_results.loc['Annual Returns'])
    Xs = list(range(0, len(means)))
    # Plot 1 std dev.
    maxs = list(summary_results.loc['Std Dev'])
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


def summary(df, portfolio_option, annual_ret, std_dev, sr):
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
    annual_return : float
        The expected annualized return of the portfolio.
    std_dev : float
        The standard deviation of the portfolio.
    sr : float
        The overall sharpe ratio of the portfolio.

    Returns
    -------
    summary_results : pd.DataFrame
        Summary results.
    """
    ml = get_metric_lists(df, portfolio_option)

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
    summary_results = pd.DataFrame(data, columns=columns, index=index)

    # Set portfolio values.
    summary_results[PORT.portfolio_title] = \
        [annual_ret, std_dev, sr, sum(ml.weights),
         annual_ret + -2*std_dev, annual_ret + -3*std_dev]

    # Plot returns.
    column_names = columns.copy()
    column_names.append(PORT.portfolio_title)
    _plot_returns(summary_results, column_names)

    return summary_results


def print_portfolio(portfolio_option):
    """
    Print portfolio options with their weights.

    Parameters
    ----------
    portfolio_option : dict
        Dictionary of investment options along with their weights.  The
        keys are the investment options and the values are the weights.
        The first entry in the dict must be the title of the portfolio.

    Returns
    -------
    None
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
    if charts is None:
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
