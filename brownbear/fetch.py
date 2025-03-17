"""
Fetch performance data for investment options.
"""

import pandas as pd

import brownbear as bb
from brownbear.portfolio import (
    PORT,
    sharpe_ratio
)
from brownbear.utility import (
    csv_to_df
)


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


def _add_sharpe_ratio_column(inv_opts, risk_free_rate):
    """
    Add Sharpe Ratio column to the inv_opts dataframe.
    """
    def _sharpe(row, risk_free_rate):
        annual_ret = row['Annual Returns']
        std_dev = row['Std Dev']
        return sharpe_ratio(annual_ret, std_dev, risk_free_rate)

    inv_opts['Sharpe Ratio'] = inv_opts.apply(_sharpe, risk_free_rate=risk_free_rate, axis=1)
    return inv_opts


def fetch(investment_universe, risk_free_rate=0,
          annual_returns='5 Yr', standard_deviation='SD 3 Yr',
          vola='Vola', ds_vola='DS Vola', clean=True):
    """
    Fetch Investment Universe and asset classes

    investment-options.csv format:
        "Investment Option", "Description"(optimal), "Asset Class",
        "Annual Returns", "Std Dev"

    asset-classes.csv format:
        "Asset Class A", "Asset Class B","Correlation"
        "Description" field is optional.  It is not referenced in code.
        "Annual Returns" column(s) can named anything.
            Recommend "1 Yr", "3 Yr", or "5 Yr".  Then
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
        (default is '5 Yr').  It will also be used
        in the sharpe_ratio calculation.
    standard_deviation : str, optional
        Specifies which column to use for standard deviation
        (default is 'SD 3 Yr').  It will also be used
        in the sharpe_ratio calculation.
    vola : str, optional
        Specifies which column to use for volatility
        (default is 'Vola').
    ds_vola : str, optional
        Specifies which column to use for downside volatility
        (default is 'DS Vola').
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

    inv_opts = csv_to_df(filepaths)

    # Drop duplicate Investment Option's, keep the first,
    # then reset index.
    inv_opts.drop_duplicates(subset=['Investment Option'], keep='first', inplace=True)
    inv_opts.reset_index(drop=True, inplace=True)

    # Allows the use of different annualized returns,
    # e.g. 1, 3, or 5 year annaulized returns.
    inv_opts['Annual Returns'] = inv_opts[annual_returns]

    # Allows the use of different standard deviations,
    # e.g. 1 or 3 year standard deviation.
    inv_opts['Std Dev'] = inv_opts[standard_deviation]

    # Add Sharpe Ratio column.
    inv_opts = _add_sharpe_ratio_column(inv_opts, risk_free_rate)

    # Asset class table.
    PORT.asset_class_table = csv_to_df(
        [bb.ROOT / 'universe' / 'asset-class-galaxy' / 'investment-options.csv'])

    # Add Annual Returns column to asset class table.
    PORT.asset_class_table['Annual Returns'] = PORT.asset_class_table[annual_returns]

    # Add Std Dev column to asset class table.
    PORT.asset_class_table['Std Dev'] = PORT.asset_class_table[standard_deviation]

    # Add Sharpe Ratio column to asset class table.
    PORT.asset_class_table = _add_sharpe_ratio_column(PORT.asset_class_table, risk_free_rate)

    # Correlation table.
    PORT.correlation_table = csv_to_df(
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
    df.rename(columns={
        'symbol': 'Investment Option',
        'previousClose': 'Previous Close',
        'trailingPE': 'Trailing PE',
        'dividendYield': 'Dividend Yield',
        'marketCap': 'Market Cap'
        }, inplace=True)
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
