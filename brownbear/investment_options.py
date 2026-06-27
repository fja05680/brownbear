"""
Build investment-options.csv from investment-options-in.csv and symbol timeseries.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import annualized_returns, annualized_standard_deviation
from .symbol_cache import compile_timeseries, fetch_timeseries
from .trade import TRADING_DAYS_PER_MONTH, TRADING_DAYS_PER_YEAR


def read_input_lines(input_path):
    """
    Return stripped lines from an investment-options input CSV.

    Parameters
    ----------
    input_path : str or Path
        Path to investment-options-in.csv.

    Returns
    -------
    list of str
    """
    with Path(input_path).open() as f:
        return [line.strip() for line in f]


def read_symbols_from_input(input_path):
    """
    Return unique symbols from an investment-options-in.csv file.

    Parameters
    ----------
    input_path : str or Path
        Path to investment-options-in.csv.

    Returns
    -------
    list of str
        Sorted unique values from the ``Investment Option`` column.
    """
    df = pd.read_csv(input_path, skip_blank_lines=True, comment='#')
    return sorted(set(df['Investment Option']))


def load_gics_asset_class_map(directory, filename='gics-2-asset-class.csv'):
    """
    Load GICS sector to brownbear asset class mapping.

    Parameters
    ----------
    directory : str or Path
        Folder containing ``gics-2-asset-class.csv``.
    filename : str, optional
        GICS mapping filename (default is ``'gics-2-asset-class.csv'``).

    Returns
    -------
    dict
        GICS sector name to brownbear asset class name.
    """
    directory = Path(directory)
    df = pd.read_csv(directory / filename, skip_blank_lines=True, comment='#')
    df.set_index('GICS', inplace=True)
    return df['Asset Class'].to_dict()


def _yahoo_symbols(universe_df):
    """
    Convert index symbols to Yahoo Finance format (e.g. BRK.B -> BRK-B).

    Parameters
    ----------
    universe_df : pd.DataFrame
        Constituent universe indexed by symbol.

    Returns
    -------
    pd.DataFrame
        Copy of ``universe_df`` with Yahoo-compatible index symbols.
    """
    universe_df = universe_df.copy()
    universe_df.index = universe_df.index.str.replace('.', '-', regex=False)
    return universe_df


def load_dow30_universe(directory):
    """
    Load and normalize the DOW 30 constituent universe.

    Parameters
    ----------
    directory : str or Path
        Folder containing ``dow30.csv`` and ``gics-2-asset-class.csv``.

    Returns
    -------
    pd.DataFrame
        Indexed by symbol with ``Description`` and ``Asset Class`` columns.
    """
    directory = Path(directory)
    gics_map = load_gics_asset_class_map(directory)
    df = pd.read_csv(directory / 'dow30.csv')
    df['Symbol'] = df['Symbol'].str.split(':').str[-1].str.strip()
    df.drop(
        columns=['Exchange', 'Date added', 'Notes', 'Index weighting'],
        inplace=True,
    )
    df.rename(
        columns={'Company': 'Description', 'Sector': 'Asset Class'},
        inplace=True,
    )
    df.set_index('Symbol', inplace=True)
    df['Asset Class'] = df['Asset Class'].map(gics_map)
    return _yahoo_symbols(df)


def load_sp500_universe(directory):
    """
    Load and normalize the S&P 500 constituent universe.

    Parameters
    ----------
    directory : str or Path
        Folder containing ``sp500.csv`` and ``gics-2-asset-class.csv``.

    Returns
    -------
    pd.DataFrame
        Indexed by symbol with ``Description`` and ``Asset Class`` columns.
    """
    directory = Path(directory)
    gics_map = load_gics_asset_class_map(directory)
    df = pd.read_csv(directory / 'sp500.csv')
    df.drop(
        columns=['Headquarters Location', 'Date added', 'CIK', 'Founded'],
        inplace=True,
    )
    df.rename(
        columns={'Security': 'Description', 'GICS Sector': 'Asset Class'},
        inplace=True,
    )
    df.set_index('Symbol', inplace=True)
    df['Asset Class'] = df['Asset Class'].map(gics_map)
    df = _yahoo_symbols(df)
    return df.drop('FDXF', errors='ignore')


def load_sp400_universe(directory):
    """
    Load and normalize the S&P 400 constituent universe.

    Parameters
    ----------
    directory : str or Path
        Folder containing ``sp400.csv`` and ``gics-2-asset-class.csv``.

    Returns
    -------
    pd.DataFrame
        Indexed by symbol with ``Description`` and ``Asset Class`` columns.
    """
    directory = Path(directory)
    gics_map = load_gics_asset_class_map(directory)
    df = pd.read_csv(directory / 'sp400.csv')
    df.drop(columns=['SEC filings'], inplace=True)
    df.rename(
        columns={
            'Security': 'Description',
            'GICS Sector': 'Asset Class',
            'GICS Sub-Industry': 'GICS Sub Industry',
        },
        inplace=True,
    )
    df.set_index('Symbol', inplace=True)
    df['Asset Class'] = df['Asset Class'].map(gics_map)
    df = _yahoo_symbols(df)
    return df.drop('CVLT', errors='ignore')


def load_sp600_universe(directory):
    """
    Load and normalize the S&P 600 constituent universe.

    Parameters
    ----------
    directory : str or Path
        Folder containing ``sp600.csv`` and ``gics-2-asset-class.csv``.

    Returns
    -------
    pd.DataFrame
        Indexed by symbol with ``Description`` and ``Asset Class`` columns.
    """
    directory = Path(directory)
    gics_map = load_gics_asset_class_map(directory)
    df = pd.read_csv(directory / 'sp600.csv')
    df.drop(columns=['SEC filings', 'CIK'], inplace=True)
    df.rename(
        columns={'Security': 'Description', 'GICS Sector': 'Asset Class'},
        inplace=True,
    )
    df.set_index('Symbol', inplace=True)
    df['Asset Class'] = df['Asset Class'].map(gics_map)
    df = _yahoo_symbols(df)
    return df.drop('VSNT', errors='ignore')


def load_nasdaq100_universe(directory):
    """
    Load and normalize the Nasdaq 100 constituent universe.

    Parameters
    ----------
    directory : str or Path
        Folder containing ``nasdaq100.csv``.

    Returns
    -------
    pd.DataFrame
        Indexed by symbol with ``Description`` and ``Asset Class`` columns.
    """
    directory = Path(directory)
    df = pd.read_csv(directory / 'nasdaq100.csv')
    df.rename(columns={'Ticker': 'Symbol', 'Company': 'Description'}, inplace=True)
    df.set_index('Symbol', inplace=True)
    df['Asset Class'] = 'US Stocks'
    return _yahoo_symbols(df)


def compute_investment_metrics(timeseries_df):
    """
    Compute return and volatility metrics from a Date-indexed price dataframe.

    Parameters
    ----------
    timeseries_df : pd.DataFrame
        Daily closing prices with one column per symbol and a Date index.

    Returns
    -------
    dict
        Metric name to pandas Series keyed by symbol.  Keys include
        ``annual_returns_3mo``, ``annual_returns_6mo``, ``annual_returns_1yr``,
        ``annual_returns_1_1yr``, ``annual_returns_3yr``, ``annual_returns_5yr``,
        ``vola``, ``ds_vola``, ``std_dev_1yr``, ``std_dev_3yr``, and
        ``std_dev_5yr``.
    """
    df = timeseries_df.copy()
    df.index = pd.to_datetime(df.index)

    daily_returns = df.pct_change(fill_method=None)
    vola_years = TRADING_DAYS_PER_MONTH / TRADING_DAYS_PER_YEAR

    monthly = df.resample('ME').ffill()
    monthly_returns = monthly.pct_change(fill_method=None)

    return {
        'annual_returns_3mo': annualized_returns(df, timeperiod='daily', years=3 / 12),
        'annual_returns_6mo': annualized_returns(df, timeperiod='daily', years=6 / 12),
        'annual_returns_1yr': annualized_returns(df, timeperiod='daily', years=1),
        'annual_returns_1_1yr': annualized_returns(
            df, timeperiod='daily', years=1, offset=1,
        ),
        'annual_returns_3yr': annualized_returns(df, timeperiod='daily', years=3),
        'annual_returns_5yr': annualized_returns(df, timeperiod='daily', years=5),
        'vola': annualized_standard_deviation(
            daily_returns, timeperiod='daily', years=vola_years,
        ),
        'ds_vola': annualized_standard_deviation(
            daily_returns, timeperiod='daily', years=vola_years, downside=True,
        ),
        'std_dev_1yr': annualized_standard_deviation(
            monthly_returns, timeperiod='monthly', years=1,
        ),
        'std_dev_3yr': annualized_standard_deviation(
            monthly_returns, timeperiod='monthly', years=3,
        ),
        'std_dev_5yr': annualized_standard_deviation(
            monthly_returns, timeperiod='monthly', years=5,
        ),
    }


def _format_metrics_row(symbol, description, asset_class, metrics):
    """
    Format one investment-options.csv data row with computed metrics.

    Falls back to shorter return periods when 3 Yr or 5 Yr data is missing.

    Parameters
    ----------
    symbol : str
        Investment option ticker symbol.
    description : str
        Human-readable description for the symbol.
    asset_class : str
        Brownbear asset class label.
    metrics : dict
        Output of :func:`compute_investment_metrics`.

    Returns
    -------
    str
        One CSV line with quoted fields for symbol metadata and metrics.
    """
    ret_3mo = metrics['annual_returns_3mo'][symbol]
    ret_6mo = metrics['annual_returns_6mo'][symbol]
    ret_1yr = metrics['annual_returns_1yr'][symbol]
    ret_1_1yr = metrics['annual_returns_1_1yr'][symbol]
    ret_3yr = metrics['annual_returns_3yr'][symbol]
    ret_5yr = metrics['annual_returns_5yr'][symbol]

    if np.isnan(ret_3yr):
        ret_3yr = ret_1yr
    if np.isnan(ret_5yr):
        ret_5yr = ret_3yr

    vola = metrics['vola'][symbol] * 100
    ds_vola = metrics['ds_vola'][symbol] * 100
    sd_1yr = metrics['std_dev_1yr'][symbol] * 100
    sd_3yr = metrics['std_dev_3yr'][symbol] * 100
    sd_5yr = metrics['std_dev_5yr'][symbol] * 100

    return (
        f'"{symbol}","{description}","{asset_class}",'
        f'"{ret_3mo:0.2f}","{ret_6mo:0.2f}","{ret_1yr:0.2f}","{ret_1_1yr:0.2f}","{ret_3yr:0.2f}",'
        f'"{ret_5yr:0.2f}","{vola:0.2f}","{ds_vola:0.2f}","{sd_1yr:0.2f}","{sd_3yr:0.2f}","{sd_5yr:0.2f}"'
    )


def format_investment_options_lines(input_lines, metrics):
    """
    Merge computed metrics into investment-options-in.csv lines.

    Preserves blank lines, comments, and the header row unchanged.

    Parameters
    ----------
    input_lines : list of str
        Lines from investment-options-in.csv.
    metrics : dict
        Output of :func:`compute_investment_metrics`.

    Returns
    -------
    list of str
        Lines ready to write to investment-options.csv.
    """
    out = []
    for line in input_lines:
        if not line or line.startswith('#'):
            out.append(line)
            continue

        fields = [field.strip() for field in line.split(',')]
        symbol = fields[0].strip('"')
        if symbol == 'Investment Option':
            out.append(line)
            continue

        out.append(_format_metrics_row(
            symbol, fields[1].strip('"'), fields[2].strip('"'), metrics,
        ))
    return out


def format_investment_options_from_universe(header_lines, universe_df, metrics):
    """
    Build investment-options.csv lines from a constituent universe dataframe.

    Parameters
    ----------
    header_lines : list of str
        Comment, format, and column header lines from investment-options-in.csv.
    universe_df : pd.DataFrame
        Constituent universe indexed by symbol with ``Description`` and
        ``Asset Class`` columns.
    metrics : dict
        Output of :func:`compute_investment_metrics`.

    Returns
    -------
    list of str
        Lines ready to write to investment-options.csv.
    """
    out = list(header_lines)
    for symbol, row in universe_df.iterrows():
        out.append(_format_metrics_row(
            symbol, row['Description'], row['Asset Class'], metrics,
        ))
    return out


def _write_output_csv(output_path, output_lines):
    """
    Write investment-options output lines to a CSV file.

    Parameters
    ----------
    output_path : str or Path
        Destination CSV path.
    output_lines : list of str
        Lines to write, one per row.

    Returns
    -------
    None
    """
    with Path(output_path).open('w') as f:
        for line in output_lines:
            f.write(line + '\n')


def update_investment_options(
    directory='.',
    input_filename='investment-options-in.csv',
    output_filename='investment-options.csv',
    timeseries_filename='symbols-timeseries.csv',
    refresh_timeseries=False,
    throttle_limit=100,
    wait_time=30,
    universe_df=None,
):
    """
    Refresh investment-options.csv from investment-options-in.csv.

    Fetches symbol timeseries when needed, computes returns and volatility
    metrics, and writes the output CSV beside the input file.

    When ``universe_df`` is provided, symbols and output rows come from the
    index constituent dataframe instead of symbol rows in the input CSV.

    Parameters
    ----------
    directory : str or Path, optional
        Folder containing the investment-options files (default is cwd).
    input_filename : str, optional
        Editable template CSV with symbol metadata and blank metric columns.
    output_filename : str, optional
        Generated CSV consumed by ``bb.fetch()``.
    timeseries_filename : str, optional
        Compiled daily close prices written and read within ``directory``.
    refresh_timeseries : bool, optional
        Download fresh Yahoo timeseries instead of using symbol cache.
    throttle_limit : int, optional
        Symbols to fetch before throttling (default is 100).
    wait_time : int, optional
        Seconds to wait after reaching ``throttle_limit`` (default is 30).
    universe_df : DataFrame, optional
        Constituent universe indexed by symbol with ``Description`` and
        ``Asset Class`` columns.

    Returns
    -------
    DataFrame
        The generated ``investment-options.csv`` contents.
    """
    directory = Path(directory)
    input_path = directory / input_filename
    output_path = directory / output_filename
    timeseries_path = directory / timeseries_filename

    if universe_df is not None:
        symbols = list(universe_df.index)
    else:
        symbols = read_symbols_from_input(input_path)
    fetch_timeseries(
        symbols,
        refresh=refresh_timeseries,
        throttle_limit=throttle_limit,
        wait_time=wait_time,
    )
    compile_timeseries(symbols, output_path=timeseries_path)

    timeseries_df = pd.read_csv(
        timeseries_path, skip_blank_lines=True, comment='#',
    )
    timeseries_df.set_index('Date', inplace=True)
    metrics = compute_investment_metrics(timeseries_df)

    input_lines = read_input_lines(input_path)
    if universe_df is not None:
        output_lines = format_investment_options_from_universe(
            input_lines, universe_df, metrics,
        )
    else:
        output_lines = format_investment_options_lines(input_lines, metrics)
    _write_output_csv(output_path, output_lines)

    return pd.read_csv(output_path, skip_blank_lines=True, comment='#')
