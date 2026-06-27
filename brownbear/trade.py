"""
Trade functions.
"""

import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import _yfinance_config  # noqa: F401
import yfinance as yf
from .utility import ROOT

_TOOLS_DIR = ROOT / 'tools'
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from symbol_replacements import (  # noqa: E402
    log_replacement,
    lookup_quote_price,
    normalize_portfolio,
    resolve_symbol,
)


########################################################################
# CONSTANTS

TRADING_DAYS_PER_YEAR = 252
"""
int : The number of trading days per year.
"""
TRADING_DAYS_PER_MONTH = 20
"""
int : The number of trading days per month.
"""
TRADING_DAYS_PER_WEEK = 5
"""
int : The number of trading days per week.
"""

LIVE_RECHECK_WAITS = (10, 20, 40, 80, 160)
"""
tuple of int : Seconds to wait between Schwab position rechecks after live trades.
"""


########################################################################
# FUNCTIONS

def calc_cash_per_investment(capital, portfolio_option):
    """ 
    Calculate how much cash to allocate to each investment based on the given
    portfolio option.

    Parameters
    ----------
    capital : float
        The total capital available for investment.
    portfolio_option : dict
        Dictionary where keys are stock symbols and values are the proportion 
        of capital allocated to each stock.

    Returns
    -------
    dict
        A dictionary where keys are stock symbols and values are the allocated
        cash amounts.
    """
    cash_per_investment = {k: float(capital * v) for k, v in portfolio_option.items()}
    return  cash_per_investment


def get_quote(symbols):
    """
    Fetch the latest stock prices for a list of symbols.

    Parameters
    ----------
    symbols : list of str
        A list of stock symbols to retrieve quotes for.

    Returns
    -------
    dict
        A dictionary where keys are stock symbols and values are the latest stock prices.
        If a quote cannot be fetched, the value will be None.
    """
    d = {}
    for symbol in symbols:
        query_symbol = resolve_symbol(symbol)
        if query_symbol != str(symbol).strip().upper():
            log_replacement(str(symbol).strip().upper(), query_symbol)
        ticker = yf.Ticker(query_symbol)
        try:
            d[symbol] = float(ticker.fast_info['last_price'])
        except KeyError:
            print(f'Could not fetch quote for {symbol}')
            d[symbol] = None
    return d


def calculate_target_portfolio(cash_per_investment, quote_per_investment):
    """
    Calculate the target portfolio based on cash per investment and the quote per investment.

    Parameters
    ----------
    cash_per_investment : dict
        Dictionary of investments and the amount of cash allocated to each.
    quote_per_investment : dict
        Dictionary of investments and the quote for each investment.

    Returns
    -------
    target_portfolio : dict
        A dictionary of stocks and the number of shares in the target portfolio.
    """
    target_portfolio = {
        k: int(v / float(quote_per_investment[k])) if quote_per_investment[k] not in ['N/A', 'None', ''] else 0
        for k, v in cash_per_investment.items()
    }
    return target_portfolio


def calculate_free_cash(capital, quote_per_investment, target_portfolio):
    """
    Calculate the free cash remaining after allocating funds to the target portfolio.

    Parameters
    ----------
    capital : float
        The total available capital for investment.
    quote_per_investment : dict
        Dictionary mapping stock symbols to their current prices.
    target_portfolio : dict
        Dictionary mapping stock symbols to the number of shares allocated.

    Returns
    -------
    float
        The amount of free cash remaining.
    """
    invested_amount = sum(target_portfolio[k] * float(quote_per_investment[k]) 
                          for k in target_portfolio if quote_per_investment[k] not in ['N/A', 'None', ''])
    free_cash = capital - invested_amount
    return free_cash


def allocate_free_cash(target_portfolio, quote_per_investment, free_cash, symbol):
    """
    Allocate free cash to purchase additional shares of a specified stock.

    Parameters
    ----------
    target_portfolio : dict
        Dictionary of stocks and the number of shares currently targeted.
    quote_per_investment : dict
        Dictionary of stocks and their current price per share.
    free_cash : float
        The amount of free cash available for allocation.
    symbol : str
        The stock symbol to which free cash should be allocated.

    Returns
    -------
    tuple
        Updated and sorted target portfolio with additional shares allocated to the specified stock,
        and the remaining free cash after allocation.
    """
    if (symbol not in quote_per_investment or
        quote_per_investment[symbol] in ['N/A', 'None', '', 0, None]):
        # Fetch the latest quote
        latest_quote = get_quote([symbol]).get(symbol)
        if latest_quote is None:
            print(f"Warning: Could not fetch quote for {symbol}. Skipping allocation.")
            return dict(sorted(target_portfolio.items())), free_cash
        quote_per_investment[symbol] = latest_quote

    share_price = float(quote_per_investment[symbol])
    shares_to_add = int(free_cash / share_price)
    allocated_cash = shares_to_add * share_price
    free_cash -= allocated_cash

    if shares_to_add > 0:
        target_portfolio[symbol] = target_portfolio.get(symbol, 0) + shares_to_add

    return dict(sorted(target_portfolio.items())), free_cash


def rebalance_portfolio(current_portfolio=None, target_portfolio=None):
    """
    Determine the stocks to buy and sell to match the target allocation.

    Parameters
    ----------
    current_portfolio : dict, optional
        Dictionary of stocks and the number of shares currently held.
        Defaults to None, which is treated as an empty portfolio.
    target_portfolio : dict, optional
        Dictionary of stocks and the number of shares desired.
        Defaults to None, which is treated as an empty allocation.

    Returns
    -------
    rebalance_orders : dict
        A dictionary with two keys:
        - 'sell': A sorted dict of stocks and the number of shares to sell.
        - 'buy': A sorted dict of stocks and the number of shares to buy.
    """
    if current_portfolio is None:
        current_portfolio = {}
    if target_portfolio is None:
        target_portfolio = {}

    current_portfolio = normalize_portfolio(current_portfolio)
    target_portfolio = normalize_portfolio(target_portfolio)

    sell = {}
    buy = {}

    all_stocks = set(current_portfolio.keys()).union(set(target_portfolio.keys()))

    for stock in all_stocks:
        current = current_portfolio.get(stock, 0)
        target = target_portfolio.get(stock, 0)

        if target > current:
            buy[stock] = target - current
        elif target < current:
            sell[stock] = current - target

    rebalance_orders = {
        "sell": dict(sorted(sell.items())),
        "buy": dict(sorted(buy.items()))
    }
    
    return rebalance_orders


def compare_portfolios(current_portfolio=None, target_portfolio=None):
    """
    Compare current holdings to a target allocation.

    Parameters
    ----------
    current_portfolio : dict, optional
        Symbol to share count currently held.
    target_portfolio : dict, optional
        Symbol to share count desired.

    Returns
    -------
    DataFrame
        One row per symbol with ``current``, ``target``, and ``delta`` columns.
        ``delta`` is current minus target (positive means overweight).
    """
    if current_portfolio is None:
        current_portfolio = {}
    if target_portfolio is None:
        target_portfolio = {}

    current_portfolio = normalize_portfolio(current_portfolio)
    target_portfolio = normalize_portfolio(target_portfolio)

    rows = []
    for symbol in sorted(set(current_portfolio) | set(target_portfolio)):
        current = current_portfolio.get(symbol, 0)
        target = target_portfolio.get(symbol, 0)
        rows.append({
            'symbol': symbol,
            'current': current,
            'target': target,
            'delta': current - target,
        })

    return pd.DataFrame(rows)


def fetch_schwab_portfolio(account_suffix, output_path):
    """
    Fetch current positions from Schwab and return a portfolio dict.

    Parameters
    ----------
    account_suffix : str
        Last digits of the Schwab account number.
    output_path : str or Path
        Where to write the account JSON snapshot.

    Returns
    -------
    dict
        Symbol to share count currently held.
    """
    output_path = Path(output_path)
    subprocess.run(
        [
            str(ROOT / 'tools' / 'schwab-account-info.sh'),
            str(account_suffix),
            '-o', str(output_path),
            '-q',
        ],
        check=True,
    )
    with output_path.open() as f:
        return json.load(f)['current_portfolio']


def verify_portfolio(
    account_suffix,
    target_portfolio,
    *,
    execute_live_trades=False,
    account_date=None,
    output_path=None,
    live_recheck_waits=LIVE_RECHECK_WAITS,
):
    """
    Verify Schwab holdings against a target allocation.

    Always fetches live positions from Schwab. When ``execute_live_trades`` is
    True and positions differ, recheck after 10s, 20s, 40s, 80s, and 160s.
    Preview mode performs a single immediate check with no delay.

    Parameters
    ----------
    account_suffix : str
        Last digits of the Schwab account number.
    target_portfolio : dict
        Symbol to share count desired.
    execute_live_trades : bool, optional
        Whether live orders were just placed (enables timed rechecks).
    account_date : str or date, optional
        Date used in the default output filename (default is today).
    output_path : str or Path, optional
        Account JSON path for Schwab snapshots.
    live_recheck_waits : sequence of int, optional
        Seconds to wait between rechecks after live trades.

    Returns
    -------
    DataFrame
        Comparison from ``compare_portfolios`` for the final check.
    """
    if account_date is None:
        account_date = datetime.date.today().isoformat()
    else:
        account_date = str(account_date)
    if output_path is None:
        output_path = Path(f'account-{account_date}-post-trade.json')

    portfolio = fetch_schwab_portfolio(account_suffix, output_path)
    comparison = compare_portfolios(portfolio, target_portfolio)
    mismatches = comparison.loc[comparison['delta'] != 0]

    if mismatches.empty:
        print('Positions match target portfolio.')
    else:
        print(f'{len(mismatches)} position(s) differ from target:')

    if execute_live_trades and not mismatches.empty:
        for wait_seconds in live_recheck_waits:
            print(f'Waiting {wait_seconds}s before rechecking...')
            time.sleep(wait_seconds)
            portfolio = fetch_schwab_portfolio(account_suffix, output_path)
            comparison = compare_portfolios(portfolio, target_portfolio)
            mismatches = comparison.loc[comparison['delta'] != 0]
            if mismatches.empty:
                print('Positions match target portfolio.')
                break
            print(f'{len(mismatches)} position(s) differ from target:')

        if not mismatches.empty:
            print('Positions still differ from target after all rechecks.')

    return comparison


def apply_sell_only_below_ma_filter(
    rebalance_orders,
    symbol='SPY',
    ma_window=200,
    end=None,
    verbose=True,
):
    """
    Remove buy orders when a market proxy is below its moving average.

    Fetches a fresh timeseries from Yahoo Finance (not symbol_cache).
    When price is below the moving average, only sell orders are kept.
    Otherwise buy and sell orders are unchanged.

    Parameters
    ----------
    rebalance_orders : dict
        Output of rebalance_portfolio with ``sell`` and ``buy`` keys.
    symbol : str, optional
        Market proxy symbol (default is ``'SPY'``).
    ma_window : int, optional
        Moving-average window in trading days (default is 200).
    end : date, optional
        End date for the timeseries (default is today).
    verbose : bool, optional
        Print price, moving average, and filter decision (default is True).

    Returns
    -------
    dict
        Updated rebalance orders with the same ``sell`` and ``buy`` keys.
    """
    if end is None:
        end = datetime.date.today()
    elif not isinstance(end, datetime.date):
        end = pd.Timestamp(end).date()

    start = end - datetime.timedelta(days=ma_window + 100)
    df = yf.download(
        symbol, start=start, end=end, progress=False,
        auto_adjust=False, multi_level_index=False,
        threads=False, timeout=30,
    )
    if df.empty:
        raise ValueError(f'No timeseries data returned for {symbol}')

    close = df['Close']
    ma = close.rolling(ma_window).mean()
    price = float(close.iloc[-1])
    moving_average = float(ma.iloc[-1])

    if pd.isna(moving_average):
        raise ValueError(
            f'Not enough data to compute {ma_window}-day MA for {symbol}'
        )

    filtered = {
        'sell': dict(rebalance_orders.get('sell', {})),
        'buy': dict(rebalance_orders.get('buy', {})),
    }

    below_ma = price < moving_average
    if below_ma:
        filtered['buy'] = {}

    if verbose:
        print(f'{symbol} price: {price:.2f}')
        print(f'{symbol} {ma_window}-day MA: {moving_average:.2f}')
        if below_ma:
            print(f'{symbol} below {ma_window}-day MA: sell orders only')
        else:
            print(f'{symbol} at or above {ma_window}-day MA: buy and sell orders')

    return filtered


def rebalance_orders_to_dataframe(
    rebalance_orders,
    account=None,
    as_of_date=None,
    quote_per_investment=None,
):
    """
    Convert rebalance_orders to a flat DataFrame for trade execution.

    Parameters
    ----------
    rebalance_orders : dict
        Output of rebalance_portfolio with ``sell`` and ``buy`` keys.
    account : str, optional
        Broker account identifier.
    as_of_date : str or date, optional
        Trade date in ISO format (default is today).
    quote_per_investment : dict, optional
        Symbol to price mapping; adds ``price`` and ``estimated_value`` columns.

    Returns
    -------
    DataFrame
        One row per order with columns ``account``, ``side``, ``symbol``,
        ``quantity``, and ``as_of_date``.
    """
    if as_of_date is None:
        as_of_date = datetime.date.today().isoformat()
    else:
        as_of_date = str(as_of_date)

    rows = []
    for side in ('sell', 'buy'):
        for symbol, quantity in rebalance_orders.get(side, {}).items():
            trade_symbol = resolve_symbol(symbol)
            if trade_symbol != str(symbol).strip().upper():
                log_replacement(str(symbol).strip().upper(), trade_symbol)
            row = {
                'account': account,
                'side': side.upper(),
                'symbol': trade_symbol,
                'quantity': quantity,
                'as_of_date': as_of_date,
            }
            price = lookup_quote_price(trade_symbol, quote_per_investment)
            if price is not None:
                row['price'] = price
                row['estimated_value'] = round(price * quantity, 2)
            rows.append(row)

    return pd.DataFrame(rows)


def write_rebalance_orders_csv(rebalance_orders, filepath, **kwargs):
    """
    Write rebalance orders to a CSV file.

    Parameters
    ----------
    rebalance_orders : dict
        Output of rebalance_portfolio.
    filepath : str or Path
        Output CSV path.
    **kwargs
        Passed to rebalance_orders_to_dataframe.

    Returns
    -------
    DataFrame
        The orders written to ``filepath``.
    """
    filepath = Path(filepath)
    df = rebalance_orders_to_dataframe(rebalance_orders, **kwargs)
    df.to_csv(filepath, index=False)
    return df
