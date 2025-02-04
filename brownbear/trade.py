"""
Trade functions.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


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
        ticker = yf.Ticker(symbol)
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
        - 'sell': A dict of stocks and the number of shares to sell.
        - 'buy': A dict of stocks and the number of shares to buy.
    """
    if current_portfolio is None:
        current_portfolio = {}
    if target_portfolio is None:
        target_portfolio = {}

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

    rebalance_orders = {"sell": sell, "buy": buy}
    return rebalance_orders


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
        Updated target portfolio with additional shares allocated to the specified stock,
        and the remaining free cash after allocation.
    """
    if symbol not in quote_per_investment or quote_per_investment[symbol] in ['N/A', 'None', '', 0]:
        raise ValueError(f"Invalid or missing price for symbol: {symbol}")

    share_price = float(quote_per_investment[symbol])
    shares_to_add = int(free_cash / share_price)
    allocated_cash = shares_to_add * share_price
    free_cash -= allocated_cash

    if shares_to_add > 0:
        target_portfolio[symbol] = target_portfolio.get(symbol, 0) + shares_to_add

    return target_portfolio, free_cash



