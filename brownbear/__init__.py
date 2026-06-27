from . import _yfinance_config  # noqa: F401

from .metrics import (
    correlation_map,
    annualized_returns,
    annualized_standard_deviation
)

from .fetch import (
    fetch,
    add_fundamental_columns,
    rank
)

from .investment_options import (
    update_investment_options,
    compute_investment_metrics,
    load_dow30_universe,
    load_sp400_universe,
    load_sp500_universe,
    load_sp600_universe,
    load_nasdaq100_universe,
)

from .analyze import (
    analyze,
    summary,
    print_portfolio,
    show_pie_charts,
    show_correlation_heatmap,
    calc_portfolio_correlation
)

from .optimizer import (
    optimizer
)

from .symbol_cache import (
    fetch_timeseries,
    compile_timeseries,
    remove_cache_symbols,
    update_cache_symbols,
    get_symbol_metadata,
    get_symbol_fundamentals,
    reset_fundamentals_cache
)

from .trade import (
    TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_WEEK,
    calc_cash_per_investment,
    get_quote,
    calculate_target_portfolio,
    rebalance_portfolio,
    compare_portfolios,
    fetch_schwab_portfolio,
    verify_portfolio,
    apply_sell_only_below_ma_filter,
    rebalance_orders_to_dataframe,
    write_rebalance_orders_csv,
    calculate_free_cash,
    allocate_free_cash
)

from .utility import (
    ROOT,
    SYMBOL_CACHE,
    notebook_display_options,
    print_full
)

def _whoami():
    """
    Return the name of the calling function.

    Returns
    -------
    str
    """
    import inspect
    return inspect.stack()[1][3]

def _whosdaddy():
    """
    Return the name of the calling function's parent.

    Returns
    -------
    str
    """
    import inspect
    return inspect.stack()[2][3]

DEBUG = False
def DBG(*s):
    """
    Print debug output when ``DEBUG`` is True.

    Parameters
    ----------
    *s
        Values to print, prefixed with the caller's function name.

    Returns
    -------
    None
    """
    if DEBUG: print('{}() {}\n'.format(_whosdaddy(), s))
    else:     pass
