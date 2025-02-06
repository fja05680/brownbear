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
    get_symbol_fundamentals
)

from .trade import (
    TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_WEEK,
    calc_cash_per_investment,
    get_quote,
    calculate_target_portfolio,
    rebalance_portfolio,
    calculate_free_cash,
    allocate_free_cash
)

from .utility import (
    ROOT,
    SYMBOL_CACHE,
    print_full
)

def _whoami():
    ''' Returns the name of the calling function '''
    import inspect
    return inspect.stack()[1][3]

def _whosdaddy():
    ''' Returns the name of the calling function's parent '''
    import inspect
    return inspect.stack()[2][3]

DEBUG = False
def DBG(*s):
    if DEBUG: print('{}() {}\n'.format(_whosdaddy(), s))
    else:     pass
