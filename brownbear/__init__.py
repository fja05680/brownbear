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
    show_pie_charts
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

from .utility import (
    TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_WEEK,
    ROOT,
    SYMBOL_CACHE,
    print_full,
    get_quote
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
