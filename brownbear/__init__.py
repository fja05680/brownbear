from .brownbear import (
    fetch,
    rank,
    print_portfolio,
    analyze,
    summary,
    show_pie_charts,
    optimizer
)

from .utility import (
    ROOT,
    SYMBOL_CACHE,
    TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_WEEK,
    dotdict,
    correlation_map,
    fetch_timeseries,
    compile_timeseries,
    print_full,
    cagr,
    annualize_returns,
    annualized_standard_deviation
)

def _whoami():
    ''' Returns the name of the calling function '''
    import inspect
    return inspect.stack()[1][3]

def _whosdaddy():
    ''' Returns the name of the calling function's parent '''
    import inspect
    return inspect.stack()[2][3]

# debug
DEBUG = False
def DBG(*s):
    if DEBUG: print('{}() {}\n'.format(_whosdaddy(), s))
    else:     pass



