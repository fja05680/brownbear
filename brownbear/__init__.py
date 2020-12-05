from .brownbear import (
    fetch,
    add_sharpe_ratio_column,
    rank,
    analyze,
    summary,
    optimizer
)

from .utility import (
    ROOT,
    SYMBOL_CACHE,
    TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_WEEK,
    correlation_map,
    fetch_timeseries,
    compile_timeseries,
    print_full,
    cagr,
    annualize_returns,
    annualized_standard_deviation
)


DEBUG = False
def DBG(s):
    if DEBUG: print(s)
    else:     pass


