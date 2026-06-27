"""Trading symbol replacements when tickers change."""

import csv
from functools import lru_cache
from pathlib import Path

REPLACEMENTS_FILE = Path(__file__).resolve().parent / 'symbol-replacements.csv'


@lru_cache(maxsize=1)
def load_replacements():
    """Return a dict of old_symbol -> new_symbol from symbol-replacements.csv."""
    mapping = {}
    if not REPLACEMENTS_FILE.is_file():
        return mapping

    with REPLACEMENTS_FILE.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            old = row.get('old_symbol', '').strip().upper()
            new = row.get('new_symbol', '').strip().upper()
            if old and new:
                mapping[old] = new
    return mapping


def resolve_symbol(symbol):
    """Return the current trading symbol, applying replacements when configured."""
    if symbol is None:
        return symbol
    symbol = str(symbol).strip().upper()
    return load_replacements().get(symbol, symbol)


def resolve_symbols(symbols):
    """Return {original: resolved} for a list of symbols."""
    return {symbol: resolve_symbol(symbol) for symbol in symbols}


def log_replacement(old_symbol, new_symbol):
    if old_symbol != new_symbol:
        print(f'Symbol replacement: {old_symbol} -> {new_symbol}')


def normalize_portfolio(portfolio):
    """
    Rename portfolio keys using symbol replacements and merge duplicate keys.
    """
    if not portfolio:
        return {}

    result = {}
    for symbol, quantity in portfolio.items():
        canonical = resolve_symbol(symbol)
        if canonical != str(symbol).strip().upper():
            log_replacement(str(symbol).strip().upper(), canonical)
        result[canonical] = result.get(canonical, 0) + int(quantity)
    return dict(sorted(result.items()))


def lookup_quote_price(symbol, quote_per_investment):
    """Find a quote price using current or replaced symbol keys."""
    if not quote_per_investment:
        return None

    resolved = resolve_symbol(symbol)
    for key in (resolved, str(symbol).strip().upper()):
        price = quote_per_investment.get(key)
        if price not in (None, 'N/A', 'None', '', 0):
            return float(price)

    for old, new in load_replacements().items():
        if new == resolved:
            price = quote_per_investment.get(old)
            if price not in (None, 'N/A', 'None', '', 0):
                return float(price)
    return None
