"""Limit price selection from bid/ask/last and spread tolerance."""

from dataclasses import dataclass

PRICING_STRATEGIES = ('aggressive', 'pennying', 'midpoint')
PENNY_TICK = 0.01


@dataclass(frozen=True)
class LimitPriceResult:
    price: float
    spread_pct: float
    strategy: str
    non_marketable: bool
    bid: float
    ask: float
    last: float


def spread_pct(bid, ask):
    """Bid-ask spread as a fraction of the ask."""
    if ask <= 0:
        raise ValueError(f'Invalid ask price: {ask}')
    if bid < 0:
        raise ValueError(f'Invalid bid price: {bid}')
    return (ask - bid) / ask


def round_price(price):
    """Round to Schwab-friendly tick precision."""
    if price >= 1:
        return round(price, 2)
    return round(price, 4)


def compute_limit_price(
    side,
    bid,
    ask,
    last,
    strategy,
    max_spread_tolerance,
):
    """
    Compute a day limit price from live quote data.

    When spread exceeds ``max_spread_tolerance``, fall back to ``last`` as a
    standard day limit (avoids lifting a wide ask or hitting a wide bid).

    Strategies (tight spread only):
    - aggressive: buy at ask, sell at bid
    - pennying: buy at bid + $0.01, sell at ask - $0.01
    - midpoint: (bid + ask) / 2
    """
    side = side.upper()
    if side not in ('BUY', 'SELL'):
        raise ValueError(f'Invalid side: {side}')
    if strategy not in PRICING_STRATEGIES:
        raise ValueError(
            f"Invalid strategy {strategy!r}; choose from {PRICING_STRATEGIES}"
        )

    spread = spread_pct(bid, ask)
    if spread > max_spread_tolerance:
        fallback = last if last > 0 else (bid + ask) / 2
        return LimitPriceResult(
            price=round_price(fallback),
            spread_pct=spread,
            strategy='wide_spread_fallback',
            non_marketable=False,
            bid=bid,
            ask=ask,
            last=last,
        )

    if strategy == 'aggressive':
        price = ask if side == 'BUY' else bid
    elif strategy == 'pennying':
        if side == 'BUY':
            price = min(ask, bid + PENNY_TICK)
        else:
            price = max(bid, ask - PENNY_TICK)
    else:  # midpoint
        price = (bid + ask) / 2

    return LimitPriceResult(
        price=round_price(price),
        spread_pct=spread,
        strategy=strategy,
        non_marketable=False,
        bid=bid,
        ask=ask,
        last=last,
    )


def parse_equity_quote(symbol, payload):
    """Extract bid, ask, and last from a Schwab get_quotes response entry."""
    if not payload or 'error' in payload:
        raise ValueError(f'No quote data for {symbol}')

    quote = payload.get('quote') or payload.get('regular') or payload
    bid = quote.get('bidPrice')
    ask = quote.get('askPrice')
    last = quote.get('lastPrice') or quote.get('mark')

    missing = [
        name for name, value in (('bid', bid), ('ask', ask), ('last', last))
        if value is None or value <= 0
    ]
    if missing:
        raise ValueError(
            f'Missing or invalid quote fields for {symbol}: {", ".join(missing)}'
        )

    return float(bid), float(ask), float(last)
