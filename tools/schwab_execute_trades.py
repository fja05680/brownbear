#!/usr/bin/env python3
"""Place Schwab equity limit orders from a trades CSV produced by brownbear."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from schwab.orders.common import (
    Duration,
    EquityInstruction,
    OrderStrategyType,
    OrderType,
    Session,
)
from schwab.orders.generic import OrderBuilder
from schwab.utils import Utils

from limit_pricing import (
    PRICING_STRATEGIES,
    compute_limit_price,
    parse_equity_quote,
)
from schwab_lib import fetch_account_numbers, make_client, match_account
from symbol_replacements import log_replacement, resolve_symbol

ORDER_DELAY_SEC = 1.0
DEFAULT_PRICING_STRATEGY = 'aggressive'
DEFAULT_MAX_SPREAD_TOLERANCE = 0.01
TRADE_COLUMNS = ("account", "side", "symbol", "quantity")


def load_trades(path):
    path = Path(path)
    required = set(TRADE_COLUMNS)
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing columns in {path}: {', '.join(sorted(missing))}")

    if df.empty:
        return df

    df["side"] = df["side"].str.upper()
    invalid = set(df["side"]) - {"BUY", "SELL"}
    if invalid:
        sys.exit(f"Invalid side values: {', '.join(sorted(invalid))}")

    accounts = df["account"].astype(str).unique()
    if len(accounts) != 1:
        sys.exit(f"Expected one account per CSV, found: {', '.join(accounts)}")

    df["quantity"] = df["quantity"].astype(int)
    df["_side_order"] = df["side"].map({"SELL": 0, "BUY": 1})
    df = df.sort_values(["_side_order", "symbol"]).drop(columns="_side_order")

    resolved_symbols = []
    for symbol in df['symbol']:
        original = str(symbol).strip().upper()
        trade_symbol = resolve_symbol(original)
        if trade_symbol != original:
            log_replacement(original, trade_symbol)
        resolved_symbols.append(trade_symbol)
    df['symbol'] = resolved_symbols
    return df


def infer_account_suffix(trades_path):
    """Read account suffix from a sibling account-YYYY-MM-DD.json, if present."""
    trades_path = Path(trades_path)
    stem = trades_path.stem
    if not stem.startswith('trades-'):
        return None

    account_path = trades_path.with_name(f"account-{stem[len('trades-'):]}.json")
    if not account_path.is_file():
        return None

    try:
        payload = json.loads(account_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    suffix = payload.get('account_suffix')
    return str(suffix) if suffix else None


def fetch_quotes(client, symbols):
    response = client.get_quotes(list(symbols))
    if response.status_code != 200:
        sys.exit(f'get_quotes failed: {response.status_code} {response.text}')
    return response.json()


def format_order_price(price):
    """Format price for Schwab OrderBuilder (string, not float)."""
    if price >= 1:
        return f'{price:.2f}'
    return f'{price:.4f}'


def build_limit_order(side, symbol, quantity, price, non_marketable=False):
    instruction = (
        EquityInstruction.SELL if side == 'SELL' else EquityInstruction.BUY
    )
    if non_marketable:
        order_type = OrderType.NON_MARKETABLE
        builder = (
            OrderBuilder()
            .set_order_type(order_type)
            .set_session(Session.NORMAL)
            .set_duration(Duration.DAY)
            .set_order_strategy_type(OrderStrategyType.SINGLE)
            .add_equity_leg(instruction, symbol, quantity)
        )
        return builder

    return (
        OrderBuilder()
        .set_order_type(OrderType.LIMIT)
        .set_session(Session.NORMAL)
        .set_duration(Duration.DAY)
        .set_order_strategy_type(OrderStrategyType.SINGLE)
        .set_price(format_order_price(price))
        .add_equity_leg(instruction, symbol, quantity)
    )


def format_label(side, quantity, symbol, pricing):
    order_type = 'NON_MARKETABLE' if pricing.non_marketable else 'LIMIT'
    return (
        f'{side} {quantity} {symbol} @ {pricing.price:.2f} '
        f'({pricing.strategy}, {order_type}, '
        f'spread={pricing.spread_pct * 100:.2f}%)'
    )


def result_row(side, symbol, quantity, pricing, status, order_id=None, **extra):
    row = {
        'side': side,
        'symbol': symbol,
        'quantity': quantity,
        'limit_price': pricing.price,
        'pricing_strategy': pricing.strategy,
        'non_marketable': pricing.non_marketable,
        'bid': pricing.bid,
        'ask': pricing.ask,
        'last': pricing.last,
        'spread_pct': pricing.spread_pct,
        'status': status,
        'order_id': order_id,
    }
    row.update(extra)
    return row


def confirm_execute(count, account_suffix):
    prompt = f"Place {count} live order(s) on account {account_suffix}? [y/N] "
    answer = input(prompt).strip().lower()
    if answer not in {"y", "yes"}:
        sys.exit("Aborted.")


def _write_log(trades_path, mode, account_suffix, account_row, account_hash, results, **meta):
    log_path = trades_path.with_name(
        trades_path.stem + f"-{mode}-{datetime.now().strftime('%H%M%S')}.json"
    )
    payload = {
        "trades_file": str(trades_path),
        "mode": mode,
        "account_suffix": account_suffix,
        "results": results,
        **meta,
    }
    if account_row is not None:
        payload["account_number"] = account_row["accountNumber"]
        payload["account_hash"] = account_hash
    log_path.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"Log: {log_path}")


def execute_trades(
    trades_path,
    execute=False,
    preview=False,
    yes=False,
    delay=ORDER_DELAY_SEC,
    pricing_strategy=DEFAULT_PRICING_STRATEGY,
    max_spread_tolerance=DEFAULT_MAX_SPREAD_TOLERANCE,
):
    trades_path = Path(trades_path)
    if not trades_path.is_file():
        sys.exit(f"Trades file not found: {trades_path}")

    df = load_trades(trades_path)
    mode = "execute" if execute else "preview" if preview else "dry-run"
    results = []
    log_meta = {
        'pricing_strategy': pricing_strategy,
        'max_spread_tolerance': max_spread_tolerance,
    }

    if df.empty:
        account_suffix = infer_account_suffix(trades_path) or 'unknown'
        print(f"Mode: {mode}")
        print(f"Pricing strategy: {pricing_strategy}")
        print(f"Max spread tolerance: {max_spread_tolerance * 100:.2f}%")
        print(f"Account suffix: {account_suffix}")
        print(f"Trades file: {trades_path}")
        print(f"Orders: 0")
        print()
        print("No orders to place.")
        _write_log(
            trades_path, mode, account_suffix, None, None, results, **log_meta,
        )
        return

    account_suffix = str(df["account"].iloc[0])
    print(f"Mode: {mode}")
    print(f"Pricing strategy: {pricing_strategy}")
    print(f"Max spread tolerance: {max_spread_tolerance * 100:.2f}%")
    print(f"Account suffix: {account_suffix}")
    print(f"Trades file: {trades_path}")
    print(f"Orders: {len(df)}")
    print()

    if mode == "dry-run":
        for row in df.itertuples(index=False):
            label = f"{row.side} {int(row.quantity)} {row.symbol}"
            print(f"[dry-run] {label}")
            results.append({
                "side": row.side,
                "symbol": row.symbol,
                "quantity": int(row.quantity),
                "status": "dry-run",
            })
        _write_log(trades_path, mode, account_suffix, None, None, results, **log_meta)
        return

    client = make_client()
    symbols = sorted(df['symbol'].unique())
    quotes = fetch_quotes(client, symbols)

    accounts = fetch_account_numbers(client)
    account_row = match_account(accounts, account_suffix)
    account_hash = account_row["hashValue"]
    utils = Utils(client, account_hash)

    if execute and not yes:
        confirm_execute(len(df), account_suffix)

    print(f"Account number: {account_row['accountNumber']}")
    print()

    for i, row in enumerate(df.itertuples(index=False)):
        side = row.side
        symbol = row.symbol
        quantity = int(row.quantity)
        bid, ask, last = parse_equity_quote(symbol, quotes.get(symbol, {}))
        pricing = compute_limit_price(
            side, bid, ask, last, pricing_strategy, max_spread_tolerance,
        )
        label = format_label(side, quantity, symbol, pricing)

        order = build_limit_order(
            side, symbol, quantity, pricing.price, pricing.non_marketable,
        )
        if preview:
            response = client.preview_order(account_hash, order)
            action = "preview"
        else:
            response = client.place_order(account_hash, order)
            action = "placed"

        if response.is_error:
            print(f"[error] {label}: {response.status_code} {response.text}")
            results.append(result_row(
                side, symbol, quantity, pricing, 'error',
                http_status=response.status_code,
                message=response.text,
            ))
            sys.exit(1)

        order_id = None
        if execute:
            try:
                order_id = utils.extract_order_id(response)
            except Exception as exc:
                print(f"[warning] {label}: could not parse order id: {exc}")

        print(f"[{action}] {label}" + (f" order_id={order_id}" if order_id else ""))
        results.append(result_row(
            side, symbol, quantity, pricing, action, order_id=order_id,
        ))

        if i < len(df) - 1 and delay:
            time.sleep(delay)

    _write_log(
        trades_path, mode, account_suffix, account_row, account_hash, results,
        **log_meta,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Execute trades from a brownbear trades CSV via Schwab API",
    )
    parser.add_argument(
        "trades_file",
        help="Path to trades-YYYY-MM-DD.csv",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Place live limit orders (default is dry-run)",
    )
    mode.add_argument(
        "--preview",
        action="store_true",
        help="Call Schwab preview API for each order (no execution)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt with --execute",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=ORDER_DELAY_SEC,
        help=f"Seconds between orders (default {ORDER_DELAY_SEC})",
    )
    parser.add_argument(
        "--pricing-strategy",
        choices=PRICING_STRATEGIES,
        default=DEFAULT_PRICING_STRATEGY,
        help=(
            "Limit pricing when spread is within tolerance (default: aggressive): "
            "aggressive (buy ask / sell bid), pennying (bid+0.01 / ask-0.01), "
            "midpoint"
        ),
    )
    parser.add_argument(
        "--max-spread-tolerance",
        type=float,
        default=DEFAULT_MAX_SPREAD_TOLERANCE,
        help=(
            "Max bid-ask spread as fraction of ask before wide-spread fallback "
            f"(default {DEFAULT_MAX_SPREAD_TOLERANCE} = 1%%)"
        ),
    )
    args = parser.parse_args()

    execute_trades(
        args.trades_file,
        execute=args.execute,
        preview=args.preview,
        yes=args.yes,
        delay=args.delay,
        pricing_strategy=args.pricing_strategy,
        max_spread_tolerance=args.max_spread_tolerance,
    )


if __name__ == "__main__":
    main()
