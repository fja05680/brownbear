#!/usr/bin/env python3
"""Place Schwab equity orders from a trades CSV produced by brownbear."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from schwab.orders import equities
from schwab.utils import Utils

from schwab_lib import fetch_account_numbers, make_client, match_account

ORDER_DELAY_SEC = 1.0


def load_trades(path):
    df = pd.read_csv(path)
    required = {"account", "side", "symbol", "quantity"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing columns in {path}: {', '.join(sorted(missing))}")

    df["side"] = df["side"].str.upper()
    invalid = set(df["side"]) - {"BUY", "SELL"}
    if invalid:
        sys.exit(f"Invalid side values: {', '.join(sorted(invalid))}")

    accounts = df["account"].astype(str).unique()
    if len(accounts) != 1:
        sys.exit(f"Expected one account per CSV, found: {', '.join(accounts)}")

    df["quantity"] = df["quantity"].astype(int)
    df["_side_order"] = df["side"].map({"SELL": 0, "BUY": 1})
    return df.sort_values(["_side_order", "symbol"]).drop(columns="_side_order")


def build_order(side, symbol, quantity):
    if side == "SELL":
        return equities.equity_sell_market(symbol, quantity)
    return equities.equity_buy_market(symbol, quantity)


def confirm_execute(count, account_suffix):
    prompt = f"Place {count} live order(s) on account {account_suffix}? [y/N] "
    answer = input(prompt).strip().lower()
    if answer not in {"y", "yes"}:
        sys.exit("Aborted.")


def _write_log(trades_path, mode, account_suffix, account_row, account_hash, results):
    log_path = trades_path.with_name(
        trades_path.stem + f"-{mode}-{datetime.now().strftime('%H%M%S')}.json"
    )
    payload = {
        "trades_file": str(trades_path),
        "mode": mode,
        "account_suffix": account_suffix,
        "results": results,
    }
    if account_row is not None:
        payload["account_number"] = account_row["accountNumber"]
        payload["account_hash"] = account_hash
    log_path.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"Log: {log_path}")


def execute_trades(trades_path, execute=False, preview=False, yes=False, delay=ORDER_DELAY_SEC):
    trades_path = Path(trades_path)
    if not trades_path.is_file():
        sys.exit(f"Trades file not found: {trades_path}")

    df = load_trades(trades_path)
    account_suffix = str(df["account"].iloc[0])
    mode = "execute" if execute else "preview" if preview else "dry-run"
    results = []

    print(f"Mode: {mode}")
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
        _write_log(trades_path, mode, account_suffix, None, None, results)
        return

    client = make_client()
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
        label = f"{side} {quantity} {symbol}"

        order = build_order(side, symbol, quantity)
        if preview:
            response = client.preview_order(account_hash, order)
            action = "preview"
        else:
            response = client.place_order(account_hash, order)
            action = "placed"

        if response.is_error:
            print(f"[error] {label}: {response.status_code} {response.text}")
            results.append({
                "side": side,
                "symbol": symbol,
                "quantity": quantity,
                "status": "error",
                "http_status": response.status_code,
                "message": response.text,
            })
            sys.exit(1)

        order_id = None
        if execute:
            try:
                order_id = utils.extract_order_id(response)
            except Exception as exc:
                print(f"[warning] {label}: could not parse order id: {exc}")

        print(f"[{action}] {label}" + (f" order_id={order_id}" if order_id else ""))
        results.append({
            "side": side,
            "symbol": symbol,
            "quantity": quantity,
            "status": action,
            "order_id": order_id,
        })

        if i < len(df) - 1 and delay:
            time.sleep(delay)

    _write_log(trades_path, mode, account_suffix, account_row, account_hash, results)


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
        help="Place live market orders (default is dry-run)",
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
    args = parser.parse_args()

    execute_trades(
        args.trades_file,
        execute=args.execute,
        preview=args.preview,
        yes=args.yes,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
