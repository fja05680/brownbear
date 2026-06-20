#!/usr/bin/env python3
"""List Schwab accounts and show positions/balance for a strategy account suffix."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from schwab.client import Client

from schwab_lib import (
    ACCOUNTS_FILE,
    fetch_account_numbers,
    make_client,
    match_account,
)


def fetch_account(client, account_hash):
    response = client.get_account(
        account_hash,
        fields=[Client.Account.Fields.POSITIONS],
    )
    if response.status_code != 200:
        sys.exit(f"get_account failed: {response.status_code} {response.text}")
    return response.json().get("securitiesAccount", response.json())


def positions_dict(positions):
    portfolio = {}
    for row in positions:
        symbol = row["instrument"]["symbol"]
        quantity = int(row["longQuantity"])
        if quantity > 0:
            portfolio[symbol] = quantity
    return dict(sorted(portfolio.items()))


def save_accounts_map(accounts):
    mapping = {}
    for row in accounts:
        number = row["accountNumber"]
        mapping[number[-3:]] = {
            "accountNumber": number,
            "hashValue": row["hashValue"],
        }
    ACCOUNTS_FILE.write_text(json.dumps(mapping, indent=2) + "\n")
    return ACCOUNTS_FILE


def list_accounts(accounts):
    print(f"{'suffix':<8} {'accountNumber':<12} hashValue")
    print("-" * 72)
    for row in accounts:
        number = row["accountNumber"]
        print(f"{number[-3:]:<8} {number:<12} {row['hashValue']}")


def format_portfolio(portfolio):
    if not portfolio:
        return "{}"
    lines = ["{"]
    items = list(portfolio.items())
    for index, (symbol, quantity) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        lines.append(f"    '{symbol}': {quantity}{comma}")
    lines.append("}")
    return "\n".join(lines)


def build_account_payload(client, accounts, suffix):
    row = match_account(accounts, suffix)
    account = fetch_account(client, row["hashValue"])
    balances = account.get("currentBalances", {})
    return {
        "account_suffix": suffix,
        "account_number": row["accountNumber"],
        "hash_value": row["hashValue"],
        "as_of": date.today().isoformat(),
        "total_capital": balances.get("liquidationValue"),
        "cash_balance": balances.get("cashBalance"),
        "current_portfolio": positions_dict(account.get("positions", [])),
    }


def write_account_json(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def show_account_payload(payload):
    portfolio = payload["current_portfolio"]
    print(f"accountNumber: {payload['account_number']}")
    print(f"hashValue:     {payload['hash_value']}")
    print(f"liquidationValue: {payload['total_capital']}")
    print(f"cashBalance:      {payload['cash_balance']}")
    print()
    print("current_portfolio = \\")
    print(format_portfolio(portfolio))
    print()
    print("# notebook globals")
    print(f"total_capital = {payload['total_capital']}")
    print(f"capital = total_capital")


def main():
    parser = argparse.ArgumentParser(description="Schwab account lookup for strategy notebooks")
    parser.add_argument(
        "account",
        nargs="?",
        help="Account suffix from strategy folder, e.g. 350, 911, 336, 453",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write account data JSON for the notebook (e.g. strategies/.../account-2026-06-20.json)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Write {ACCOUNTS_FILE} (suffix -> accountNumber + hash)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only print output path when writing JSON",
    )
    args = parser.parse_args()

    client = make_client()
    accounts = fetch_account_numbers(client)

    if args.save:
        path = save_accounts_map(accounts)
        print(f"Wrote {path}")

    if args.account:
        payload = build_account_payload(client, accounts, args.account)
        if args.output:
            path = write_account_json(payload, args.output)
            if args.quiet:
                print(path)
            else:
                print(f"Wrote {path}")
        if not args.quiet:
            show_account_payload(payload)
    elif not args.save:
        list_accounts(accounts)
        print()
        print('Example: tools/schwab-account-info.sh 350 -o "strategies/asset-allocation-portfolio (350 IRA)/account-2026-06-20.json"')


if __name__ == "__main__":
    main()
