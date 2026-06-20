"""Shared Schwab API helpers for brownbear tools."""

import sys
from pathlib import Path

from schwab.auth import client_from_token_file

SCHWAB_DIR = Path.home() / "schwab"
ENV_FILE = SCHWAB_DIR / ".env"
TOKEN_FILE = SCHWAB_DIR / "token.json"
ACCOUNTS_FILE = SCHWAB_DIR / "accounts.json"


def load_env(path):
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def make_client():
    if not ENV_FILE.is_file():
        sys.exit(f"Missing {ENV_FILE}")
    if not TOKEN_FILE.is_file():
        sys.exit(f"Missing {TOKEN_FILE}; run tools/schwab-generate-token.sh first")

    env = load_env(ENV_FILE)
    api_key = env.get("SCHWAB_CLIENT_ID") or env.get("SCHWAB_APP_KEY")
    app_secret = env.get("SCHWAB_CLIENT_SECRET") or env.get("SCHWAB_APP_SECRET")
    if not api_key or not app_secret:
        sys.exit(f"Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET in {ENV_FILE}")

    return client_from_token_file(str(TOKEN_FILE), api_key, app_secret)


def fetch_account_numbers(client):
    response = client.get_account_numbers()
    if response.status_code != 200:
        sys.exit(f"get_account_numbers failed: {response.status_code} {response.text}")
    return response.json()


def match_account(accounts, suffix):
    matches = [
        row for row in accounts
        if row["accountNumber"].endswith(str(suffix))
    ]
    if not matches:
        sys.exit(f"No account ending in {suffix!r}")
    if len(matches) > 1:
        nums = ", ".join(row["accountNumber"] for row in matches)
        sys.exit(f"Multiple accounts match {suffix!r}: {nums}")
    return matches[0]
