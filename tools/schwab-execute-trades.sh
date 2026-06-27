#!/usr/bin/env bash
# Place Schwab limit orders from a trades-YYYY-MM-DD.csv file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/venv/bin/activate"
fi

exec python3 "${ROOT}/tools/schwab_execute_trades.py" "$@"
