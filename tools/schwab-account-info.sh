#!/usr/bin/env bash
# Show Schwab accounts/positions using ~/schwab/.env and ~/schwab/token.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/venv/bin/activate"
fi

exec python3 "${ROOT}/tools/schwab_account_info.py" "$@"
