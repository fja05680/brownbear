#!/usr/bin/env bash
# OAuth login for Charles Schwab API; reads ~/schwab/.env and writes ~/schwab/token.json
set -euo pipefail

SCHWAB_DIR="${HOME}/schwab"
ENV_FILE="${SCHWAB_DIR}/.env"
TOKEN_FILE="${SCHWAB_DIR}/token.json"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/venv/bin/activate"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE}" >&2
    echo "Create it with SCHWAB_CLIENT_ID, SCHWAB_CLIENT_SECRET, and SCHWAB_CALLBACK_URL" >&2
    exit 1
fi

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

API_KEY="${SCHWAB_CLIENT_ID:-${SCHWAB_APP_KEY:-}}"
APP_SECRET="${SCHWAB_CLIENT_SECRET:-${SCHWAB_APP_SECRET:-}}"
CALLBACK_URL="${SCHWAB_CALLBACK_URL:-https://127.0.0.1:8182}"
TOKEN_FILE="${SCHWAB_TOKEN_FILE:-${TOKEN_FILE}}"

if [[ -z "${API_KEY}" || -z "${APP_SECRET}" ]]; then
    echo "Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET in ${ENV_FILE}" >&2
    exit 1
fi

if ! command -v schwab-generate-token.py >/dev/null 2>&1; then
    echo "schwab-generate-token.py not found; install with: pip install schwab-py" >&2
    exit 1
fi

mkdir -p "${SCHWAB_DIR}"

echo "Token file: ${TOKEN_FILE}"
echo "Callback URL: ${CALLBACK_URL}"
echo "Opening Schwab login flow..."

schwab-generate-token.py \
    --token_file "${TOKEN_FILE}" \
    --api_key "${API_KEY}" \
    --app_secret "${APP_SECRET}" \
    --callback_url "${CALLBACK_URL}"

echo "Done. Token saved to ${TOKEN_FILE}"
