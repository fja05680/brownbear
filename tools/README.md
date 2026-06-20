# tools

Scripts and notebooks for maintaining brownbear data and running monthly strategy workflows.

## Data maintenance

### `update-universe.sh`

Restartable batch run for [UPDATE](../UPDATE) steps 1–4 (universe galaxies, fundamentals, portfolios).

```bash
./tools/update-universe.sh --full      # first run: clean caches, run all steps
./tools/update-universe.sh             # resume after an error
```

| Flag | Effect |
|------|--------|
| `--full` | `--delete-symbol-cache`, `--reset-fundamentals-cache`, `--reset-progress` |
| `--delete-symbol-cache` | Step 0: delete `symbol-cache/` and clear progress |
| `--reset-fundamentals-cache` | Delete `fundamentals_cache.json` before step 3 |
| `--reset-progress` | Ignore `tools/.update-universe.state` and re-run all steps |

Progress is saved in `tools/.update-universe.state` (gitignored). Completed notebooks are skipped on re-run.

### `cleanup-kernels.sh`

Stops stray Jupyter kernels and clears stale connection files. Run between heavy notebook batches if kernels pile up (also called automatically by `update-universe.sh`).

```bash
./tools/cleanup-kernels.sh
```

### `symbol-cache/`

Notebooks to refresh fundamentals, metadata, and timeseries cache:

- `get-symbol-fundamentals.ipynb` — step 3 in UPDATE (also run by `update-universe.sh`)
- `get-symbol-metadata.ipynb`, `get-symbol-timeseries.ipynb`, `update-cache-symbols.ipynb`

## Schwab API

Credentials and tokens live in **`~/schwab/`** (not in git):

| File | Purpose |
|------|---------|
| `~/schwab/.env` | `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET`, `SCHWAB_CALLBACK_URL` |
| `~/schwab/token.json` | OAuth token (created by `schwab-generate-token.sh`) |
| `~/schwab/accounts.json` | Optional account map (`schwab-account-info.sh --save`) |

Requires `pip install schwab-py` in the project venv.

### `schwab-generate-token.sh`

OAuth login; writes or refreshes `~/schwab/token.json`. Re-run when the refresh token expires (~7 days) or at the start of a monthly rebalance.

```bash
./tools/schwab-generate-token.sh
```

### `schwab-account-info.sh`

Fetch live account value and positions from Schwab.

```bash
./tools/schwab-account-info.sh                              # list accounts
./tools/schwab-account-info.sh 350                          # print summary
./tools/schwab-account-info.sh 350 -o strategies/.../account-2026-06-20.json
./tools/schwab-account-info.sh --save                       # write ~/schwab/accounts.json
```

Strategy notebooks read `account-YYYY-MM-DD.json` for `total_capital` and `current_portfolio`.

### `schwab-execute-trades.sh`

Place market orders from a `trades-YYYY-MM-DD.csv` produced by a strategy notebook.

```bash
# Local check only (no API)
./tools/schwab-execute-trades.sh path/to/trades-2026-06-20.csv

# Schwab validates orders, does not execute (default in notebook when execute_live_trades=False)
./tools/schwab-execute-trades.sh path/to/trades-2026-06-20.csv --preview

# Live orders (prompts for confirmation)
./tools/schwab-execute-trades.sh path/to/trades-2026-06-20.csv --execute

# Live orders, no prompt
./tools/schwab-execute-trades.sh path/to/trades-2026-06-20.csv --execute --yes
```

| Mode | API calls | Orders placed |
|------|-----------|---------------|
| default (dry-run) | None | No |
| `--preview` | Yes | No |
| `--execute` | Yes | Yes |

Writes a log file beside the CSV, e.g. `trades-2026-06-20-preview-133648.json`.

### Python modules

| Module | Used by |
|--------|---------|
| `schwab_lib.py` | Shared client, env, account matching |
| `schwab_account_info.py` | `schwab-account-info.sh` |
| `schwab_execute_trades.py` | `schwab-execute-trades.sh` |

## Monthly strategy workflow (example: 350 IRA)

From the repo root, with venv activated:

```bash
# 1. Refresh market data (optional, periodic)
./tools/update-universe.sh

# 2. Schwab token if needed
./tools/schwab-generate-token.sh

# 3. Run strategy notebook manually
#    - fetches account-YYYY-MM-DD.json from Schwab
#    - generates trades-YYYY-MM-DD.csv
#    - previews orders (execute_live_trades = False)
#    - set execute_live_trades = True to place live orders
```

Strategy notebooks live under `strategies/` (gitignored).
