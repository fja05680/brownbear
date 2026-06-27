# universe

Investment **galaxies** — curated sets of securities, asset-class correlations,
and the `investment-options.csv` files that `bb.fetch()` loads.

Each galaxy is a folder under `universe/` (for example `sp500-galaxy`,
`extra-auto-galaxy`, `asset-class-galaxy`).

## Galaxies

| Folder | Type | Notes |
|--------|------|-------|
| `asset-class-galaxy` | Asset classes | Builds `asset-classes.csv` and `investment-options.csv` via `asset-classes.ipynb` |
| `dow30-galaxy` | Index | Constituents from `dow30.csv` |
| `sp400-galaxy` | Index | S&P MidCap 400 |
| `sp500-galaxy` | Index | S&P 500 |
| `sp600-galaxy` | Index | S&P SmallCap 600 |
| `nasdaq100-galaxy` | Index | Nasdaq 100 |
| `etf-galaxy` | Manual list | ETFs in `investment-options-in.csv` |
| `alabama-galaxy` | Manual list | Alabama pension holdings |
| `extra-auto-galaxy` | Manual list | Reference pattern for custom symbol lists |
| `extra-manual-galaxy` | Manual list | Metrics filled in by hand |

Index galaxies also contain constituent CSVs (`sp500.csv`, `gics-2-asset-class.csv`,
etc.) downloaded or maintained alongside the notebook.

## Investment options

`investment-options.csv` is the performance table brownbear uses for screening,
weighting, and analysis. `bb.fetch(['sp500-galaxy'])` reads it from the matching
galaxy folder.

### Files in each galaxy

| File | Role |
|------|------|
| `investment-options-in.csv` | **Edit this** — symbols, descriptions, asset classes; metric columns left blank |
| `investment-options.csv` | **Generated** — same rows with returns, volatility, and standard deviation filled in |
| `investment-options.ipynb` | Notebook that rebuilds `investment-options.csv` |
| `symbols-timeseries.csv` | Compiled daily closes used to compute metrics (written by the notebook) |

`investment-options-in.csv` format:

```csv
# Description: S&P 500 Galaxy

# Format
"Investment Option","Description","Asset Class","3 mo","6 mo","1 Yr","1-1 Yr","3 Yr","5 Yr","Vola","DS Vola","SD 1 Yr","SD 3 Yr","SD 5 Yr"

"AAPL","Apple Inc.","US Stocks:Information Technology","","","","","","","","","","",""
```

Comment lines (`# ...`) and the header row are preserved in the output file.

### How to rebuild `investment-options.csv`

1. Open the galaxy folder and edit `investment-options-in.csv` if you are adding or
   removing symbols (manual galaxies only; index galaxies take symbols from the
   constituent CSV).
2. Open `investment-options.ipynb` in Jupyter.
3. Set notebook settings:
   - `refresh_timeseries = True` — download fresh Yahoo prices into
     [symbol-cache/](../symbol-cache/)
   - `refresh_timeseries = False` — use cached prices (normal for a quick rebuild)
   - `wait_time = 30` — pause after every `throttle_limit` downloads (default 100)
4. Run all cells. The notebook fetches timeseries, computes metrics, and writes
   `investment-options.csv`.

### Notebook patterns

**Manual symbol list** (`extra-auto-galaxy`, `etf-galaxy`, `alabama-galaxy`,
`extra-manual-galaxy`):

```python
import brownbear as bb

investment_options = bb.update_investment_options(
    directory='.',
    refresh_timeseries=refresh_timeseries,
    throttle_limit=throttle_limit,
    wait_time=wait_time,
)
```

**Index constituents** (`sp500-galaxy`, `dow30-galaxy`, etc.):

```python
import brownbear as bb

directory = '.'
universe = bb.load_sp500_universe(directory)  # or load_dow30_universe, etc.

investment_options = bb.update_investment_options(
    directory=directory,
    refresh_timeseries=refresh_timeseries,
    throttle_limit=throttle_limit,
    wait_time=wait_time,
    universe_df=universe,
)
```

**Asset class galaxy** — use `asset-classes.ipynb` instead. It builds
`asset-classes.csv` (correlations) and `investment-options.csv` (one row per
asset class, keyed off representative ETFs in `asset-classes-2-etf.csv`).

### API

See `bb.update_investment_options()` and `bb.compute_investment_metrics()` in
[brownbear/investment_options.py](../brownbear/investment_options.py).

## Maintainer workflow

Galaxy notebooks are run in bulk by [tools/update-universe.sh](../tools/update-universe.sh).
See [UPDATE](../UPDATE) for the full step order. Run `asset-class-galaxy/asset-classes.ipynb`
first, then the remaining universe notebooks.
