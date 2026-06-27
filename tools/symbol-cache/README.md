# tools/symbol-cache

Notebooks and reference for [symbol-cache/](../../symbol-cache/) — the repo's
cached Yahoo Finance daily timeseries — and the shared fundamentals table used
by `bb.add_fundamental_columns()`.

Run notebooks from the repo root with the project virtual environment activated.

## symbol-cache directory

`symbol-cache/` at the repo root is working storage, not source code. It holds
one CSV per ticker, read and written by `bb.fetch_timeseries()` and
`bb.compile_timeseries()`. Galaxy `investment-options.ipynb` notebooks populate
it when `refresh_timeseries=True`, then compile selected symbols into
`symbols-timeseries.csv` inside each galaxy folder.

### File layout

```
symbol-cache/
  AAPL.csv
  MSFT.csv
  SPY.csv
  ...
```

Each file is a Yahoo download with a `Date` index and OHLCV columns. Compiled
galaxy timeseries use the `Adj Close` column.

### Refreshing timeseries

From a notebook:

```python
import brownbear as bb

bb.fetch_timeseries(['AAPL', 'MSFT'], refresh=True, throttle_limit=100, wait_time=30)
```

Or run `get-symbol-timeseries.ipynb` or `update-cache-symbols.ipynb` in this
folder.

To refresh **all** cached symbols:

```python
bb.update_cache_symbols()  # re-download every symbol already in the cache
```

To remove symbols:

```python
bb.remove_cache_symbols(['OLD_TICKER'])  # or None to clear the whole cache
```

A full maintainer reset deletes `symbol-cache/` and rebuilds from scratch —
see [tools/update-universe.sh](../update-universe.sh) with `--full`.

### Notes

- Symbols prefixed with `__` in the cache are ignored by metadata and
  fundamentals utilities.
- Yahoo ticker format uses hyphens for share classes (for example `BRK-B`);
  index loaders in `bb.load_*_universe()` apply this conversion automatically.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `get-symbol-timeseries.ipynb` | Download timeseries for a symbol list into `symbol-cache/` |
| `update-cache-symbols.ipynb` | Re-download every symbol already in the cache |
| `get-symbol-metadata.ipynb` | Report start date, end date, and years of data per cached symbol |
| `get-symbol-fundamentals.ipynb` | Build `fundamentals.csv` and `fundamentals_cache.json` from Yahoo |

`get-symbol-fundamentals.ipynb` is **step 3** in [UPDATE](../../UPDATE) and is
run automatically by [tools/update-universe.sh](../update-universe.sh).

## Fundamentals output files

| File | Used by |
|------|---------|
| `fundamentals.csv` | `bb.add_fundamental_columns()` in portfolio analysis |
| `fundamentals_cache.json` | Incremental cache for `bb.get_symbol_fundamentals()` |

Reset the fundamentals cache with `bb.reset_fundamentals_cache()` or
`./tools/update-universe.sh --reset-fundamentals-cache`.

## Related

- [tools/README.md](../README.md) — `update-universe.sh`, Schwab scripts, monthly workflow
- [universe/README.md](../../universe/README.md) — rebuilding `investment-options.csv`
