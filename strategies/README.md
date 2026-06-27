# Strategies

Per-account portfolio notebooks live here. **Strategy folders are not
committed** — only this README is tracked.

Most strategies only need a `portfolio.ipynb` and pull their investment
universe from `universe/` galaxies (for example `sp500-galaxy`,
`extra-auto-galaxy`). That is the normal setup.

## Optional: strategy-specific investment options

`bb.fetch()` can also read `strategies/<name>/investment-options.csv`, but
you usually do **not** need this. Add it only when a symbol you need is not
already in one of your galaxies — for example cash (`BIL`) when you are not
including `etf-galaxy`.

1. Create `investment-options-in.csv` and `investment-options.ipynb` in your
   strategy folder (see [universe/README.md](../universe/README.md) and
   `universe/extra-auto-galaxy/` for the pattern). Edit the CSV, run the
   notebook to build `investment-options.csv`.

2. Add the folder name to `investment_universe` in `portfolio.ipynb`:

   ```python
   investment_universe = [
       'sp500-galaxy',
       'my-strategy (123 IRA)',
   ]
   ```

## Git

- Tracked: `strategies/README.md`
- Ignored: everything else under `strategies/` (notebooks, account JSON,
  trade CSVs, and any optional investment-options files)
