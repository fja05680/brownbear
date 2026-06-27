# portfolios

Ready-made **portfolio notebooks** — examples of `bb.fetch()`, `bb.analyze()`,
pie charts, correlation heatmaps, and the optional optimizer.

Each subfolder is a self-contained example (for example `sp500/`, `dow30/`,
`asset-classes/`). Open `portfolio.ipynb` and run all cells.

## Typical workflow

1. Choose a portfolio folder and open `portfolio.ipynb`.
2. Set `investment_universe` to one or more galaxy names under
   [universe/](../universe/) (for example `['sp500-galaxy']`).
3. Define `portfolio_option` — a dict of symbols and weights (first entry is the
   portfolio title).
4. Run all cells to load metrics, analyze the portfolio, and view charts.
5. Optionally set `run_portfolio_optimizer = True` to run the optimizer.

Portfolio notebooks **do not** maintain market data themselves. They read
`investment-options.csv` from the galaxies you list in `investment_universe`.

## Investment options

Most portfolio folders only contain `portfolio.ipynb`. Metrics come from
[universe/](../universe/) galaxies — rebuild those with each galaxy's
`investment-options.ipynb` when prices need refreshing.

See [universe/README.md](../universe/README.md) for the full guide to
`investment-options-in.csv`, `investment-options.csv`, and the rebuild notebooks.

### When a portfolio folder has its own investment options

Use this only when the holdings are **not** already covered by a universe galaxy.

| Folder | Files | Pattern |
|--------|-------|---------|
| `florida-retirement-system/` | `investment-options.csv`, `investment-options.ipynb` | **Manual** — edit `investment-options.csv` by hand (returns, vol, std dev), then run the notebook to review |

`florida-retirement-system` does not use `investment-options-in.csv` or
`bb.update_investment_options()`. Update the CSV directly from your source data,
then run `investment-options.ipynb` to display the table.

To add a similar portfolio-specific universe:

1. Copy the pattern from [universe/extra-auto-galaxy/](../universe/extra-auto-galaxy/)
   (`investment-options-in.csv` + `investment-options.ipynb`).
2. Reference the folder name in `investment_universe` inside `portfolio.ipynb`:

   ```python
   investment_universe = ['sp500-galaxy', 'florida-retirement-system']
   ```

   `bb.fetch()` looks under `universe/`, `portfolios/`, and `strategies/` for
   `investment-options.csv`.

## Portfolio folders

| Folder | `investment_universe` (typical) |
|--------|----------------------------------|
| `sp500` | `sp500-galaxy` |
| `sp400` | `sp400-galaxy` |
| `sp600` | `sp600-galaxy` |
| `dow30` | `dow30-galaxy` |
| `nasdaq100` | `nasdaq100-galaxy` |
| `etf` | `etf-galaxy` |
| `etf-sp500` | `etf-galaxy`, `sp500-galaxy` |
| `asset-classes` | `asset-class-galaxy` |
| `global-asset-allocation` | `asset-class-galaxy` |
| `sp500-sectors` | `sp500-galaxy` |
| `alabama` | `alabama-galaxy` |
| `total-market` | Multiple index galaxies |
| `florida-retirement-system` | Local `investment-options.csv` |
