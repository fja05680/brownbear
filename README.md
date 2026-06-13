<p align="center">
  <img src="images/brownbear-hi.png" width="400" alt="brownbear">
</p>

# brownbear

A portfolio analysis tool for screening stocks, building diversified allocations, and measuring risk-adjusted performance.

Brownbear helps you work through a full portfolio workflow in Jupyter: define an **investment universe** (S&P 500, Dow 30, ETFs, asset classes, and more), compare weighting schemes, visualize allocations and correlations, optionally optimize weights, and size positions using live quotes.

Start with [portfolios/sp500/portfolio.ipynb](portfolios/sp500/portfolio.ipynb).

## What brownbear is good at

- **Multi-level portfolios** — allocate by asset class, asset subclass, and individual securities
- **Correlation-aware analysis** — correlations between asset classes and between holdings inform diversification
- **Flexible weighting** — equal weight, returns, Sharpe ratio, volatility (inverse), or manual percentages
- **Pre-built universes** — Dow 30, S&P 400/500/600, Nasdaq 100, ETFs, state pension examples, and asset-class galaxies under [universe/](universe/)
- **Cached market data** — daily prices from Yahoo Finance stored in [symbol-cache/](symbol-cache/)
- **Portfolio optimizer** — optional constrained optimization from a notebook
- **Trade sizing** — translate target weights into share counts using current quotes

Brownbear complements [pinkfish](https://github.com/fja05680/pinkfish): pinkfish is built for backtesting rule-based strategies on a fixed basket of symbols; brownbear is built for **screening, constructing, and analyzing portfolios** across a broader investment universe.

## Project layout


| Path                                       | Purpose                                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------ |
| [brownbear/](brownbear/)                   | Python package (fetch, analyze, optimize, metrics, symbol cache)                     |
| [universe/](universe/)                     | Investment galaxies — index constituents, correlations, and `investment-options.csv` |
| [portfolios/](portfolios/)                 | Ready-made portfolio notebooks (S&P 500, sectors, asset classes, etc.)               |
| [symbol-cache/](symbol-cache/)             | Cached Yahoo Finance time series (one CSV per symbol)                                |
| [tools/symbol-cache/](tools/symbol-cache/) | Notebooks to refresh cache, fundamentals, and metadata                               |
| [images/](images/)                         | Project artwork                                                                      |


## Installation

Brownbear works on Linux, macOS, and Windows. I recommend a virtual environment.

```bash
git clone https://github.com/fja05680/brownbear.git
cd brownbear
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install setuptools
pip install -e .
```

Dependencies are listed in [requirements.txt](requirements.txt).

## Jupyter

Most workflows are notebooks. After installing brownbear and activating your virtual environment:

```bash
cd brownbear
source venv/bin/activate   # Windows: venv\Scripts\activate
jupyter nbclassic
```

Open [portfolios/](portfolios/), choose a portfolio type, and run `portfolio.ipynb` from top to bottom.

To open a specific example directly:

```bash
jupyter nbclassic portfolios/sp500/portfolio.ipynb
```

If you prefer `jupyter notebook` or JupyterLab, install the extra packages first: `pip install notebook` or `pip install jupyterlab`.

## Quick start — portfolio notebook

1. Open a notebook under [portfolios/](portfolios/) — for example [portfolios/sp500/portfolio.ipynb](portfolios/sp500/portfolio.ipynb).
2. Set `investment_universe` to the galaxy you want (for example `['sp500-galaxy']`).
3. Set `risk_free_rate` if you use Sharpe-based weighting.
4. Choose or define `portfolio_option` — a dict of symbols and target weights.
5. Run all cells to fetch metrics, analyze the portfolio, and view pie charts.
6. Optionally set `run_portfolio_optimizer = True` to run the optimizer.

Each portfolio folder may also include `investment-options.ipynb` for refreshing that universe's data.

## Updating market data

Maintainers refreshing the full dataset should follow [UPDATE](UPDATE). In short:

1. Rebuild [symbol-cache/](symbol-cache/) when needed.
2. Run universe notebooks under [universe/](universe/) to refresh indices and `investment-options.csv`.
3. Run [tools/symbol-cache/get-symbol-fundamentals.ipynb](tools/symbol-cache/get-symbol-fundamentals.ipynb).
4. Run portfolio notebooks.

Run `tools/cleanup-kernels.sh` between notebook runs if Jupyter becomes sluggish.

## Documentation

API reference (generated with [pdoc3](https://pdoc3.github.io/pdoc/)) lives in [docs/html/brownbear/](docs/html/brownbear/index.html).

Regenerate after API changes:

```bash
cd docs
./generate-docs.sh
```

View locally:

```bash
xdg-open docs/html/brownbear/index.html    # Linux
# open docs/html/brownbear/index.html      # macOS
# start docs/html/brownbear/index.html     # Windows
```

Source modules with docstrings are in [brownbear/](brownbear/).

## License

MIT — see [LICENSE](LICENSE).