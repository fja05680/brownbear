brownbear
======

brownbear is a tool to help you screen for stocks to build a portfolio,
analyze past performance of the portfolio you constructed, and
calculate the shares needed to build the portfolio using real time quotes.

brownbear is a very sophisticated tool that takes into account
the correlations between asset classes, asset sub-classes,
and individual stocks.

The shares allocated to each investment choice can be based on
returns, sharp ratio, volatility, equal, or manually specified 
using percentages.  Allocations can be specified by asset class,
asset sub-class, and individual stocks.

brownbear could give you the edge to beat the professionals :)
All for free!!

Install

    $ git clone https://github.com/fja05680/brownbear.git
    $ cd brownbear
    $ python -m venv venv
    $ pip install setuptools
    $ . venv/bin/activate
    $ python setup.py install (or develop)

How to guide

    - Within jupyter notebook, naviate to brownbear/portfolios
    - Click the folder for the portfolio type you are interested in analyzing.
    - Open portfolio.ipynb.
    - Assign the variables investment_universe and risk_free_rate.
    - Specify custom portfolios using your investment options.
    - Assign the variable portfolio_option to point to the portfolio you wish to analyze
    - Run the notebook
    - Optionally, run the optimizer by setting run_portfolio_optimizer to True
    
