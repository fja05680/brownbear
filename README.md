brownbear
======

brownbear is a tool to help you screen for stocks to build a portfolio,
analyze past performance of the portfolio you constructed, and
calculate the shares need to build the portfolio using real time quotes.

brownbear is a very sophisticated tool that takes into account
asset classes, asset sub-classes, individual stocks, and correlations
between any combination of these.

The shares allocated to each investment choise can be manual, or
adjust based on returns, sharp ratio, volatility, or equal.

brownbear could give you the edge to beat the professionals :)
All for free!!

Install

    - git clone https://github.com/fja05680/brownbear.git
    - cd brownbear
    - sudo python setup.py install

How to guide

    - Within jupyter notebook, naviate to brownbear/portfolios
    - Click the folder for the portfolio type you are interested in analyzing.
    - Open portfolio.ipynb.
    - Assign the variables investment_universe and risk_free_rate.
    - Specify custom portfolios using your investment options.
    - Assign the variable portfolio_option to point to the portfolio you wish to analyze
    - Run the notebook
    - Optionally, run the optimizer by setting run_portfolio_optimizer to True
    
