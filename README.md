brownbear
======

A financial tool that can analyze and maximize investment portfolios on a risk adjusted basis

Features

    - Determines the highest performing Investment Options for each Asset Class, based on risk
      adjusted returns.
    - Calculates Performance Metrics for User Specified Portfolios
    - Optimizes a portfolio for risk adjusted returns with an option to specify the minimum annual
      return rate

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

Annual Returns, Risk, and Risk Adjusted Returns

    - Annual Returns are the equivalent annual return an investor receives over a given period,
      expressed as a percentage.
    - Risk refers to the degree of uncertainty and/or potential financial loss inherent in an
      investment, expressed as annualized Standard Deviation of the returns.
    - Risk Adjusted Return is a calculation of the profit or potential profit from an investment
      that takes into account the degree of risk that must be accepted in order to achieve it,
      expressed as the Sharpe Ratio. The risk is measured in comparison to that of a virtually
      risk-free investment. U.S. Treasuries are usually used for comparison.

Diversification

"By building a portfolio invested in multiple uncorrelated assets an investor can achieve better
returns per unit of risk (Sharpe ratio) due to portfolio diversification". This is accomplished
by choosing investments from different asset classes.

Annualized Returns

Use 5 year returns because some of the investment option returns are "hypothetical" beyond that,
i.e. some of the 10 year mark returns are simulated based on modeling and not real returns.
The funds may not have existed beyond 5 years. The returns should be already net of expenses,
so no need to subtract them.

Risk Free Rate

"Investors commonly use the interest rate on a three-month U.S. Treasury bill (T-bill) as a proxy
for the short-term risk-free rate because short-term government-issued securities have virtually
zero risks of default, as they are backed by the full faith and credit of the U.S. government."
If you don't have a short-term treasury option available, then use any investment option in your
account that has a guaranteed return with zero (very small) risk. If no such option exists, then
set risk_free_rate to zero, i.e. risk_free_rate=0. For the sample securian-401k.csv, the risk free
option is 'Minnesota Life General Account'.

Asset Classes

Standard asset classes and subclasses are specified within universe/asset-classes.csv.

    - US Stocks- aka equities are stocks and mutual funds, represent shares of ownership in publicly
      held companies
    - Global Stocks - focus on International Equities
    - Bonds - aka fixed income investments, generally pay a set rate of interest over a given period,
      then return the investor's principal.
    - Cash Equivalents - assets similar to cash in regards to risk and liquidity
    - Real Estate - home or investment property, plus shares of funds that invest in commercial
      real estate
    - Commodities - physical goods such as gold, copper, crude oil, natural gas, wheat, corn,
      and even electricity.
    - Currencies - foreign exchange market, e.g. EUR/USD currency pair.

Optimizer

Optimize sharpe ratio while allowing a min_annual_rate. Setting min_annual_rate to None optimizes
sharpe_ratio without regard to min_annual_rate. Calculation is done via a Monte Carlo Simulation
by trying random combinations of weights and checking which combination has the best sharpe_ratio.
As a result, results may vary slightly between optimizations even when using the same inputs.

References

https://www.blueskycapitalmanagement.com/portfolio-diversification-how-to-potentially-gain-better-returns-per-unit-of-risk/  
https://en.wikipedia.org/wiki/Modern_portfolio_theory  
https://link.springer.com/article/10.1057/jt.2009.5  
https://www.investopedia.com/terms/s/standarddeviation.asp  
https://en.wikipedia.org/wiki/Asset_classes  
https://en.wikipedia.org/wiki/Cash_and_cash_equivalents

