# Portfolio Optimization Project

## Overview
This project is a simple implementation of a portfolio optimization model using historical financial data, for a competition I took part in.
The optimization framework is built on convex programming techniques and incorporates the Black-Litterman model for managing views on asset returns. 
The system retrieves data from multiple sources, preprocesses it, and constructs an optimized portfolio based on risk constraints and return expectations.

## Features
- Fetches historical market data using `yfinance` and predefined CSV datasets.
- Processes fixed income and futures data using custom classes.
- Implements mean-variance optimization with cardinality constraints using `cvxpy`.
- Supports Black-Litterman adjustments for incorporating subjective market views.
- Provides visualization tools for portfolio allocation and performance backtesting.

## Installation
Ensure you have Python installed along with the following dependencies:

```bash
pip install numpy pandas cvxpy dxlib matplotlib locale
```

## How it works
1. We load historical market data with a modularized function, as follows:
   ```python
   histories = load_history_groups()
   ```
2. Manually define portfolio constraints and bounds based on the macroeconomic view of the portfolio manager.
3. Optimize the portfolio:
   ```python
   weights, df = build_portfolio(histories, bounds)
   ```
4. Visualize the results using `matplotlib`.

## Data Sources
- `yfinance` API for external market data.
- CSV files (`ibrx100.csv`, `bdrs.csv`) for local stock market tickers.
- Futures and fixed income processed through `Futures` and `IPCA` classes.

## Optimization Approach
The portfolio optimization is done using:
- Mean-variance optimization with variance constraints.
- Black-Litterman model for incorporating investor views.
- Convex optimization with constraints on sector allocation, position limits, and asset selection.


## Running with our Views
To run the same model and views we used for the competition, simply run the `main.py` file.


## License
This project is released under the MIT License.

