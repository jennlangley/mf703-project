
# Credit-Driven Portfolio Optimization with MPT, LSTM, and GRU

This project implements a hybrid approach to portfolio optimization using both Modern Portfolio Theory (MPT) and deep learning techniques (LSTM and GRU). It combines traditional Monte Carlo simulation for risk-return tradeoff with neural networks to adaptively rebalance portfolio weights based on time-series forecasting.

## Overview

This strategy uses the following steps:
1. Download historical price data for a basket of ETFs.
2. Simulate portfolios using MPT to identify maximum Sharpe and minimum volatility allocations.
3. Train LSTM and GRU models on historical price patterns to generate predictive-based portfolio weights.
4. Evaluate all portfolios on out-of-sample data (2021–2023).
5. Compare results to a benchmark (S&P 500) in terms of Sharpe ratio, drawdown, MSE, and CAGR.

## Key Features

- Uses daily adjusted close prices via `yfinance`.
- Incorporates Fama-French risk-free rate for excess return calculations.
- Simulates 100,000 portfolios with and without short selling to trace the efficient frontier.
- Implements LSTM and GRU neural networks with rolling look-back windows to estimate forward-looking weights.
- Calculates key performance metrics: Sharpe Ratio, Max Drawdown, Mean Squared Error, and CAGR.
- Visualizes portfolio growth compared to the S&P 500.

## Methodology

- **MPT:** Generates optimal risk-adjusted weights through Monte Carlo sampling of random portfolios.
- **LSTM/GRU:** Time-series models predict future price paths and generate data-driven allocations.
- **Risk Metrics:** Portfolios are evaluated on out-of-sample returns with risk-adjusted metrics.

## Performance Metrics

- Calculated using test data from 2021–2023.
- Metrics include:
  - Sharpe Ratio
  - Maximum Drawdown
  - Mean Squared Error
  - Compound Annual Growth Rate (CAGR)

## Requirements

- Python 3.8+
- Libraries: `numpy`, `pandas`, `yfinance`, `matplotlib`, `tensorflow`, `scikit-learn`

## Running the Project

To execute the full pipeline:

```bash
python main.py
```

Make sure `F-F_Research_Data_Factors_daily.csv` is present in the project directory for risk-free rate data.

## Acknowledgments

- Robert C. Merton (1974), for foundational work in portfolio theory.
- Fama-French for risk-free rate data.
- TensorFlow and Keras for neural network implementations.
