#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:04:43 2024

@author: alexawhitesell
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input
import matplotlib.pyplot as plt

# Download data for selected assets
def download_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data.index = data.index.tz_localize(None) # Accounts for different timezones in yf data and ff data
        return data.dropna(axis=0)  # Drop rows with missing data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

def load_risk_free_rate(file_path):
    try:
        ff_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        ff_data.index = ff_data.index.tz_localize(None)
        risk_free_df = ff_data['RF'] / 100
        return risk_free_df
    except Exception as e:
        print(f"Error loading risk-free rate data: {e}")
        return pd.Series()

def merge_data_with_rf(asset_data, risk_free_data):
    try:
        merged_data = asset_data.merge(risk_free_data, left_index=True, right_index=True, how='inner')
        return merged_data
    except Exception as e:
        print(f"Error merging data: {e}")
        return pd.DataFrame()

def calculate_excess_returns(merged_data):
    returns = merged_data[merged_data.columns[:-1]].pct_change().dropna()
    excess_returns = returns.sub(merged_data['RF'], axis=0)  # Subtract risk-free rate row-wise
    return excess_returns

# Monte Carlo Simulation for Portfolio Optimization
def simulate_portfolios(data, num_portfolios=100000, risk_free_rate=0.04, allow_short=False):
    returns = data.pct_change().mean() * 252  # Annualized returns
    cov_matrix = data.pct_change().cov() * 252  # Annualized covariance
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.uniform(-1, 1, len(data.columns)) if allow_short else np.random.random(len(data.columns))
        weights /= np.sum(np.abs(weights))  # Normalize weights (long-short normalization)

        # Calculate portfolio return and risk
        portfolio_return = np.dot(weights, returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std

        # Skip invalid results
        if np.isnan(portfolio_return) or np.isnan(portfolio_std):
            continue

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio
        weights_record.append(weights)

    # Ensure at least one valid portfolio exists
    if len(weights_record) == 0:
        raise ValueError("No valid portfolios generated. Check your input data.")

    # Identify the optimal portfolio (highest Sharpe ratio) and lowest-risk portfolio
    max_sharpe_idx = np.argmax(results[2, :])
    min_risk_idx = np.argmin(results[1, :])
    return results, weights_record[max_sharpe_idx], weights_record[min_risk_idx], max_sharpe_idx, min_risk_idx

# Plot Efficient Frontier
def plot_efficient_frontier(results, max_sharpe_idx, min_risk_idx):
    plt.figure(figsize=(10, 7))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.6)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], color='red', marker='*', s=200, label='Max Sharpe Ratio')
    plt.scatter(results[1, min_risk_idx], results[0, min_risk_idx], color='blue', marker='^', s=200, label='Min Risk Portfolio')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Return')
    plt.legend()
    plt.grid()
    plt.show()

# Prepare data for LSTM/GRU
def prepare_data(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Build and train LSTM/GRU model
def build_and_train_model(X_train, y_train, model_type='LSTM'):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(units=50, return_sequences=True) if model_type == 'LSTM' else GRU(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50) if model_type == 'LSTM' else GRU(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=21, verbose=0)
    return model

# Calculate performance metrics (Sharpe Ratio, Max Drawdown, MSE)
def calculate_metrics(weights, returns, cov_matrix, risk_free_rate, actual_prices):
    portfolio_return = np.dot(weights, returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    drawdown = np.min(actual_prices / np.maximum.accumulate(actual_prices) - 1)  # Max drawdown
    mse = np.mean((actual_prices - np.dot(actual_prices, weights)) ** 2)  # MSE of weights vs actual
    return sharpe_ratio, drawdown, mse

# Calculate CAGR
def calculate_cagr(initial_value, final_value, years):
    return (final_value / initial_value) ** (1 / years) - 1

def calculate_daily_portfolio_returns(test_data, weights):
    daily_returns = test_data.pct_change().dropna()
    portfolio_returns = daily_returns @ weights
    return portfolio_returns

def calculate_cumulative_returns(daily_returns):
    return (1 + daily_returns).cumprod()

def get_sp500_data(start_date, end_date):
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    sp500_returns = sp500.pct_change().dropna()
    sp500_cumulative_returns = (1 + sp500_returns).cumprod()
    return sp500_cumulative_returns

# Main script
def main():
    # Step 1: Define tickers and download data
    tickers = ['GLD', 'TLT', 'EFA', 'QQQ', 'VNQ']
    print("Downloading data for selected assets...")
    data = download_data(tickers, start_date='2000-01-01', end_date='2023-01-01')

    if data.empty:
        print("No valid data downloaded. Exiting.")
        return

    # Download risk free rate from Fama-French data
    risk_free_data = load_risk_free_rate('./data/F-F_Research_Data_Factors_daily.csv')
    merged_data = merge_data_with_rf(data, risk_free_data)
    excess_returns = calculate_excess_returns(merged_data).dropna()

    # Step 2: Split into training (2000-2020) and testing (2021-2023)
    train_data = data.loc[:'2020-12-31']
    test_data = data.loc['2021-01-01':]

    # Step 3: Portfolio Simulation using MPT
    print("Simulating portfolios...")
    try:
        results, optimal_weights_mpt, min_risk_weights, max_idx, min_risk_idx = simulate_portfolios(train_data, allow_short=True)
        plot_efficient_frontier(results, max_idx, min_risk_idx)  # Plot the efficient frontier
    except ValueError as e:
        print(f"Error during portfolio simulation: {e}")
        return

    # Print optimal weights for MPT
    print("\nOptimal Weights:")
    print(f"Max Sharpe Ratio Portfolio (MPT): {optimal_weights_mpt}")
    print(f"Min Risk Portfolio: {min_risk_weights}")

    # Step 4: LSTM/GRU Training and Performance Evaluation
    scaler = MinMaxScaler()
    rebalanced_weights = {'LSTM': [], 'GRU': []}
    for stock in train_data.columns:
        stock_data = train_data[stock].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(stock_data)

        X, y = prepare_data(scaled_data)
        if len(X) == 0:  # Skip if no valid data for training
            print(f"Skipping stock {stock} due to insufficient training samples.")
            continue

        X_train, y_train = X[:-21], y[:-21]  # Using training set

        # LSTM
        lstm_model = build_and_train_model(X_train, y_train, 'LSTM')
        lstm_weights = lstm_model.predict(X[-21:]).flatten()
        rebalanced_weights['LSTM'].append(np.mean(lstm_weights))

        # GRU
        gru_model = build_and_train_model(X_train, y_train, 'GRU')
        gru_weights = gru_model.predict(X[-21:]).flatten()
        rebalanced_weights['GRU'].append(np.mean(gru_weights))

    # Normalize weights
    rebalanced_weights['LSTM'] /= np.sum((rebalanced_weights['LSTM']))
    rebalanced_weights['GRU'] /= np.sum((rebalanced_weights['GRU']))

    print("\nNormalized LSTM Weights:", rebalanced_weights['LSTM'])
    print("Normalized GRU Weights:", rebalanced_weights['GRU'])
    # Adjust predictions to allow for short positions

    # Step 5: Evaluate performance on test data
    test_returns = test_data.pct_change().mean() * 252
    test_cov_matrix = test_data.pct_change().cov() * 252
    actual_prices = test_data.iloc[-21:].mean(axis=0)

    print("\nPerformance Metrics on Test Data:")
    for method, weights in {'MPT': optimal_weights_mpt, **rebalanced_weights}.items():
        sharpe_ratio, drawdown, mse = calculate_metrics(weights, test_returns, test_cov_matrix, 0.02, actual_prices)
        # Calculate CAGR
        initial_value = 1000  # Starting value of $1000
        final_value = initial_value * (1 + sharpe_ratio)  # This is a simplification for CAGR calculation
        years = 3  # Period from 2021 to 2023
        cagr = calculate_cagr(initial_value, final_value, years)

        print(f"{method}: Sharpe Ratio = {sharpe_ratio:.2f}, Max Drawdown = {drawdown:.2f}, MSE = {mse:.2f}, CAGR = {cagr:.2%}")

    # Step 6: Backtest Portfolio Performance
    print("\nBacktesting Portfolio Performance...")

    # Calculate portfolio returns for each strategy
    backtest_results = {}
    for method, weights in {'MPT': optimal_weights_mpt, **rebalanced_weights}.items():
        portfolio_returns = calculate_daily_portfolio_returns(test_data, weights)
        backtest_results[method] = calculate_cumulative_returns(portfolio_returns)

    # Get S&P 500 cumulative returns for comparison
    sp500_cumulative_returns = get_sp500_data(start_date='2021-01-01', end_date='2023-01-01')

    # Plot cumulative returns for portfolio strategies and benchmark
    plt.figure(figsize=(12, 6))
    for method, cumulative_returns in backtest_results.items():
        plt.plot(cumulative_returns, label=f'{method} Portfolio')
    plt.plot(sp500_cumulative_returns, label='S&P 500', linestyle='--', color='orange')
    plt.title('Backtest: Growth of $1')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
