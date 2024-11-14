# Imports
import os
import yfinance as yf

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def create_df(ticker, start_date, end_date):
    '''
        Function to download the data from yfinance and create a dataframe
        ticker = string name of ticker
        start and end date are string formatted dates
    '''
    file_path = './' + ticker + '.csv'

    # downloads the data to local csv if the filename does not exist
    if not os.path.exists(file_path):
        data = yf.download([ticker], start=start_date, end=end_date)
        data.to_csv(file_path)

    df = pd.read_csv(file_path, parse_dates=True, index_col="Date")
    # using backwards fill to fill in missing data to prevent any bias from using prediction
    df.bfill(inplace=True)

    # Add lag
    df['Lag1'] = df['Adj Close'].shift(1)
    df['Lag2'] = df['Adj Close'].shift(2)
    df['Lag3'] = df['Adj Close'].shift(3)
    # Simple Moving Averge
    df['SMA_1'] = df['Adj Close'].rolling(window=1).mean()
    # Add target variable
    df['Next Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    features = ['Adj Close', 'Lag1', 'Lag2', 'Lag3', 'SMA_1']
    X = data[features]
    Y = data['Next Close']

    # Scale the features
    scaler = sklearn.StandardScaler()



if __name__ == "__main__":
    start_date = "2000-01-01"
    end_date = "2024-01-01"
    ticker = 'RUT'
    data = create_df(ticker, start_date, end_date)
    print(data.head())
