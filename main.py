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
    #df = df.filter('Adj Close')

    # using backwards fill to fill in missing data to prevent any bias from using prediction
    df.bfill(inplace=True)
    return df

def create_model(df):
    # Add lag
    df['Lag1'] = df['Adj Close'].shift(1) # previous day's price
    df['Lag5'] = df['Adj Close'].shift(5) # 5 days ago price
    df['Lag20'] = df['Adj Close'].shift(20) # previous month's price
    # Simple Moving Averge
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean() # 20 day window for monthly stocks, 100 could be window for ETFs
    # Add target variable
    df['Next Close'] = df['Adj Close'].shift(-1) # creates next close price
    df.dropna(inplace=True)
    features = ['Adj Close', 'Lag1', 'Lag5', 'Lag20', 'SMA_20']
    X = df[features]
    Y = df['Next Close']

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, shuffle=False)

    # Define and train the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, verbose=0) # preserves 10% of data for validation

    # Predict on the full dataset
    df['Predicted Close'] = model.predict(X_scaled)

    # Forecast the next day's closing price
    latest_data = df[features].iloc[-1].values.reshape(1, -1)  # Use the last row of data as input
    latest_data_scaled = scaler.transform(latest_data)  # Scale the data
    next_day_prediction = model.predict(latest_data_scaled)

    # Print the last known close price, predicted close price, and next day forecast
    print(f"Last Known Closing Price: ${df['Adj Close'].iloc[-1]:.2f}")
    print(f"Last Predicted Closing Price: ${df['Predicted Close'].iloc[-1]:.2f}")
    print(f"Next Day Forecasted Closing Price: ${next_day_prediction[0][0]:.2f}")
    return df

def plot_actual_vs_predicted(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Adj Close'], label='Actual Close', color='blue')
    plt.plot(df.index, df['Predicted Close'], label='Predicted Close', color='red')
    plt.title('Actual vs Predicted Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    start_date = "2000-01-01"
    end_date = "2024-01-01"
    ticker = 'RUT'
    data = create_df(ticker, start_date, end_date)
    model = create_model(data)
    plot_actual_vs_predicted(model)


    data = create_df('SPY', start_date, end_date)
    model = create_model(data)
    plot_actual_vs_predicted(model)
