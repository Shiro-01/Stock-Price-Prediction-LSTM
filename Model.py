import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load and preprocess data
def load_data(ticker, start_date='2010-01-01', end_date='2020-12-31', split_fraction=0.8):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']].copy()

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    # Split data into training and testing sets
    train_size = int(len(scaled_data) * split_fraction)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    return train_data, test_data, scaler

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length), 0]
        y = data[i+seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Build LSTM model
def build_model(seq_length):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and evaluate the model
def train_and_evaluate(ticker, start_date='2010-01-01', end_date='2020-12-31', seq_length=60, epochs=10):
    train_data, test_data, scaler = load_data(ticker, start_date, end_date)
    train_x, train_y = create_sequences(train_data, seq_length)
    test_x, test_y = create_sequences(test_data, seq_length)
    
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    
    model = build_model(seq_length)
    model.fit(train_x, train_y, epochs=epochs, batch_size=32)
    
    # Evaluate the model
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    
    # Inverse transform the predictions
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    
    # Calculate metrics
    train_score = np.sqrt(np.mean(np.square(train_predict - train_y)))
    test_score = np.sqrt(np.mean(np.square(test_predict - test_y)))
    
    print(f'Train RMSE: {train_score}')
    print(f'Test RMSE: {test_score}')
    
    # Plot the predictions
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(train_data)), scaler.inverse_transform(train_data), label='Original Data')
    plt.plot(np.arange(seq_length, len(train_predict) + seq_length), train_predict, label='Training Predictions')
    plt.plot(np.arange(len(train_data) + seq_length, len(train_data) + seq_length + len(test_predict)),
             test_predict, label='Test Predictions')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main function to run the project
if __name__ == '__main__':
    ticker = 'AAPL'  # Apple stock ticker
    train_and_evaluate(ticker)
