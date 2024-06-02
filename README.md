# Stock Price Prediction using LSTM

This repository contains a project aimed at predicting stock prices of Apple Inc. (AAPL) using Long Short-Term Memory (LSTM) neural networks. The project leverages historical stock price data obtained from Yahoo Finance and includes additional technical indicators to enhance the predictive power of the model.

## Project Overview

Stock prices are influenced by numerous factors and exhibit high volatility and randomness. Despite these challenges, time series forecasting models like LSTM can capture underlying patterns and trends in historical data. This project demonstrates a basic yet effective approach to stock price prediction using machine learning.

## Features

- **Data Collection**: Historical stock price data for Apple Inc. (AAPL) is downloaded using the Yahoo Finance API.
- **Data Preprocessing**: The data is normalized, and additional technical indicators (SMA_20, SMA_50, RSI) are added.
- **LSTM Model**: A Long Short-Term Memory (LSTM) neural network is built using TensorFlow and Keras.
- **Training and Evaluation**: The model is trained and evaluated using Root Mean Squared Error (RMSE) as the performance metric.
- **Visualization**: Predicted stock prices are plotted against actual prices to visualize the model's performance.

## Repository Structure

- `stock_price_prediction_lstm.py`: The main script that includes data collection, preprocessing, model building, training, and evaluation.
- `README.md`: Project description and instructions.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `scikit-learn`
- `tensorflow`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
