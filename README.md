# Stock Market Prediction using Neural Network

This repository contains scripts for training a neural network to predict stock prices and using the trained model to make future forecasts.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example Results](#example-results)
  
---

## Features  

- Downloads stock market data (`Open`, `Close`, `High`, `Low`, `Volume`)  
- Computes technical indicators:  
  - Moving Average  
  - Relative Strength Index (RSI)  
  - Moving Average Convergence Divergence (MACD)  
- Scales and prepares data for training  
- Trains a neural network and saves the model  
- Forecasts future stock prices using the trained model 
  
---

## Requirements

The project requires the following:
- Python 3.10
- Libraries: yfinance, pandas, numpy, matplotlib.pyplot, tensorflow, sklearn


---

## Installation 
git clone https://github.com/pzimnota/finance_NN.git

---

## Usage
 1. Training the Model
- Run `python train_model.py` script to train a neural network on stock market data:

    -  Downloads historical stock data
    -  Computes technical indicators
    -  Scales the data and creates sequences
    -  Trains the neural network
    -  Saves the trained model
    -  Generates a plot comparing actual vs. predicted prices
     
2. Forecasting Future Prices
- Run `python forecast.py` script to predict the next day's closing price:

    -  Downloads the latest stock data
    -  Scales the data and creates sequences
    -  Loads the saved model
    -  Predicts the next day's closing price
    -  Generates a plot comparing actual vs. predicted prices
---

## Example Results
IN PROGRESS
</p>

---
