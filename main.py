import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def get_data():
    # data download
    ticker = 'AAPL'
    data = yf.download(ticker, start='2018-01-01', end='2022-01-01', interval='1d')

    # organizing the table
    data = data[[ 'Open', 'Close', 'High', 'Low', 'Volume']]
    data = data.dropna()
    return data


def normalize(data, scaler):
    # Data normalization
    data_scaled = scaler.fit_transform(data[['Open', 'Close', 'High', 'Low', 'Volume']])
    data_scaled = pd.DataFrame(data_scaled, columns=['Open', 'Close', 'High', 'Low', 'Volume'])
    return data_scaled


def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length, :])
        targets.append(data[i+sequence_length, :])
    
    return np.array(sequences), np.array(targets)


def train(X_train, X_test, y_train, y_test):
    model = Sequential()

    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))


    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))


    model.add(Dense(units=100))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(units=100, kernel_regularizer=l2(0.01)))

    model.add(Dense(5))  # 5 wyjścia: Open, Close, High, Low, Volume


    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=100, batch_size= 8, validation_data=(X_test, y_test), callbacks = [early_stopping])
    return model


def plot(y_test_rescaled,y_pred_rescaled):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_rescaled[:, 1], label='Real Close Prices', color='blue')  # y_test_rescaled for 'Close'
    plt.plot(y_pred_rescaled[:, 1], label='Predicted Close Prices', color='red')  # y_pred_rescaled for 'Close'
    plt.legend()
    plt.show()


data = get_data()
scaler = MinMaxScaler()
data_scaled = normalize(data, scaler)

# data has x sequences for 10 days
sequence_length = 10
X, y = create_sequences(data_scaled.values, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train(X_train, X_test, y_train, y_test)
predictions = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(predictions)  # Denormalize the predictions (all 5 columns)
y_test_rescaled = scaler.inverse_transform(y_test)  # Denormalize the actual values (all 5 columns)

plot(y_test_rescaled, y_pred_rescaled)