import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def get_data():
    # data download
    ticker = 'AAPL'
    data = yf.download(ticker, interval='1d')

    # organizing the table
    data = data[[ 'Open', 'Close', 'High', 'Low', 'Volume']]
    data = data.dropna()
    return data


def calculate_moving_average(data, window):
    data['SMA'] = data['Close'].rolling(window=window).mean()
    data = data.dropna() 
    return data


def calculate_rsi(data, window):
    delta = data['Close'].diff()    #diference between close[i] and close[i-1]
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Average profit during window
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Average loss during window

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
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

    model.add(LSTM(units=200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(BatchNormalization())

    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.2))



    model.add(LSTM(units=10, return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))


    model.add(Dense(5))  # 5 Outputs: Open, Close, High, Low, Volume


    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=_EPOCHS, batch_size= _BATCH_SIZE, validation_data=(X_test, y_test), callbacks = [early_stopping])
    return model


def plot(y_test_rescaled,y_pred_rescaled):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_rescaled[:, 1], label='Real Close Prices', color='blue')  # y_test_rescaled for 'Close'
    plt.plot(y_pred_rescaled[:, 1], label='Predicted Close Prices', color='red')  # y_pred_rescaled for 'Close'
    plt.legend()
    plt.show()


# model const
_BATCH_SIZE = 4
_EPOCHS = 100

data = get_data()
scaler = MinMaxScaler()
data_scaled = normalize(data, scaler)

# data has x sequences for 5 days
sequence_length = 5
X, y = create_sequences(data_scaled.values, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train(X_train, X_test, y_train, y_test)

predictions = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(predictions)  # Denormalize the predictions (all 5 columns)
y_test_rescaled = scaler.inverse_transform(y_test)  # Denormalize the actual values (all 5 columns)
model.save("Apple_model.keras")
plot(y_test_rescaled, y_pred_rescaled)
