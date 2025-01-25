import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# from train_model import get_data, normalize, create_sequences


DATA_PERIOD = 60
SEQUENCE_LENGTH = 5

def get_data():
    # data download
    ticker = 'AAPL'
    data = yf.download(ticker, interval='1d')

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



def load_trained_model():
    return load_model("Apple_model.keras")


def predict_next_day(model, X_last):
    prediction = model.predict(X_last)
    return prediction

def main():
    data = get_data()
    scaler = MinMaxScaler()
    data_scaled = normalize(data, scaler)

    
 
    X, _ = create_sequences(data_scaled.values, SEQUENCE_LENGTH)
    X_last = X[-1:]  # Last sequence for forecast



    # Load the saved model
    model = load_trained_model()

    # Next day forecast
    prediction = predict_next_day(model, X_last)
    y_pred_rescaled = scaler.inverse_transform(prediction)  # Denormalize

    # Prepating data for plot
    y_test_rescaled = scaler.inverse_transform(data_scaled.values[-DATA_PERIOD:])  # Last 60 days

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(range(DATA_PERIOD), y_test_rescaled[:, 1], label='Real Close Prices', color='blue')  # Last 60 days
    plt.plot(DATA_PERIOD, y_pred_rescaled[0, 1], marker='o', color='red', label='Predicted Next Day')  # 60th day prediction
    plt.plot([DATA_PERIOD - 1, DATA_PERIOD], [y_test_rescaled[-1, 1], y_pred_rescaled[0, 1]], linestyle='--', color='red')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()