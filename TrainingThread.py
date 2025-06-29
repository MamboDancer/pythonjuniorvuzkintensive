import requests
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback
import numpy as np

class TrainingThread(QThread):
    progress_changed = pyqtSignal(int)
    prediction_done = pyqtSignal(list, list, list)

    def __init__(self, symbol, interval, epochs, history_limit, model_type):
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.epochs = epochs
        self.history_limit = history_limit
        self.model_type = model_type
        self.future_steps = 10
        self.loss_history = []

    def get_klines(self, symbol, interval):
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": self.history_limit}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return [float(item[4]) for item in response.json()]
        except Exception as e:
            print(f"Error fetching data: {e}")
            self.prediction_done.emit([], [], [])
            return []

    def run(self):
        prices = self.get_klines(self.symbol, self.interval)
        if not prices:
            return

        data = np.array(prices).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(50, len(scaled_data) - self.future_steps):
            X.append(scaled_data[i - 50:i, 0])
            y.append(scaled_data[i:i + self.future_steps, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        if self.model_type == "LSTM":
            from tensorflow.keras.layers import LSTM
            model.add(LSTM(50, input_shape=(X.shape[1], 1)))
        if self.model_type == "GRU":
            from tensorflow.keras.layers import GRU
            model.add(GRU(50, input_shape=(X.shape[1], 1)))

        model.add(Dense(self.future_steps))
        model.compile(optimizer='adam', loss='mean_squared_error')

        class ProgressCallback(Callback):
            def __init__(self, signal, loss_storage):
                self.signal = signal
                self.loss_storage = loss_storage

            def on_epoch_end(self, epoch, logs=None):
                self.loss_storage.append(logs['loss'])
                progress = int((epoch + 1) / self.params['epochs'] * 100)
                self.signal.emit(progress)

        model.fit(X, y, epochs=self.epochs, batch_size=32,
                  callbacks=[ProgressCallback(self.progress_changed, self.loss_history)])

        last_sequence = scaled_data[-50:].reshape(1, 50, 1)
        prediction = model.predict(last_sequence)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

        self.prediction_done.emit(prices, prediction.tolist(), self.loss_history)