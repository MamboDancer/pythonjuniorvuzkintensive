import sys
import requests
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback

class Trainer:
    def __init__(self, symbol, epochs, history_limit, future_steps=10):
        self.symbol = symbol
        self.epochs = epochs
        self.history_limit = history_limit
        self.future_steps = future_steps

    def get_klines(self):
        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": self.symbol, "interval": "1h", "limit": self.history_limit}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [float(item[4]) for item in data]
        except Exception as e:
            print(f"Error getting data from Binance: {e}")
            return []

    def train(self):
        prices = self.get_klines()
        if not prices:
            return [], []

        data = np.array(prices).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(50, len(scaled_data) - self.future_steps):
            X.append(scaled_data[i - 50:i, 0])
            y.append(scaled_data[i:i + self.future_steps, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
        model.add(Dense(self.future_steps))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X, y, epochs=self.epochs, batch_size=32)

        last_sequence = scaled_data[-50:].reshape(1, 50, 1)
        prediction = model.predict(last_sequence, verbose=0)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        print(prediction.tolist())
        return prices, prediction.tolist()

class CryptoPredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto LSTM Predictor")
        self.setGeometry(100, 100, 900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.top_layout = QHBoxLayout()
        self.pair_label = QLabel("Trading pair:")
        self.pair_combo = QComboBox()
        self.pair_combo.addItems(["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"])
        self.pair_combo.currentTextChanged.connect(self.plot_current_price)

        self.epochs_label = QLabel("Epochs:")
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(10)

        self.history_label = QLabel("Klines qt. from Binance:")
        self.history_input = QSpinBox()
        self.history_input.setRange(50, 1000)
        self.history_input.setValue(200)

        self.train_button = QPushButton("Predict")
        self.train_button.clicked.connect(self.run_prediction)

        self.top_layout.addWidget(self.pair_label)
        self.top_layout.addWidget(self.pair_combo)
        self.top_layout.addWidget(self.epochs_label)
        self.top_layout.addWidget(self.epochs_input)
        self.top_layout.addWidget(self.history_label)
        self.top_layout.addWidget(self.history_input)
        self.top_layout.addWidget(self.train_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addLayout(self.top_layout)
        layout.addWidget(self.progress_bar)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.toggle_top_interface(True)
        self.plot_current_price(self.pair_combo.currentText())

    def toggle_top_interface(self, enabled):
        self.train_button.setEnabled(enabled)
        self.pair_combo.setEnabled(enabled)
        self.epochs_input.setEnabled(enabled)
        self.history_input.setEnabled(enabled)

    def run_prediction(self):
        symbol = self.pair_combo.currentText()
        epochs = self.epochs_input.value()
        history_limit = self.history_input.value()

        self.toggle_top_interface(False)
        self.progress_bar.setValue(0)

        trainer = Trainer(symbol, epochs, history_limit)
        real_prices, predicted = trainer.train()

        self.plot_prediction(real_prices, predicted)
        self.progress_bar.setValue(100)
        self.toggle_top_interface(True)

    def plot_prediction(self, real_prices, predicted):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(real_prices, label="Actual Prices")
        last_index = len(real_prices)
        future_index = list(range(last_index, last_index + len(predicted)))
        ax.plot(future_index, predicted, label="Prediction")
        ax.legend()
        ax.set_title("Price LSTM predictor")
        self.canvas.draw()

    def plot_current_price(self, symbol):
        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": self.history_input.value()}
        try:
            response = requests.get(url, params=params)
            data = response.json()
            prices = [float(item[4]) for item in data]
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(prices, label="Actual prices")
            ax.legend()
            ax.set_title(f"Actual prices for {symbol}")
            self.canvas.draw()
        except Exception as e:
            print("Error loading chart:", e)

Trainer("BTCUSDT", 50, 500).train()
"""
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CryptoPredictor()
    window.show()
    sys.exit(app.exec_())
"""