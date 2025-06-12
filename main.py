import sys
import requests
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QComboBox, QSpinBox, QPushButton, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback


class TrainingThread(QThread):
    progress_changed = pyqtSignal(int)
    prediction_done = pyqtSignal(list, list)

    def __init__(self, symbol, interval, epochs, history_limit, model_type):
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.epochs = epochs
        self.history_limit = history_limit
        self.model_type = model_type
        self.future_steps = 10

    def get_klines(self, symbol, interval):
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": self.history_limit}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return [float(item[4]) for item in response.json()]
        except Exception as e:
            print(f"Error fetching data: {e}")
            self.prediction_done.emit([], [])
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
            def __init__(self, signal):
                self.signal = signal

            def on_epoch_end(self, epoch, logs=None):
                progress = int((epoch + 1) / self.params['epochs'] * 100)
                self.signal.emit(progress)

        model.fit(X, y, epochs=self.epochs, batch_size=32, callbacks=[ProgressCallback(self.progress_changed)])

        last_sequence = scaled_data[-50:].reshape(1, 50, 1)
        prediction = model.predict(last_sequence)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

        self.prediction_done.emit(prices, prediction.tolist())


class CryptoPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Price Predictor")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Верхній блок
        self.top_layout = QHBoxLayout()
        self.pair_combo = QComboBox()
        self.pair_combo.addItems(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 500)
        self.epochs_input.setValue(30)
        self.history_input = QSpinBox()
        self.history_input.setRange(100, 1000)
        self.history_input.setValue(300)

        self.interval_label = QLabel("Interval:")
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["15m", "1h", "4h", "1d"])

        self.model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "GRU"])

        self.train_button = QPushButton("Predict")
        self.train_button.clicked.connect(self.run_prediction)

        self.save_button = QPushButton("Save Chart")
        self.save_button.clicked.connect(self.save_chart)

        self.top_layout.addWidget(QLabel("Pair:"))
        self.top_layout.addWidget(self.pair_combo)
        self.top_layout.addWidget(QLabel("Epochs:"))
        self.top_layout.addWidget(self.epochs_input)
        self.top_layout.addWidget(QLabel("History:"))
        self.top_layout.addWidget(self.history_input)
        self.top_layout.addWidget(self.interval_label)
        self.top_layout.addWidget(self.interval_combo)
        self.top_layout.addWidget(self.model_label)
        self.top_layout.addWidget(self.model_combo)
        self.top_layout.addWidget(self.train_button)
        self.top_layout.addWidget(self.save_button)

        # Графік
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Прогрес
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        layout.addLayout(self.top_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def toggle_top_interface(self, enabled):
        for widget in [
            self.pair_combo, self.epochs_input, self.history_input,
            self.interval_combo, self.model_combo, self.train_button
        ]:
            widget.setEnabled(enabled)

    def run_prediction(self):
        symbol = self.pair_combo.currentText()
        interval = self.interval_combo.currentText()
        epochs = self.epochs_input.value()
        history_limit = self.history_input.value()
        model_type = self.model_combo.currentText()

        self.toggle_top_interface(False)
        self.progress_bar.setValue(0)

        self.thread = TrainingThread(symbol, interval, epochs, history_limit, model_type)
        self.thread.progress_changed.connect(self.progress_bar.setValue)
        self.thread.prediction_done.connect(self.plot_prediction)
        self.thread.finished.connect(lambda: self.toggle_top_interface(True))
        self.thread.start()

    def plot_prediction(self, prices, prediction):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(prices, label="Actual Price")
        ax.plot(list(range(len(prices), len(prices) + len(prediction))), prediction, label="Prediction", linestyle='dashed')
        ax.legend()
        ax.set_title("Price Prediction")
        self.canvas.draw()

    def save_chart(self):
        self.figure.savefig("prediction_chart.png")
        print("Chart saved as prediction_chart.png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CryptoPredictor()
    window.show()
    sys.exit(app.exec_())
