import sys
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QComboBox, QSpinBox, QPushButton, QProgressBar, QMessageBox, QFileDialog  # ➕ Нове: QMessageBox, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from TrainingThread import TrainingThread


class CryptoPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Price Predictor")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.top_layout = QHBoxLayout()
        self.pair_combo = QComboBox()
        self.pair_combo.addItems(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        self.pair_combo.setToolTip("Choose crypto pair")  # ➕ Нове
        self.pair_combo.currentIndexChanged.connect(self.plot_current_price)  # ➕ Нове

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 500)
        self.epochs_input.setValue(30)
        self.epochs_input.setToolTip("Epoch count to train")  # ➕ Нове

        self.history_input = QSpinBox()
        self.history_input.setRange(100, 1000)
        self.history_input.setValue(300)
        self.history_input.setToolTip("How many history points to train")  # ➕ Нове

        self.interval_label = QLabel("Interval:")
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["15m", "1h", "4h", "1d"])
        self.interval_combo.setToolTip("Candle interval")  # ➕ Нове
        self.interval_combo.currentIndexChanged.connect(self.plot_current_price)  # ➕ Нове

        self.model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "GRU"])
        self.model_combo.setToolTip("Model type to train")  # ➕ Нове

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

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        layout.addLayout(self.top_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.plot_current_price()  # ➕ Нове: показ графіка одразу

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

    def plot_prediction(self, prices, prediction, loss_history):  # ➕ Нове: приймаємо втрати
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)  # ➕ Нове: верхній графік — ціни
        ax1.plot(prices, label="Actual Price")
        ax1.plot(list(range(len(prices), len(prices) + len(prediction))), prediction, label="Prediction", linestyle='dashed')
        ax1.legend()
        ax1.set_title("Price Prediction")

        ax2 = self.figure.add_subplot(212)  # ➕ Нове: нижній графік — втрати
        ax2.plot(loss_history, color='red')
        ax2.set_title("Model Loss per Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")

        self.canvas.draw()
        QMessageBox.information(self, "Готово", "Прогноз завершено 🎉")  # ➕ Нове

    def plot_current_price(self):  # ➕ Нове
        symbol = self.pair_combo.currentText()
        interval = self.interval_combo.currentText()
        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": self.history_input.value()}
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
            QMessageBox.warning(self, "Помилка", f"Не вдалося завантажити графік: {e}")  # ➕ Нове

    def save_chart(self):  # ➕ Нове
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Chart", "", "PNG Files (*.png)", options=options)
        if file_name:
            self.figure.savefig(file_name)
            QMessageBox.information(self, "Saved!", f"Plot saved as:\n{file_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CryptoPredictor()
    window.show()
    sys.exit(app.exec_())
