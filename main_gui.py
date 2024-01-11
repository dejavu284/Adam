import numpy as np
import sys
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from sympy import sympify, symbols
import tensorflow as tf
from tensorflow.keras import layers

class RGR3(QtWidgets.QWidget):
    def __init__(self):
        super(RGR3, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Adam')
        self.setGeometry(100, 100, 800, 600)

        self.label_function = QtWidgets.QLabel('Введите функцию от x:', self)
        self.label_function.move(10, 10)

        self.lineEdit_function = QtWidgets.QLineEdit(self)
        self.lineEdit_function.setGeometry(150, 10, 200, 25)

        self.label_learning_rate = QtWidgets.QLabel('Выберите learning rate:', self)
        self.label_learning_rate.move(10, 40)

        self.comboBox_learning_rate = QtWidgets.QComboBox(self)
        self.comboBox_learning_rate.addItems(["0.1", "0.01", "0.001"])
        self.comboBox_learning_rate.setGeometry(150, 40, 80, 25)

        self.label_epochs = QtWidgets.QLabel('Выберите количество эпох:', self)
        self.label_epochs.move(10, 70)

        self.comboBox_epochs = QtWidgets.QComboBox(self)
        self.comboBox_epochs.addItems(["100", "300", "500"])
        self.comboBox_epochs.setGeometry(150, 70, 80, 25)

        self.btn_start = QtWidgets.QPushButton('Start Training', self)
        self.btn_start.setGeometry(10, 100, 120, 25)
        self.btn_start.clicked.connect(self.start_training)

        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setGeometry(10, 140, 780, 450)

    def get_user_function(self):
        expression_str = self.lineEdit_function.text()
        x = symbols('x')
        expression = sympify(expression_str)
        return lambda x_val: expression.subs(x, x_val)

    def generate_data(self, user_function):
        np.random.seed(42)
        x = np.sort(10 * np.random.rand(100))
        y = np.array([user_function(x_val) + 0 * np.random.randn(1)[0] for x_val in x])
        return x, y

    def build_model(self, learning_rate):
        model = tf.keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(1,)),
            layers.Dense(20, activation='relu'),
            layers.Dense(20, activation='relu'),
            layers.Dense(20, activation='relu'),
            layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def train_and_animate(self, x, y, learning_rate, num_epochs, loss_threshold=0.01):
        model = self.build_model(learning_rate)

        real_data = self.plot_widget.plot(x, y, pen=None, symbol='o', symbolPen='r', symbolSize=5, symbolBrush='r')
        prediction_data = self.plot_widget.plot(x, np.zeros_like(x), pen='g')  # Инициализация графика предсказаний

        losses = []

        for epoch in range(num_epochs):
            history = model.fit(x, y, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            losses.append(loss)

            prediction = model.predict(x)
            real_data.setData(x, y)
            prediction_data.setData(x, prediction.flatten())

            epoch_text = pg.TextItem(f'Epoch: {epoch + 1}', color=(0, 0, 255), anchor=(1, 1))
            loss_text = pg.TextItem(f'Loss: {loss:.4f}', color=(255, 0, 0), anchor=(1, 0))
            self.plot_widget.addItem(epoch_text)
            self.plot_widget.addItem(loss_text)

            QtCore.QCoreApplication.processEvents()  # Позволяет интерфейсу обновляться

        # Отрисовываем график ошибок
        self.plot_widget.clear()
        self.plot_widget.plot(losses, pen='b')
        self.plot_widget.setLabel('left', 'Loss', units='a.u.')
        self.plot_widget.setLabel('bottom', 'Epoch', units='a.u.')

    def start_training(self):
        user_function = self.get_user_function()
        x, y = self.generate_data(user_function)
        learning_rate = float(self.comboBox_learning_rate.currentText())
        num_epochs = int(self.comboBox_epochs.currentText())
        self.train_and_animate(x, y.astype('float32'), learning_rate, num_epochs)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    application = RGR3()
    application.show()
    sys.exit(app.exec_())
