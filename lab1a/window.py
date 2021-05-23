from PyQt5 import QtWidgets
import lab1a, interface.mainw
from PyQt5.QtWidgets import *


def showError(text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Йой!")
    msg.setInformativeText(text)
    msg.setWindowTitle("Помилка")
    msg.exec_()


class MyApp(QtWidgets.QMainWindow, interface.mainw.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.solve_btn)

    def solve_btn(self):
        try:
            n = int(self.lineEdit.text())
            if n % 2 == 0:
                showError("Введене число має бути непарним")
            else:
                result = lab1a.factorization(n)
                if len(result) == 1:
                    self.textBrowser.clear()
                    self.textBrowser.append("Задане число вже є простим")
                else:
                    text = "n = "
                    for i in result:
                        text += "{} \u00b7 ".format(i)
                    text = text[:-2]
                    self.textBrowser.clear()
                    self.textBrowser.append(text)
        except():
            showError("Перевірте правильність введених даних")
