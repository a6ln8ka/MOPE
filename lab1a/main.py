import sys
from PyQt5 import QtWidgets
import window


def main():
    app = QtWidgets.QApplication(sys.argv)
    mwindow = window.MyApp()
    mwindow.show()
    app.exec_()

main()