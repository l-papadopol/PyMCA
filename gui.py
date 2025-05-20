# gui.py  (quasi invariato)
from PyQt5 import QtWidgets, QtCore
import sys, pyqtgraph as pg

from models.scope import SimplePicoScope2000
from views.osc_tab import OscTab
from views.mca_tab import McaTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.scope = SimplePicoScope2000()
        self.setWindowTitle("PicoScope 2204A – Oscilloscope + MCA")
        self.resize(1100, 600)

        tabs = QtWidgets.QTabWidget()
        self.osc_tab = OscTab(self.scope, self)
        self.mca_tab = McaTab(self.scope, self)
        tabs.addTab(self.osc_tab, "Oscilloscope")
        tabs.addTab(self.mca_tab, "MCA")
        self.setCentralWidget(tabs)

    def closeEvent(self, e):
        self.osc_tab.clean_up()
        self.mca_tab.clean_up()
        self.scope.close()
        super().closeEvent(e)


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

