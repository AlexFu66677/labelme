from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtCore import Qt


class TrackingPredictionWindow(QtWidgets.QDialog):
    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super(TrackingPredictionWindow, self).__init__(parent)
        self.setWindowTitle(self.tr("跟踪预测"))
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.Tool, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(300, 180)

    def closeEvent(self, event):
        self.closed.emit()
        super(TrackingPredictionWindow, self).closeEvent(event)
