from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtCore import Qt


class SequenceWarpingWindow(QtWidgets.QDialog):
    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super(SequenceWarpingWindow, self).__init__(parent)
        self.setWindowTitle(self.tr("Sequence Warping"))
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.Tool, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(280, 240)

    def closeEvent(self, event):
        self.closed.emit()
        super(SequenceWarpingWindow, self).closeEvent(event)
