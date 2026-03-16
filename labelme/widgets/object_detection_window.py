from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtCore import Qt


class ObjectDetectionWindow(QtWidgets.QDialog):
    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super(ObjectDetectionWindow, self).__init__(parent)
        self.setWindowTitle(self.tr("Object Detection"))
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.Tool, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(320, 430)

    def closeEvent(self, event):
        self.closed.emit()
        super(ObjectDetectionWindow, self).closeEvent(event)
