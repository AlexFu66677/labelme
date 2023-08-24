from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class QCWidget(QtWidgets.QLabel):
    def __init__(self):
        super(QCWidget, self).__init__()
        self.result_label = QtWidgets.QLabel("QC Result:")
        self.result_text_edit = QtWidgets.QTextEdit()

