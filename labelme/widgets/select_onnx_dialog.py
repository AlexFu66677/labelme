
from qtpy import QtWidgets
from .. import dataset


class Selectonnx(QtWidgets.QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Onnx")
        layout =QtWidgets.QVBoxLayout()
        self.config_label = QtWidgets.QLabel('Configuration File:')
        self.config_input = QtWidgets.QLineEdit(self)
        self.config_button = QtWidgets.QPushButton('Select File', self)
        self.config_button.clicked.connect(self.select_onnx_file)
        layout.addWidget(self.config_label)
        layout.addWidget(self.config_input)
        layout.addWidget(self.config_button)
        self.setLayout(layout)
        self.file_path=''

    def select_onnx_file(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file_dialog.setNameFilter('Text files (*.onnx);;All files (*.*)')
        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            self.config_input.setText(self.file_path)

