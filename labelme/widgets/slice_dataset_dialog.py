
from qtpy import QtWidgets
from .. import dataset
from sahi.scripts.slice_coco import slice

class Slice_dataset(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slice_dataset")

        self.folder_labels = []
        self.folder_inputs = []
        self.value_labels = []
        self.value_inputs = []
        layout =QtWidgets.QVBoxLayout()

        type_label = QtWidgets.QLabel("Type:")
        self.type_combobox = QtWidgets.QComboBox()
        self.type_combobox.addItem("coco")
        layout.addWidget(type_label)
        layout.addWidget(self.type_combobox)

        self.json_label = QtWidgets.QLabel(f"标注文件:")
        self.json_input = QtWidgets.QLineEdit()
        self.json_input.setReadOnly(True)
        json_button = QtWidgets.QPushButton("Select")
        json_button.clicked.connect(self.select_json_file)
        json_layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.json_label)
        json_layout.addWidget(self.json_input)
        json_layout.addWidget(json_button)
        layout.addLayout(json_layout)

        folder_names = ["图像路径", "输出路径"]
        for i in range(2):
            folder_label = QtWidgets.QLabel(f"{folder_names[i]}:")
            folder_input = QtWidgets.QLineEdit()
            folder_input.setReadOnly(True)
            folder_button = QtWidgets.QPushButton("Select")
            folder_button.clicked.connect(lambda _, index=i: self.select_folder(index))
            self.folder_labels.append(folder_label)
            self.folder_inputs.append(folder_input)
            folder_layout = QtWidgets.QHBoxLayout()
            folder_layout.addWidget(folder_input)
            folder_layout.addWidget(folder_button)
            layout.addWidget(folder_label)
            layout.addLayout(folder_layout)
        # # 添加数值输入控件
        value_names = ["slice_size", "overlap_ratio"]
        value_layout = QtWidgets.QHBoxLayout()
        for i in range(2):
            value_label = QtWidgets.QLabel(f"{value_names[i]}:")
            value_input = QtWidgets.QLineEdit()

            self.value_labels.append(value_label)
            self.value_inputs.append(value_input)

            value_sub_layout = QtWidgets.QVBoxLayout()
            value_sub_layout.addWidget(value_label)
            value_sub_layout.addWidget(value_input)

            value_layout.addLayout(value_sub_layout)

        layout.addLayout(value_layout)

        self.result_label = QtWidgets.QLabel("Result:")
        self.result_text_edit = QtWidgets.QTextEdit()
        self.result_text_edit.setReadOnly(True)

        layout.addWidget(self.result_label)
        layout.addWidget(self.result_text_edit)

        start_button = QtWidgets.QPushButton("Slice")
        start_button.clicked.connect(self.slice)

        layout.addWidget(start_button)

        self.setLayout(layout)

    def select_folder(self, index):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_inputs[index].setText(folder)
    def select_json_file(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file_dialog.setNameFilter('Text files (*.json);;All files (*.*)')
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()
            self.json_input.setText(file_path[0])
    def slice(self):
        type_data = self.type_combobox.currentText()

        folder_data = [folder_input.text() for folder_input in self.folder_inputs]
        value_data = [value_input.text() for value_input in self.value_inputs]
        if type_data=='coco':
            result = slice(folder_data[0], self.json_input.text(), int(value_data[0]), float(value_data[1]), True, folder_data[1])
            self.result_text_edit.setText(result)
