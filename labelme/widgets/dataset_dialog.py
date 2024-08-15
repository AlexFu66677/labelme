
from qtpy import QtWidgets
from .. import dataset


class DatasetDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset")

        self.folder_labels = []
        self.folder_inputs = []
        self.value_labels = []
        self.value_inputs = []
        layout =QtWidgets.QVBoxLayout()
        # 添加下拉选择控件
        type_label = QtWidgets.QLabel("Type:")
        self.type_combobox = QtWidgets.QComboBox()
        self.type_combobox.addItem("coco")
        self.type_combobox.addItem("voc")
        self.type_combobox.addItem("yolo")

        layout.addWidget(type_label)
        layout.addWidget(self.type_combobox)

        folder_names = ["json_input", "image_input", "output"]
        for i in range(3):
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

        # 添加数值输入控件
        value_names = ["train", "test", "val"]
        value_layout = QtWidgets.QHBoxLayout()
        for i in range(3):
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

        start_button = QtWidgets.QPushButton("Start")
        start_button.clicked.connect(self.start)

        layout.addWidget(start_button)

        self.setLayout(layout)

    def select_folder(self, index):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_inputs[index].setText(folder)
    def start(self):
        folder_data = [folder_input.text() for folder_input in self.folder_inputs]
        value_data = [value_input.text() for value_input in self.value_inputs]
        type_data = self.type_combobox.currentText()
        if type_data=='coco':
           result = dataset.CocoGenerator(folder_data, value_data)
        elif type_data=='yolo':
           result =dataset.YoloGenerator(folder_data, value_data)
        elif type_data == 'voc':
           result = dataset.VocGenerator(folder_data, value_data)
        #键值对
        # folder_data=['C:/Users/fjl\Desktop\data/anno', 'C:/Users/fjl/Desktop/data/img', 'C:/Users/fjl\Desktop\data']
        # value_data=[0.4,0.3,0.3]
        # result = datset.Generator(folder_data,value_data)
        self.result_text_edit.setText(result)
