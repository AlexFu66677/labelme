from qtpy import QtWidgets
from .. import dataset
from sahi.scripts.slice_coco import slice
import cv2
from pathlib import Path

class Slice_dataset(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slice_dataset")

        self.folder_labels = []
        self.folder_inputs = []
        self.value_labels = []
        self.value_inputs = []
        layout = QtWidgets.QVBoxLayout()

        type_label = QtWidgets.QLabel("Type:")
        self.type_combobox = QtWidgets.QComboBox()
        self.type_combobox.addItem("coco")
        self.type_combobox.addItem("yolo")
        layout.addWidget(type_label)
        layout.addWidget(self.type_combobox)

        self.json_label = QtWidgets.QLabel(f"标注文件:")
        self.json_input = QtWidgets.QLineEdit()
        self.json_input.setReadOnly(True)
        self.json_button = QtWidgets.QPushButton("Select")
        self.type_combobox.currentIndexChanged.connect(self.update_json_button_connection)
        self.update_json_button_connection()
        json_layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.json_label)
        json_layout.addWidget(self.json_input)
        json_layout.addWidget(self.json_button)
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

    def update_json_button_connection(self):
        try:
            self.json_button.clicked.disconnect()
        except TypeError:
            pass
        if self.type_combobox.currentText() == 'coco':
            self.json_button.clicked.connect(self.select_json_file)
        elif self.type_combobox.currentText() == 'yolo':
            self.json_button.clicked.connect(self.select_input_folder)

    def select_input_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.json_input.setText(folder)

    def yolo_slice(self, image_dir, label_dir, slice_size, overlap_ratio, out_dir):
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        out_dir = Path(out_dir)

        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        def process_file(image_file, label_file):
            image = cv2.imread(str(image_file))
            slice_height, slice_width = slice_size,slice_size
            height, width, _ = image.shape
            if height <= slice_height and width <= slice_width:
                return
            with open(label_file, 'r') as f:
                labels = f.readlines()
            stride_x = int(slice_width * (1 - overlap_ratio))
            stride_y = int(slice_height * (1 - overlap_ratio))

            for y in range(0, height, stride_y):
                for x in range(0, width, stride_x):
                    # Adjust x and y to make sure the slice covers the remaining area
                    if x + slice_width > width:
                        x = width - slice_width
                    if y + slice_height > height:
                        y = height - slice_height

                    # Ensure the slice doesn't go out of image bounds
                    end_x = min(x + slice_width, width)
                    end_y = min(y + slice_height, height)

                    img_slice = image[y:end_y, x:end_x]
                    slice_filename = f"{image_file.stem}_{x}_{y}.jpg"
                    slice_path = out_dir / slice_filename
                    cv2.imwrite(str(slice_path), img_slice)

                    label_slice = []
                    for label in labels:
                        parts = label.strip().split()
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])

                        # Convert YOLO coordinates to absolute coordinates
                        x_center_abs = x_center * width
                        y_center_abs = y_center * height
                        bbox_width_abs = bbox_width * width
                        bbox_height_abs = bbox_height * height

                        # Calculate bounding box edges
                        x_min_abs = x_center_abs - (bbox_width_abs / 2)
                        x_max_abs = x_center_abs + (bbox_width_abs / 2)
                        y_min_abs = y_center_abs - (bbox_height_abs / 2)
                        y_max_abs = y_center_abs + (bbox_height_abs / 2)

                        # Compute intersection with slice
                        x_min_slice = max(x_min_abs, x)
                        x_max_slice = min(x_max_abs, end_x)
                        y_min_slice = max(y_min_abs, y)
                        y_max_slice = min(y_max_abs, end_y)

                        # Check if there is an intersection
                        if x_min_slice < x_max_slice and y_min_slice < y_max_slice:
                            # Calculate clipped bounding box dimensions
                            clipped_width = x_max_slice - x_min_slice
                            clipped_height = y_max_slice - y_min_slice
                            clipped_x_center = (x_min_slice + clipped_width / 2 - x) / (end_x - x)
                            clipped_y_center = (y_min_slice + clipped_height / 2 - y) / (end_y - y)
                            clipped_bbox_width = clipped_width / (end_x - x)
                            clipped_bbox_height = clipped_height / (end_y - y)

                            # Add the adjusted label to the slice
                            label_slice.append(
                                f"{class_id} {clipped_x_center} {clipped_y_center} {clipped_bbox_width} {clipped_bbox_height}")

                    if label_slice:
                        label_filename = f"{image_file.stem}_{x}_{y}.txt"
                        label_path = out_dir / label_filename
                        with open(label_path, 'w') as f:
                            f.write("\n".join(label_slice))

        # Define the image formats you want to support
        image_formats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]

        for fmt in image_formats:
            for image_file in image_dir.glob(fmt):
                label_file = label_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    process_file(image_file, label_file)
        return "Done"
    def slice(self):
        type_data = self.type_combobox.currentText()

        folder_data = [folder_input.text() for folder_input in self.folder_inputs]
        value_data = [value_input.text() for value_input in self.value_inputs]
        if type_data == 'coco':
            result = slice(folder_data[0], self.json_input.text(), int(value_data[0]), float(value_data[1]), True,
                           folder_data[1])
            self.result_text_edit.setText(result)
        if type_data == 'yolo':
            result = self.yolo_slice(folder_data[0], self.json_input.text(), int(value_data[0]), float(value_data[1]),
                           folder_data[1])
            self.result_text_edit.setText(result)

