from qtpy import QtWidgets, QtGui, QtCore
import os
import random

class Yolo_Vis_Dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset")

        self.default_image_size = 640  # Default size for the longest side
        self.folder_labels = []
        self.folder_inputs = []
        layout = QtWidgets.QVBoxLayout()
        folder_names = ["label_input", "image_input"]
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

        # Add navigation buttons
        navigation_layout = QtWidgets.QHBoxLayout()
        prev_button = QtWidgets.QPushButton("Previous")
        prev_button.clicked.connect(self.show_prev_image)
        next_button = QtWidgets.QPushButton("Next")
        next_button.clicked.connect(self.show_next_image)
        navigation_layout.addWidget(prev_button)
        navigation_layout.addWidget(next_button)

        layout.addLayout(navigation_layout)

        # Add image size input field and current image name label
        size_layout = QtWidgets.QHBoxLayout()
        size_input_label = QtWidgets.QLabel("Image Size:")
        self.size_input = QtWidgets.QLineEdit(str(self.default_image_size))
        self.size_input.setValidator(QtGui.QIntValidator(1, 4096))  # Allow valid size range from 1 to 4096
        size_layout.addWidget(size_input_label)
        size_layout.addWidget(self.size_input)

        self.image_name_label = QtWidgets.QLabel("No image loaded")
        size_layout.addWidget(self.image_name_label)

        layout.addLayout(size_layout)

        # Add checkbox for random order
        self.random_order_checkbox = QtWidgets.QCheckBox("Random Order")
        layout.addWidget(self.random_order_checkbox)

        # Add a label to show the images
        self.image_label = QtWidgets.QLabel()
        self.image_label.setStyleSheet("background-color: white;")
        layout.addWidget(self.image_label)

        start_button = QtWidgets.QPushButton("Start")
        start_button.clicked.connect(self.start)

        layout.addWidget(start_button)

        self.setLayout(layout)

        self.image_paths = []
        self.current_index = 0

        self.shortcut_prev = QtWidgets.QShortcut(QtGui.QKeySequence("A"), self)
        self.shortcut_prev.activated.connect(self.show_prev_image)
        self.shortcut_next = QtWidgets.QShortcut(QtGui.QKeySequence("D"), self)
        self.shortcut_next.activated.connect(self.show_next_image)

    def select_folder(self, index):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_inputs[index].setText(folder)

    def start(self):
        image_folder = self.folder_inputs[1].text()
        if os.path.exists(image_folder):
            self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
            if self.random_order_checkbox.isChecked():
                random.shuffle(self.image_paths)
            else:
                self.image_paths.sort()
            self.current_index = 0
            if self.image_paths:
                self.update_image_size()
                self.display_image_with_annotations(self.image_paths[self.current_index])

    def show_prev_image(self):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.display_image_with_annotations(self.image_paths[self.current_index])

    def show_next_image(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.display_image_with_annotations(self.image_paths[self.current_index])

    def update_image_size(self):
        # This function is called to update the size of the image display based on the input value
        self.max_size = int(self.size_input.text())

    def display_image_with_annotations(self, image_path):
        # Load the image
        pixmap = QtGui.QPixmap(image_path)
        img_width = pixmap.width()
        img_height = pixmap.height()

        # Update image name label
        self.image_name_label.setText(os.path.basename(image_path))

        # Determine the scaling factor based on the max size input
        if img_width > img_height:
            scale_factor = self.max_size / img_width
        else:
            scale_factor = self.max_size / img_height

        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        # Scale the image to fit the new dimensions while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(new_width, new_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        # Create a white background image with the new dimensions
        result_pixmap = QtGui.QPixmap(new_width, new_height)
        result_pixmap.fill(QtCore.Qt.white)

        # Center the scaled image
        painter = QtGui.QPainter(result_pixmap)
        x_offset = (new_width - scaled_pixmap.width()) // 2
        y_offset = (new_height - scaled_pixmap.height()) // 2
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)

        # Get the corresponding label file
        label_folder = self.folder_inputs[0].text()
        label_path = os.path.join(label_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

        if os.path.exists(label_path):
            # Create a pen to draw the bounding boxes
            pen = QtGui.QPen(QtCore.Qt.red)
            pen.setWidth(2)
            painter.setPen(pen)

            font = QtGui.QFont()
            font.setPointSize(12)
            painter.setFont(font)

            with open(label_path, 'r') as f:
                for line in f:
                    cls, x_center, y_center, width, height = map(float, line.split())
                    cls = int(cls)  # Ensure the class label is an integer
                    label = cls
                    x_center *= scaled_pixmap.width()
                    y_center *= scaled_pixmap.height()
                    width *= scaled_pixmap.width()
                    height *= scaled_pixmap.height()

                    # Calculate the bounding box coordinates
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Draw the bounding box
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                    # Draw the label text at the top-left corner of the bounding box
                    painter.drawText(x1, y1 - 10, str(label))

        painter.end()

        # Display the image with annotations
        self.image_label.setPixmap(result_pixmap)
        self.image_label.setFixedSize(new_width, new_height)
        self.image_label.adjustSize()