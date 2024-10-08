from qtpy import QtWidgets
from .. import dataset
import concurrent.futures
import cv2
import os
import av


class Video_slice_Dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video_slice")

        self.value_labels = []
        self.value_inputs = []
        layout = QtWidgets.QVBoxLayout()
        # 添加下拉选择控件
        type_label = QtWidgets.QLabel("Type:")
        self.type_combobox = QtWidgets.QComboBox()
        self.type_combobox.addItem("H265")
        self.type_combobox.addItem("H264")


        layout.addWidget(type_label)
        layout.addWidget(self.type_combobox)

        video_input_label = QtWidgets.QLabel("video_input}:")
        self.video_input = QtWidgets.QLineEdit()
        self.video_input.setReadOnly(True)
        video_input_button = QtWidgets.QPushButton("Select")
        video_input_button.clicked.connect(self.select_video_file)
        video_input_layout = QtWidgets.QHBoxLayout()
        video_input_layout.addWidget(self.video_input)
        video_input_layout.addWidget(video_input_button)
        layout.addWidget(video_input_label)
        layout.addLayout(video_input_layout)

        image_output_label = QtWidgets.QLabel("output_dir:")
        self.image_output = QtWidgets.QLineEdit()
        self.image_output.setReadOnly(True)
        image_output_button = QtWidgets.QPushButton("Select")
        image_output_button.clicked.connect(self.select_folder)
        image_output_layout = QtWidgets.QHBoxLayout()
        image_output_layout.addWidget(self.image_output)
        image_output_layout.addWidget(image_output_button)
        layout.addWidget(image_output_label)
        layout.addLayout(image_output_layout)

        # 添加数值输入控件
        value_names = ["step"]
        value_layout = QtWidgets.QHBoxLayout()
        for i in range(1):
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

    def select_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.image_output.setText(folder)

    def select_video_file(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file_dialog.setNameFilter('Video files (*.mp4 *.avi *.mov *.mkv *.flv *.h265 *.H265);;All files (*.*)')
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()
            self.video_input.setText(file_path[0])

    def slice_h264(self, input_file, out_dir, step):
        def save_frame(frame, count):
            video_name = os.path.splitext(os.path.basename(input_file))[0]
            file_name = '{}_{:d}.jpg'.format(video_name, count)
            file_path = os.path.join(out_dir, file_name)
            cv2.imwrite(file_path, frame)

        try:
            cap = cv2.VideoCapture(input_file)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                count = 0
                frame_number = 0
                futures = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        if frame_number % int(step) == 0:
                            futures.append(executor.submit(save_frame, frame, count))
                            count += 1
                        frame_number += 1
                    else:
                        break
                concurrent.futures.wait(futures)
            cap.release()
            return "DONE"
        except:
            return "ERROR"

    def slice_h265(self, input_file, out_dir, step):
        def save_frame(frame, count):
            video_name = os.path.splitext(os.path.basename(input_file))[0]
            file_name = '{}_{:d}.jpg'.format(video_name, count)
            file_path = os.path.join(out_dir, file_name)
            frame.to_image().save(file_path)

        container = av.open(input_file)
        stream = container.streams.video[0]
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                count = 0
                frame_number = 0
                futures = []
                for frame in container.decode(stream):
                    if frame_number % int(step) == 0:
                        futures.append(executor.submit(save_frame, frame, count))
                        count += 1
                    frame_number += 1
                concurrent.futures.wait(futures)
            return "DONE"
        except:
            return "ERROR"

    def start(self):
        value_data = [value_input.text() for value_input in self.value_inputs]
        type_data = self.type_combobox.currentText()
        if type_data == 'H264':
            result = self.slice_h264(self.video_input.text(), self.image_output.text(), value_data[0])
        elif type_data == 'H265':
            result = self.slice_h265(self.video_input.text(), self.image_output.text(), value_data[0])
        self.result_text_edit.setText(result)
