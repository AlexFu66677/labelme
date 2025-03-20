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

        video_input_label = QtWidgets.QLabel("video_input:")
        self.video_input = QtWidgets.QLineEdit()
        self.video_input.setReadOnly(True)
        video_input_button = QtWidgets.QPushButton("Select")
        video_input_button.clicked.connect(self.select_video_file)
        video_input_layout = QtWidgets.QHBoxLayout()
        video_input_layout.addWidget(self.video_input)
        video_input_layout.addWidget(video_input_button)
        layout.addWidget(video_input_label)
        layout.addLayout(video_input_layout)

        videopath_label = QtWidgets.QLabel("video_path_input:")
        self.videopath_input = QtWidgets.QLineEdit()
        self.videopath_input.setReadOnly(True)
        videopath_input_button = QtWidgets.QPushButton("Select")
        videopath_input_button.clicked.connect(self.select_videopath_file)
        videopath_input_layout = QtWidgets.QHBoxLayout()
        videopath_input_layout.addWidget(self.videopath_input)
        videopath_input_layout.addWidget(videopath_input_button)
        layout.addWidget(videopath_label)
        layout.addLayout(videopath_input_layout)

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

        self.merge_checkbox = QtWidgets.QCheckBox("Merge to MP4")
        layout.addWidget(self.merge_checkbox)

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

    def select_videopath_file(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        file_path = file_dialog.getExistingDirectory(None, "Select Folder", "")
        if file_path:
            self.videopath_input.setText(file_path)

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

    def slice_h265(self, input_file, videopath_input, out_dir, step):
        def save_frame(frame, count):
            video_name = os.path.splitext(os.path.basename(input_file))[0]
            file_name = '{}_{:d}.jpg'.format(video_name, count)
            file_path = os.path.join(out_dir, file_name)
            frame.to_image().save(file_path)

        def batch_save_frame(file_name, frame, count):
            video_name = os.path.splitext(os.path.basename(file_name))[0]
            file_name = '{}_{:d}.jpg'.format(video_name, count)
            file_path = os.path.join(out_dir, file_name)
            frame.to_image().save(file_path)

        if os.path.isfile(input_file):
            container = av.open(input_file)
            stream = container.streams.video[0]
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    count = 0
                    frame_number = 0
                    futures = []
                    for frame in container.decode(stream):
                        count += 1
                        if frame_number % int(step) == 0:
                            futures.append(executor.submit(save_frame, frame, count))
                        frame_number += 1
                    concurrent.futures.wait(futures)
                return "DONE"
            except:
                return "ERROR"
        if os.path.isdir(videopath_input):
            video_exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".h265", ".H265")
            for file in os.listdir(videopath_input):
                if file.endswith(video_exts):
                    input_path = os.path.join(videopath_input, file)
                    container = av.open(input_path)
                    stream = container.streams.video[0]
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            count = 0
                            frame_number = 0
                            futures = []
                            for frame in container.decode(stream):
                                count += 1
                                if frame_number % int(step) == 0:
                                    futures.append(executor.submit(batch_save_frame, input_path, frame, count))
                                frame_number += 1
                            concurrent.futures.wait(futures)
                    except:
                        frame_number = 0

            return "DONE"

    def merge_images_to_video(self, image_dir, output_video):
        try:
            images = sorted(
                [img for img in os.listdir(image_dir) if img.endswith(".jpg")],
                key=lambda x: int(x.split('_')[-1].split('.')[0])  # 按序号排序
            )
            if not images:
                return "No images to merge"
            first_image_path = os.path.join(image_dir, images[0])
            first_frame = cv2.imread(first_image_path)
            height, width, layers = first_frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

            for image in images:
                frame = cv2.imread(os.path.join(image_dir, image))
                out.write(frame)

            out.release()
            return "Video merged successfully"
        except Exception as e:
            return f"Error merging video: {str(e)}"

    def start(self):
        value_data = [value_input.text() for value_input in self.value_inputs]
        type_data = self.type_combobox.currentText()
        output_dir = self.image_output.text()
        if type_data == 'H264':
            result = self.slice_h264(self.video_input.text(), self.videopath_input.text(), self.image_output.text(),
                                     value_data[0])
        elif type_data == 'H265':
            result = self.slice_h265(self.video_input.text(), self.videopath_input.text(), self.image_output.text(),
                                     value_data[0])
        if self.merge_checkbox.isChecked() and result == "DONE":
            video_output_path = os.path.join(output_dir, "merged_output.mp4")
            merge_result = self.merge_images_to_video(output_dir, video_output_path)
            result += f"\n{merge_result}"
        self.result_text_edit.setText(result)
