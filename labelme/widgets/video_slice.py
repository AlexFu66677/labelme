from qtpy import QtWidgets
from .. import dataset
import concurrent.futures
import cv2
import os
import av
import random


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

        self.staged_sampling_checkbox = QtWidgets.QCheckBox(
            "Enable staged random + continuous sampling"
        )
        self.staged_sampling_checkbox.toggled.connect(
            self.update_staged_sampling_controls
        )
        layout.addWidget(self.staged_sampling_checkbox)

        stable_layout = QtWidgets.QHBoxLayout()
        self.stable_frames_label = QtWidgets.QLabel("Continuous frames:")
        self.stable_frames_input = QtWidgets.QSpinBox()
        self.stable_frames_input.setRange(1, 1000000)
        self.stable_frames_input.setValue(100)
        self.stable_frames_input.setToolTip(
            "The middle section uses min(total frames / 3, this value) consecutive frames."
        )
        stable_layout.addWidget(self.stable_frames_label)
        stable_layout.addWidget(self.stable_frames_input)
        layout.addLayout(stable_layout)
        self.update_staged_sampling_controls(False)

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
        file_dialog.setNameFilter('Video files (*.mp4 *.avi *.mov *.mkv *.flv *.h265 *.H265 *.h264);;All files (*.*)')
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()
            self.video_input.setText(file_path[0])

    def select_videopath_file(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        file_path = file_dialog.getExistingDirectory(None, "Select Folder", "")
        if file_path:
            self.videopath_input.setText(file_path)

    def update_staged_sampling_controls(self, enabled):
        self.stable_frames_label.setEnabled(enabled)
        self.stable_frames_input.setEnabled(enabled)

    @staticmethod
    def build_staged_frame_indices(total_frames, step, requested_stable_frames):
        """Build sparse-random / continuous / sparse-random frame indices."""
        if total_frames <= 0:
            return set()
        if step <= 0:
            raise ValueError("step must be greater than 0")

        stable_count = min(total_frames // 3, requested_stable_frames)
        remaining = total_frames - stable_count
        leading_count = remaining // 2
        stable_start = leading_count
        stable_end = stable_start + stable_count

        indices = set(range(stable_start, stable_end))
        rng = random.SystemRandom()

        # Each fixed-size window contributes one random frame.  The window
        # starts (0, step, 2*step, ...) remain fixed, avoiding repeated actions.
        for region_start, region_end in (
            (0, stable_start),
            (stable_end, total_frames),
        ):
            window_start = region_start
            while window_start < region_end:
                window_size = min(step, region_end - window_start)
                indices.add(window_start + rng.randrange(window_size))
                window_start += step
        return indices

    def slice_h264(
        self, input_file, out_dir, step, staged_sampling=False, stable_frames=100
    ):
        def save_frame(frame, count):
            video_name = os.path.splitext(os.path.basename(input_file))[0]
            file_name = '{}_{:d}.jpg'.format(video_name, count)
            file_path = os.path.join(out_dir, file_name)
            cv2.imwrite(file_path, frame)

        try:
            cap = cv2.VideoCapture(input_file)
            if not cap.isOpened():
                return "ERROR: unable to open video"
            selected_indices = None
            if staged_sampling:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    total_frames = 0
                    while cap.grab():
                        total_frames += 1
                    cap.release()
                    cap = cv2.VideoCapture(input_file)
                selected_indices = self.build_staged_frame_indices(
                    total_frames, int(step), stable_frames
                )
            with concurrent.futures.ThreadPoolExecutor() as executor:
                count = 0
                frame_number = 0
                futures = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        should_save = (
                            frame_number in selected_indices
                            if selected_indices is not None
                            else frame_number % int(step) == 0
                        )
                        if should_save:
                            futures.append(executor.submit(save_frame, frame, count))
                            count += 1
                        frame_number += 1
                    else:
                        break
                concurrent.futures.wait(futures)
            cap.release()
            return "DONE"
        except Exception as e:
            return f"ERROR: {e}"

    def slice_h265(
        self, input_file, videopath_input, out_dir, step,
        staged_sampling=False, stable_frames=100
    ):
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

        def get_video_stream(container):
            """Return the first video stream, or None for audio/invalid files."""
            video_streams = container.streams.video
            return video_streams[0] if video_streams else None

        def selected_for_container(container, stream):
            if not staged_sampling:
                return None, container, stream
            total_frames = stream.frames
            if not total_frames:
                total_frames = sum(1 for _ in container.decode(stream))
                container.close()
                container = av.open(input_file_for_selection)
                stream = get_video_stream(container)
                if stream is None:
                    raise ValueError("no video stream found after reopening input")
            selected = self.build_staged_frame_indices(
                int(total_frames), int(step), stable_frames
            )
            return selected, container, stream

        if os.path.isfile(input_file):
            input_file_for_selection = input_file
            try:
                container = av.open(input_file)
                stream = get_video_stream(container)
                if stream is None:
                    container.close()
                    return "ERROR: no video stream found in input file"
                selected_indices, container, stream = selected_for_container(
                    container, stream
                )
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    count = 0
                    frame_number = 0
                    futures = []
                    for frame in container.decode(stream):
                        should_save = (
                            frame_number in selected_indices
                            if selected_indices is not None
                            else frame_number % int(step) == 0
                        )
                        if should_save:
                            futures.append(executor.submit(save_frame, frame, count))
                            count += 1
                        frame_number += 1
                    concurrent.futures.wait(futures)
                container.close()
                return "DONE"
            except Exception as e:
                if "container" in locals():
                    container.close()
                return f"ERROR: {e}"
        if os.path.isdir(videopath_input):
            video_exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".h265", ".H265")
            for file in os.listdir(videopath_input):
                if file.endswith(video_exts):
                    input_path = os.path.join(videopath_input, file)
                    try:
                        container = av.open(input_path)
                        stream = get_video_stream(container)
                        if stream is None:
                            container.close()
                            continue
                        input_file_for_selection = input_path
                        selected_indices, container, stream = selected_for_container(
                            container, stream
                        )
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            count = 0
                            frame_number = 0
                            futures = []
                            for frame in container.decode(stream):
                                should_save = (
                                    frame_number in selected_indices
                                    if selected_indices is not None
                                    else frame_number % int(step) == 0
                                )
                                if should_save:
                                    futures.append(
                                        executor.submit(
                                            batch_save_frame,
                                            input_path,
                                            frame,
                                            count,
                                        )
                                    )
                                    count += 1
                                frame_number += 1
                            concurrent.futures.wait(futures)
                        container.close()
                    except Exception as e:
                        if "container" in locals():
                            container.close()
                        # One bad file should not prevent the remaining batch
                        # from being processed.
                        continue

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
        try:
            step = int(value_data[0])
            if step <= 0:
                raise ValueError
        except ValueError:
            self.result_text_edit.setText("ERROR: step must be a positive integer")
            return
        staged_sampling = self.staged_sampling_checkbox.isChecked()
        stable_frames = self.stable_frames_input.value()
        if type_data == 'H264':
            result = self.slice_h264(
                self.video_input.text(), output_dir, step,
                staged_sampling, stable_frames
            )
        elif type_data == 'H265':
            result = self.slice_h265(
                self.video_input.text(), self.videopath_input.text(), output_dir,
                step, staged_sampling, stable_frames
            )
        if self.merge_checkbox.isChecked() and result == "DONE":
            video_output_path = os.path.join(output_dir, "merged_output.mp4")
            merge_result = self.merge_images_to_video(output_dir, video_output_path)
            result += f"\n{merge_result}"
        self.result_text_edit.setText(result)
