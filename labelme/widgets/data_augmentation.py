from qtpy import QtWidgets, QtCore
import cv2
import os
import numpy as np


class Data_augmentation_Dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data_Augmentation")
        self.resize(400, 500)  # 调整窗口宽度
        self.folder_labels = []
        self.folder_inputs = []
        layout = QtWidgets.QVBoxLayout()

        # 添加下拉选择控件
        type_label = QtWidgets.QLabel("Type:")
        self.type_combobox = QtWidgets.QComboBox()
        self.type_combobox.addItem("YOLO_HBB")  # 添加选择类型
        self.type_combobox.addItem("YOLO_OBB")  # 添加选择类型
        layout.addWidget(type_label)
        layout.addWidget(self.type_combobox)

        # 添加输入输出文件夹选择
        folder_names = ["input", "output"]
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

        # 增强选项
        self.augmentation_layout = QtWidgets.QVBoxLayout()

        # 旋转（固定为90度）
        self.rotation_checkbox = QtWidgets.QCheckBox("Rotate (90°)")
        self.rotation_checkbox.stateChanged.connect(self.update_name)
        rotation_layout = QtWidgets.QHBoxLayout()  # 横向布局
        rotation_layout.addWidget(self.rotation_checkbox)
        self.augmentation_layout.addLayout(rotation_layout)

        # 翻转
        self.flip_checkbox = QtWidgets.QCheckBox("Flip")
        self.flip_checkbox.stateChanged.connect(self.update_name)
        self.flip_combobox = QtWidgets.QComboBox()
        self.flip_combobox.addItem("Horizontal")
        self.flip_combobox.addItem("Vertical")
        self.flip_combobox.currentIndexChanged.connect(self.update_name)
        flip_layout = QtWidgets.QHBoxLayout()  # 横向布局
        flip_layout.addWidget(self.flip_checkbox)
        flip_layout.addWidget(self.flip_combobox)
        self.augmentation_layout.addLayout(flip_layout)

        # 动态模糊
        self.motion_blur_checkbox = QtWidgets.QCheckBox("Motion Blur")
        self.motion_blur_checkbox.stateChanged.connect(self.update_name)
        self.motion_blur_slider = QtWidgets.QSpinBox()
        self.motion_blur_slider.setRange(1, 31)
        self.motion_blur_slider.setValue(3)
        motion_blur_layout = QtWidgets.QHBoxLayout()  # 横向布局
        motion_blur_layout.addWidget(self.motion_blur_checkbox)
        motion_blur_layout.addWidget(self.motion_blur_slider)
        self.augmentation_layout.addLayout(motion_blur_layout)

        # 高斯模糊
        self.gaussian_blur_checkbox = QtWidgets.QCheckBox("Gaussian Blur")
        self.gaussian_blur_checkbox.stateChanged.connect(self.update_name)
        self.gaussian_blur_slider = QtWidgets.QSpinBox()
        self.gaussian_blur_slider.setRange(1, 31)
        self.gaussian_blur_slider.setValue(3)
        gaussian_blur_layout = QtWidgets.QHBoxLayout()  # 横向布局
        gaussian_blur_layout.addWidget(self.gaussian_blur_checkbox)
        gaussian_blur_layout.addWidget(self.gaussian_blur_slider)
        self.augmentation_layout.addLayout(gaussian_blur_layout)

        # 高斯噪声
        self.gaussian_noise_checkbox = QtWidgets.QCheckBox("Gaussian Noise")
        self.gaussian_noise_checkbox.stateChanged.connect(self.update_name)
        self.gaussian_noise_slider = QtWidgets.QSpinBox()
        self.gaussian_noise_slider.setRange(1, 100)  # 设置噪声强度范围（1到100）
        self.gaussian_noise_slider.setValue(10)  # 默认值为10
        gaussian_noise_layout = QtWidgets.QHBoxLayout()  # 横向布局
        gaussian_noise_layout.addWidget(self.gaussian_noise_checkbox)
        gaussian_noise_layout.addWidget(self.gaussian_noise_slider)
        self.augmentation_layout.addLayout(gaussian_noise_layout)

        # JPG压缩
        self.jpg_compression_checkbox = QtWidgets.QCheckBox("JPG Compression")
        self.jpg_compression_checkbox.stateChanged.connect(self.update_name)
        self.jpg_quality_slider = QtWidgets.QSpinBox()
        self.jpg_quality_slider.setRange(10, 100)
        self.jpg_quality_slider.setValue(90)
        jpg_compression_layout = QtWidgets.QHBoxLayout()  # 横向布局
        jpg_compression_layout.addWidget(self.jpg_compression_checkbox)
        jpg_compression_layout.addWidget(self.jpg_quality_slider)
        self.augmentation_layout.addLayout(jpg_compression_layout)

        # 反相
        self.invert_checkbox = QtWidgets.QCheckBox("Invert Colors")
        self.invert_checkbox.stateChanged.connect(self.update_name)
        invert_layout = QtWidgets.QHBoxLayout()  # 横向布局
        invert_layout.addWidget(self.invert_checkbox)
        self.augmentation_layout.addLayout(invert_layout)

        # 灰度
        self.grayscale_checkbox = QtWidgets.QCheckBox("Grayscale")
        self.grayscale_checkbox.stateChanged.connect(self.update_name)
        grayscale_layout = QtWidgets.QHBoxLayout()  # 横向布局
        grayscale_layout.addWidget(self.grayscale_checkbox)
        self.augmentation_layout.addLayout(grayscale_layout)

        layout.addLayout(self.augmentation_layout)

        # 添加图像名称显示框
        filename_layout = QtWidgets.QHBoxLayout()  # 横向排列
        filename_label = QtWidgets.QLabel("New Name:")
        self.filename_display = QtWidgets.QLineEdit("XXXX")
        self.filename_display.setReadOnly(True)  # 设置为只读
        filename_layout.addWidget(filename_label)
        filename_layout.addWidget(self.filename_display)
        layout.addLayout(filename_layout)

        self.use_new_name_checkbox = QtWidgets.QCheckBox("Use new name (otherwise overwrite original)")
        self.use_new_name_checkbox.setChecked(True)  # 默认选中
        layout.addWidget(self.use_new_name_checkbox)

        # 添加进度条
        progress_layout = QtWidgets.QHBoxLayout()  # 横向布局
        self.progress_label = QtWidgets.QLabel("Progress:")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)

        # 开始按钮
        start_button = QtWidgets.QPushButton("Start")
        layout.addWidget(start_button)
        start_button.clicked.connect(self.start)
        self.setLayout(layout)

    def select_folder(self, index):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_inputs[index].setText(folder)

    def update_name(self):
        # 初始化名称
        name = "XXXX"

        if self.rotation_checkbox.isChecked():
            name += "_r"  # 旋转

        if self.flip_checkbox.isChecked():
            flip_type = self.flip_combobox.currentText()
            if flip_type == "Horizontal":
                name += "_flipx"
            elif flip_type == "Vertical":
                name += "_flipy"

        if self.motion_blur_checkbox.isChecked():
            name += "_m"  # 动态模糊

        if self.gaussian_blur_checkbox.isChecked():
            name += "_g"  # 高斯模糊

        if self.jpg_compression_checkbox.isChecked():
            name += "_j"  # JPG压缩

        if self.invert_checkbox.isChecked():
            name += "_i"  # 反相

        if self.grayscale_checkbox.isChecked():
            name += "_gray"  # 灰度

        if self.gaussian_noise_checkbox.isChecked():
            name += "_gn"
        # 更新显示框
        self.filename_display.setText(name)

    def start(self):
        input_folder = self.folder_inputs[0].text()
        output_folder = self.folder_inputs[1].text()

        if not input_folder or not output_folder:
            self.errorMessage("Error", "Please select both input and output folders.")
            return

        image_paths = self.get_images_from_folder(input_folder)

        if len(image_paths) == 0:
            self.errorMessage("Error", "No images found in the selected input folder.")
            return

        self.progress_bar.setValue(0)
        total_images = len(image_paths)

        # 检查选择的增强操作并应用
        image_paths = self.get_images_from_folder(input_folder)

        for index, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            filename = os.path.basename(image_path)
            txt_path = os.path.splitext(image_path)[0] + ".txt"  # 对应的YOLO标注
            yolo_data = self.load_yolo_labels(txt_path)

            if self.rotation_checkbox.isChecked():
                img, yolo_data = self.apply_rotation(img, yolo_data)

            if self.flip_checkbox.isChecked():
                flip_type = self.flip_combobox.currentText()
                img, yolo_data = self.apply_flip(img, yolo_data, flip_type)
            # 应用动态模糊
            if self.motion_blur_checkbox.isChecked():
                blur_size = self.motion_blur_slider.value()
                img = self.apply_motion_blur(img, blur_size)

            # 应用高斯模糊
            if self.gaussian_blur_checkbox.isChecked():
                blur_size = self.gaussian_blur_slider.value()
                img = self.apply_gaussian_blur(img, blur_size)

            if self.gaussian_noise_checkbox.isChecked():
                noise_level = self.gaussian_noise_slider.value()
                img = self.apply_gaussian_noise(img, noise_level)

            # 应用JPG压缩
            if self.jpg_compression_checkbox.isChecked():
                quality = self.jpg_quality_slider.value()
                img = self.apply_jpg_compression(img, quality)

            # 应用反相
            if self.invert_checkbox.isChecked():
                img = self.apply_invert(img)

            # 应用灰度
            if self.grayscale_checkbox.isChecked():
                img = self.apply_grayscale(img)

            # 更新文件名
            if self.use_new_name_checkbox.isChecked():
                enhanced_filename = self.filename_display.text()
                if "_" in enhanced_filename:
                    suffix = enhanced_filename.split("_", 1)[1]  # 提取 "_" 之后的所有内容
                else:
                    suffix = ""  # 如果没有找到 "_"，则没有后缀
                new_filename = os.path.splitext(filename)[0] + "_" + suffix + ".jpg"
            else:
                new_filename = filename  # 不使用新名称，覆盖原文件
            self.save_image_and_labels(img, yolo_data, output_folder, new_filename)

            progress = int((index + 1) / total_images * 100)
            self.progress_bar.setValue(progress)
        self.progress_bar.setValue(100)  # 完成后将进度条设置为100%
        QtWidgets.QMessageBox.information(self, "Done", "All images processed successfully!")

    def apply_rotation(self, img, yolo_data):
        """对图像和YOLO标注执行旋转90度操作"""
        # 图像顺时针旋转90度
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # 读取YOLO标注并旋转
        rotated_labels = []
        for label in yolo_data:
            rotated_label = self.rotate_yolo_label_90(label, img.shape)
            rotated_labels.append(rotated_label)
        return rotated_img, rotated_labels

    def rotate_yolo_label_90(self, label, img_shape):
        """旋转YOLO格式标注，90度顺时针"""
        type_data = self.type_combobox.currentText()
        if type_data == 'YOLO_HBB':
            x, y, w, h = label[1:]
            return [label[0], 1 - y, x, h, w]
        if type_data == 'YOLO_OBB':
            x1, y1, x2, y2, x3, y3, x4, y4 = label[1:]
            return [label[0], 1 - y1, x1, 1 - y2, x2,1 - y3, x3,1 - y4, x4]

    def apply_gaussian_noise(self, img, noise_level):
        """给图像添加高斯噪声"""
        row, col, ch = img.shape
        mean = 0
        sigma = noise_level / 100  # 将滑块值转换为标准差
        gauss = np.random.normal(mean, sigma, (row, col, ch))  # 生成高斯噪声
        noisy_img = img + gauss * 255  # 调整噪声的影响
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)  # 确保图像像素值在0-255之间
        return noisy_img

    def apply_flip(self, img, yolo_data, flip_type):
        """翻转图像和标注"""
        if flip_type == "Horizontal":
            flipped_img = cv2.flip(img, 1)  # 水平翻转
            flipped_labels = self.flip_yolo_labels_horizontal(yolo_data)
        else:
            flipped_img = cv2.flip(img, 0)  # 垂直翻转
            flipped_labels = self.flip_yolo_labels_vertical(yolo_data)

        return flipped_img, flipped_labels

    def flip_yolo_labels_horizontal(self, yolo_data):
        """水平翻转YOLO标注"""
        type_data = self.type_combobox.currentText()
        if type_data == 'YOLO_HBB':
            flipped_data = []
            for label in yolo_data:
                x, y, w, h = label[1:]
                flipped_data.append([label[0], 1 - x, y, w, h])
            return flipped_data
        if type_data == 'YOLO_OBB':
            flipped_data = []
            for label in yolo_data:
                x1, y1, x2, y2, x3, y3, x4, y4 = label[1:]
                flipped_data.append([label[0], 1 - x1, y1, 1 - x2, y2, 1 - x3, y3, 1 - x4, y4])
            return flipped_data

    def flip_yolo_labels_vertical(self, yolo_data):
        """垂直翻转YOLO标注"""
        type_data = self.type_combobox.currentText()
        if type_data == 'YOLO_HBB':
            flipped_data = []
            for label in yolo_data:
                x, y, w, h = label[1:]
                flipped_data.append([label[0], x, 1 - y, w, h])
            return flipped_data
        if type_data == 'YOLO_OBB':
            flipped_data = []
            for label in yolo_data:
                x1, y1, x2, y2, x3, y3, x4, y4 = label[1:]
                flipped_data.append([label[0], x1, 1 - y1, x2, 1 - y2, x3, 1 - y3, x4, 1 - y4])
            return flipped_data

    def load_yolo_labels(self, txt_path):
        """加载YOLO格式的标注"""
        with open(txt_path, "r") as f:
            lines = f.readlines()
        yolo_data = []
        for line in lines:
            parts = line.split()
            parts[0] = int(float(parts[0]))  # 将第一个数据转换为int
            parts[1:] = map(float, parts[1:])  # 将其余数据转换为float
            yolo_data.append(parts)
        return yolo_data

    def apply_invert(self, img):
        """反相图像"""
        return cv2.bitwise_not(img)

    def apply_grayscale(self, img):
        """将图像转换为灰度"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_motion_blur(self, img, size):
        """应用动态模糊效果"""
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        return cv2.filter2D(img, -1, kernel_motion_blur)

    def apply_gaussian_blur(self, img, size):
        """应用高斯模糊效果"""
        return cv2.GaussianBlur(img, (size, size), 0)

    def apply_jpg_compression(self, img, quality):
        """应用JPG压缩效果"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        is_success, im_buf_arr = cv2.imencode(".jpg", img, encode_param)
        img = cv2.imdecode(im_buf_arr, cv2.IMREAD_COLOR)
        return img

    def get_images_from_folder(self, folder):
        """从文件夹中获取所有图像路径"""
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def save_image_and_labels(self, img, yolo_data, output_folder, new_filename):
        """保存处理后的图像和标注"""
        # 保存图像
        image_output_path = os.path.join(output_folder, new_filename)
        cv2.imwrite(image_output_path, img)

        # 保存YOLO标注文件
        output_label_path = os.path.join(output_folder, os.path.splitext(image_output_path)[0] + ".txt")
        with open(output_label_path, "w") as f:
            for label in yolo_data:
                label_str = " ".join(map(str, label))
                f.write(label_str + "\n")

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(self, title, f"<p><b>{title}</b></p>{message}")
