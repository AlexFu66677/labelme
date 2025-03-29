import sys
import os
import json
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QImage, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, \
    QFileDialog, QPushButton, QListWidget, QSplitter, QCheckBox, QButtonGroup, QSpacerItem, QSizePolicy, QMenu, QAction, \
    QListWidgetItem, QGridLayout, QComboBox,QLineEdit,QMessageBox,QDialog
from PyQt5.QtGui import QPen, QColor, QKeySequence, QCursor
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QShortcut
import cv2
import numpy as np
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import QRect


class copy_paste_augmentation_Dialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据增强")
        self.setGeometry(100, 100, 1400, 1000)

        # 创建主布局
        main_layout = QHBoxLayout()
        self.main_splitter = QSplitter(Qt.Horizontal)

        # 创建左侧布局
        self.view_type = 'medium'  # 默认图标大小

        left_layout = QVBoxLayout()
        self.load_folder_button = QPushButton("加载切片文件夹")
        self.load_folder_button.clicked.connect(self.load_slice_folder)
        left_layout.addWidget(self.load_folder_button)
        self.slice_image_list_widget = QListWidget()
        self.slice_image_list_widget.setViewMode(QListWidget.IconMode)  # 以图标模式显示
        self.slice_image_list_widget.setIconSize(QPixmap(200, 200).size())  # 设置缩略图大小
        self.slice_image_list_widget.setResizeMode(QListWidget.Adjust)  # 自动调整
        self.slice_image_list_widget.setSpacing(10)  # 设置间距
        self.slice_image_list_widget.clicked.connect(self.switch_slice_image_from_list)
        size_control_layout = QHBoxLayout()
        self.thumbnail_size_input = QLineEdit()
        self.thumbnail_size_input.setPlaceholderText("缩略图大小(默认 200)")  # 提示信息
        size_control_layout.addWidget(QLabel("size:"))
        size_control_layout.addWidget(self.thumbnail_size_input)
        self.set_thumbnail_size_button = QPushButton("确定")
        self.set_thumbnail_size_button.clicked.connect(self.set_thumbnail_size)
        size_control_layout.addWidget(self.set_thumbnail_size_button)
        left_layout.addWidget(self.slice_image_list_widget)
        left_layout.addLayout(size_control_layout)
        left_widget = QWidget(self)
        left_widget.setLayout(left_layout)
        self.main_splitter.addWidget(left_widget)

        middle_layout = QVBoxLayout()
        self.middle_splitter = QSplitter(Qt.Vertical)  # 垂直分隔图像与标注

        # 创建滚动区域用于显示图像
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(True)

        # 创建融合方式选择区域
        fusion_layout = QHBoxLayout()
        self.original_fusion = QCheckBox("原始", self)
        self.feathering_fusion = QCheckBox("羽化", self)
        self.poisson_fusion = QCheckBox("泊松融合", self)
        self.fusion_group = QButtonGroup(self)
        self.fusion_group.addButton(self.original_fusion)
        self.fusion_group.addButton(self.feathering_fusion)
        self.fusion_group.addButton(self.poisson_fusion)
        self.fusion_group.setExclusive(True)  # 只能单选
        self.original_fusion.setChecked(True)  # 默认选择“原始”
        fusion_layout.addWidget(self.original_fusion)
        fusion_layout.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Fixed, QSizePolicy.Minimum))
        fusion_layout.addWidget(self.feathering_fusion)
        fusion_layout.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Fixed, QSizePolicy.Minimum))
        fusion_layout.addWidget(self.poisson_fusion)
        fusion_widget = QWidget(self)
        fusion_widget.setLayout(fusion_layout)
        fusion_widget.setFixedHeight(50)
        # 创建标注信息列表
        self.annotation_list_widget = QTableWidget(self)
        self.annotation_list_widget.setColumnCount(5)  # 5 列
        self.annotation_list_widget.setHorizontalHeaderLabels(["cls", "x1", "y1", "x2", "y2"])
        self.annotation_list_widget.setEditTriggers(QTableWidget.NoEditTriggers)  # 禁止编辑
        self.annotation_list_widget.setSelectionBehavior(QTableWidget.SelectRows)  # 选择整行
        self.annotation_list_widget.setAlternatingRowColors(True)  # 交替行颜色

        self.middle_splitter.addWidget(self.scroll_area)  # 图像区域
        self.middle_splitter.addWidget(fusion_widget)
        self.middle_splitter.addWidget(self.annotation_list_widget)  # 标注信息区域

        middle_layout.addWidget(self.middle_splitter)
        middle_widget = QWidget(self)
        middle_widget.setLayout(middle_layout)
        self.main_splitter.addWidget(middle_widget)  # 添加到主 splitter

        # 右侧区域（图像列表 + 按钮）
        right_layout = QVBoxLayout()
        self.image_list_widget = QListWidget(self)
        self.image_list_widget.clicked.connect(self.switch_image_from_list)
        right_layout.addWidget(self.image_list_widget)

        self.prev_button = QPushButton("上一张", self)
        self.prev_button.clicked.connect(self.show_previous_image)
        right_layout.addWidget(self.prev_button)
        prev_shortcut = QShortcut(QKeySequence("a"), self)
        prev_shortcut.activated.connect(self.show_previous_image)

        self.next_button = QPushButton("下一张", self)
        self.next_button.clicked.connect(self.show_next_image)
        right_layout.addWidget(self.next_button)
        next_shortcut = QShortcut(QKeySequence("d"), self)
        next_shortcut.activated.connect(self.show_next_image)

        self.save_button = QPushButton("保存", self)
        self.save_button.clicked.connect(self.save_image)
        right_layout.addWidget(self.save_button)
        save_shortcut = QShortcut(QKeySequence("w"), self)
        save_shortcut.activated.connect(self.save_image)

        self.load_folder_button = QPushButton("加载", self)
        self.load_folder_button.clicked.connect(self.load_folder)
        right_layout.addWidget(self.load_folder_button)

        right_widget = QWidget(self)
        right_widget.setLayout(right_layout)
        self.main_splitter.addWidget(right_widget)  # 添加到主 splitter
        # 在 main_splitter 和 left_splitter 添加默认尺寸
        self.main_splitter.setSizes([400, 1000, 400])  # 左侧占 600，右侧占 200
        self.middle_splitter.setSizes([800, 50, 350])  # 图像区域占 400，标注区域占 200

        # 监听 splitter 调整事件
        self.main_splitter.splitterMoved.connect(self.show_image)
        self.middle_splitter.splitterMoved.connect(self.show_image)

        # 设置主布局
        main_layout.addWidget(self.main_splitter)
        container = QWidget(self)
        container.setLayout(main_layout)
        # self.setCentralWidget(container)
        self.sample_list_widget = None
        self.grid_layout = QGridLayout()  # 网格布局
        self.annotation_list_widget.cellPressed.connect(self.handle_cell_pressed)
        # 初始化变量
        self.image_path = None
        self.image = None
        self.update_image = None
        self.image_files = []
        self.current_image_index = -1
        self.annotations = []
        self.is_moving_bbox = False
        self.mouse_click_position = None
        self.new_height = None
        self.new_width = None
        self.anno_data = None
        self.folder_path = None
        self.selected_bbox = None
        self.slice_folder = None
        self.selected_bbox_is_slice = False
        self.slice_image = None
        self.slice_image_list = None

    def set_thumbnail_size(self):
        """从输入框获取缩略图大小，并更新 QListWidget 的缩略图尺寸"""
        size_text = self.thumbnail_size_input.text()
        if size_text.isdigit():  # 确保输入是数字
            size = int(size_text)
            self.slice_image_list_widget.setIconSize(QPixmap(size, size).size())
            self.display_slice_images(self.slice_folder)# 更新缩略图大小
        else:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字！", QMessageBox.Ok)
    def load_slice_folder(self):
        self.slice_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.slice_folder:
            self.display_slice_images(self.slice_folder)

    def display_slice_images(self, folder):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        self.slice_image_list = [os.path.join(folder, f) for f in os.listdir(folder) if
                       any(f.lower().endswith(ext) for ext in valid_extensions)]
        self.slice_image_list_widget.clear()
        for image_file in self.slice_image_list:
            item = QListWidgetItem()
            pixmap = QPixmap(image_file)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 生成缩略图
            item.setIcon(QIcon(pixmap))
            self.slice_image_list_widget.addItem(item)

    def switch_slice_image_from_list(self):
        if self.slice_image_list_widget.selectedItems():
            self.selected_bbox_is_slice = True
            selected_item = self.slice_image_list_widget.selectedItems()[0]
            selected_index = self.slice_image_list_widget.row(selected_item)
            selected_image_name = self.slice_image_list[selected_index]
            cls = os.path.basename(selected_image_name).split('_')[0]
            image_path = os.path.join(self.slice_folder, selected_image_name) # 用 OpenCV 读取图像
            self.slice_image = QPixmap(image_path)
            self.selected_bbox = {"label": cls, "points": [[0, 0], [int(self.slice_image.width()), int(self.slice_image.height())]]}
            self.update_image_with_moved_bbox()

    def load_folder(self):
        """加载文件夹中的图像"""
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.folder_path:
            self.image_files = [f for f in os.listdir(self.folder_path) if
                                f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'xpm'))]
            self.image_list_widget.clear()
            self.image_list_widget.addItems(self.image_files)
            self.current_image_index = 0
            self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]),
                            self.folder_path)

    def get_image_position_in_label(self):
        """获取 QPixmap 在 QLabel 内的左上角坐标（相对 QLabel）"""
        if not self.image:
            return None

        label_width, label_height = self.label.width(), self.label.height()
        image_width, image_height = self.new_width, self.new_height
        # 计算图像在 QLabel 内的左上角偏移量
        offset_x = (label_width - image_width) // 2 if label_width > image_width else 0
        offset_y = (label_height - image_height) // 2 if label_height > image_height else 0

        return offset_x, offset_y

    def load_image(self, image_path, folder_path):
        """加载图像和对应的标注文件"""
        self.image = None
        self.update_image = None
        self.selected_bbox = None
        self.image_path = image_path
        self.image = QPixmap(image_path)
        self.load_annotations(os.path.splitext(image_path)[0] + '.json', folder_path)
        self.show_image()

    def handle_cell_pressed(self, row, column):
        """ 根据鼠标按键区分不同的操作 """
        mouse_button = QApplication.mouseButtons()  # 获取当前鼠标按键状态

        if mouse_button == Qt.LeftButton:
            self.highlight_selected_bbox(row, column)
        elif mouse_button == Qt.RightButton:
            self.highlight_selected_bbox(row, column)
            self.show_context_menu()

    def show_context_menu(self):
        """ 在右键点击时显示自定义菜单，并显示在鼠标点击位置的右下方 """
        context_menu = QMenu(self)

        # 创建菜单项
        action1 = context_menu.addAction("添加到样本库")
        action2 = context_menu.addAction("   ")

        # 连接菜单项的点击事件
        action1.triggered.connect(lambda: self.add_to_sample_library())
        action2.triggered.connect(lambda: self.menu_action2())

        # 获取鼠标当前位置，并将菜单显示在右下方
        pos = QCursor.pos()
        menu_pos = QPoint(pos.x(), pos.y())  # 设置菜单位置为鼠标点击位置
        context_menu.exec_(menu_pos)

    def add_to_sample_library(self):
        selected_bbox = self.selected_bbox
        if selected_bbox:
            x1, y1 = selected_bbox["points"][0]
            cls = selected_bbox["label"]
            width, height = abs(selected_bbox["points"][1][0] - x1), abs(selected_bbox["points"][1][1] - y1)
            image_content = self.image.copy(QRect(int(x1), int(y1), int(width), int(height)))
            sample_library_dir = "sample_library"
            if not os.path.exists(sample_library_dir):
                os.makedirs(sample_library_dir)

            # 创建文件名（例如，使用当前时间戳来避免覆盖）
            import time
            timestamp = str(int(time.time()))
            file_path = os.path.join(sample_library_dir, f"{cls}_{timestamp}.jpg")

            # 将QImage保存为JPG文件
            image_content.save(file_path, "JPG")
            self.display_slice_images(self.slice_folder)

    def menu_action2(self, row, column):
        print(f"菜单操作2 - {row}, {column}")

    def highlight_selected_bbox(self, row, column):
        """ 在图像上高亮选中的矩形框 """
        if row < len(self.annotations):
            self.selected_bbox = self.annotations[row]  # 存储选中标注
            self.show_image()  # 重新绘制图像以更新高亮
            self.selected_bbox_is_slice = False

    def load_annotations(self, json_path, folder_path):
        """加载JSON格式的标注信息"""
        self.annotation_list_widget.clear()
        self.annotations.clear()
        self.annotation_list_widget.setRowCount(0)
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                self.anno_data = json.load(file)
                for shape in self.anno_data["shapes"]:
                    if shape["shape_type"] == "rectangle":  # 仅处理矩形标注
                        label = shape["label"]
                        x1, y1 = shape["points"][0]
                        x2, y2 = shape["points"][1]
                        w, h = abs(x2 - x1), abs(y2 - y1)

                        # 添加到表格
                        row_position = self.annotation_list_widget.rowCount()
                        self.annotation_list_widget.insertRow(row_position)
                        self.annotation_list_widget.setItem(row_position, 0, QTableWidgetItem(label))
                        self.annotation_list_widget.setItem(row_position, 1, QTableWidgetItem(f"{x1:.1f}"))
                        self.annotation_list_widget.setItem(row_position, 2, QTableWidgetItem(f"{y1:.1f}"))
                        self.annotation_list_widget.setItem(row_position, 3, QTableWidgetItem(f"{w:.1f}"))
                        self.annotation_list_widget.setItem(row_position, 4, QTableWidgetItem(f"{h:.1f}"))

                        # 存储到 self.annotations
                        self.annotations.append({"label": label, "points": [(x1, y1), (x2, y2)]})
                self.annotation_list_widget.setHorizontalHeaderLabels(["cls", "x", "y", "w", "h"])

    def get_bbox_at_position(self, mouse_click_x, mouse_click_y):
        """Check if the mouse click is inside any bounding box in self.anno_data['shapes'].
           If so, return the corresponding bounding box.
        """
        off1, off2 = self.get_image_position_in_label()
        label_x, label_y = self.label.mapTo(self, QPoint(0, 0)).x(), self.label.mapTo(self, QPoint(0, 0)).y()
        new_x1 = (self.mouse_click_position.x() - off1 - label_x) / self.new_width * self.image.width()
        new_y1 = (self.mouse_click_position.y() - off2 - label_y) / self.new_height * self.image.height()
        for annotation in self.annotations:

            x1, y1 = annotation["points"][0]
            x2, y2 = annotation["points"][1]

            # 规范化坐标，确保 (x1, y1) 是左上角，(x2, y2) 是右下角
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # 判断鼠标点击点是否在该标注框内
            if x_min <= new_x1 <= x_max and y_min <= new_y1 <= y_max:
                return annotation  # 返回找到的标注框

        return None  # 如果没有找到合适的框，返回 None

    def mousePressEvent(self, event):
        """Handle right-click to move the selected bounding box"""
        if self.image:
            if event.button() == Qt.LeftButton:
                if self.selected_bbox is not None:
                    # Record the mouse position where the right-click happens
                    self.mouse_click_position = event.pos()
                    # Get the bounding box of the selected annotation
                    selected_bbox = self.selected_bbox
                    if selected_bbox:
                        self.is_moving_bbox = True  # Indicate that we're moving the bounding box
                        # self.update_image_with_moved_bbox()
                else:
                    self.mouse_click_position = event.pos()
                    self.selected_bbox = self.get_bbox_at_position(self.mouse_click_position.x(),
                                                                   self.mouse_click_position.y())
                    self.selected_bbox_is_slice = False
                    if self.selected_bbox:
                        self.is_moving_bbox = True
                        # self.update_image_with_moved_bbox()
                        # self.show_image()

            if event.button() == Qt.RightButton:  # 右键点击：取消选择框
                self.selected_bbox = None  # 清除选中的框
                self.show_image()  # 更新显示

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop moving the bounding box"""
        if self.is_moving_bbox:
            # Finalize the position of the bounding box after release
            self.is_moving_bbox = False
            new_x1, new_y1, new_x2, new_y2, label = self.update_image_with_moved_bbox()
            self.annotations.append({"label": label, "points": [(new_x1, new_y1), (new_x2, new_y2)]})
            new_shape = {
                "label": label,
                "points": [[new_x1, new_y1], [new_x2, new_y2]],
                "shape_type": "rectangle"
            }
            if "shapes" in self.anno_data:
                self.anno_data["shapes"].append(new_shape)
            else:
                self.anno_data["shapes"] = [new_shape]
            self.image = self.update_image

    def mouseMoveEvent(self, event):
        """Handle mouse move while moving the bounding box"""
        if self.is_moving_bbox and self.mouse_click_position:
            # Update the mouse position and redraw the image with the new bounding box position
            self.mouse_click_position = event.pos()
            self.update_image_with_moved_bbox()

    def qpixmap_to_opencv(self, qpixmap):
        img_qt = qpixmap.toImage()
        w, h = img_qt.width(), img_qt.height()
        bytes_ = img_qt.bits().asstring(w * h * 4)
        img_arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, 4))
        img_arr = img_arr[:, :, :3]
        return img_arr

    def opencv_to_qpixmap(self, img):
        """将 OpenCV 图像转换为 QPixmap"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def feather_blend(self, src_np, dst_np, new_x1, new_y1, width, height):
        """使用梯度羽化方式，仅对 src_np 和 dst_np 的交界区域模糊，使边缘平滑过渡"""

        # 计算羽化边缘的宽度（可调整）
        border_size = max(5, min(width, height) // 20)  # 取较小边长的 5% 作为羽化区域

        # 创建渐变掩码（0 在边缘，1 在中间）
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.rectangle(mask, (border_size, border_size), (width - border_size, height - border_size), 1, -1)
        mask = cv2.GaussianBlur(mask, (border_size * 2 + 1, border_size * 2 + 1), 0)

        # 复制目标区域
        dst_subsection = dst_np[new_y1:new_y1 + height, new_x1:new_x1 + width].astype(np.float32)

        # 加权混合
        blended_np = (src_np.astype(np.float32) * mask[:, :, None] + dst_subsection * (1 - mask[:, :, None])).astype(
            np.uint8)

        # 将混合后的图像覆盖到 dst_np
        dst_np[new_y1:new_y1 + height, new_x1:new_x1 + width] = blended_np

        return self.opencv_to_qpixmap(dst_np)

    def poisson_blend(self, src_np, dst_np, new_x1, new_y1, width, height):
        """使用泊松融合进行无缝克隆"""
        # 创建掩码（完全不透明）
        mask = np.full(src_np.shape, 255, dtype=np.uint8)

        # 计算中心点
        center = (int(new_x1 + width // 2), int(new_y1 + height // 2))

        # 执行泊松融合
        blended_np = cv2.seamlessClone(src_np, dst_np, mask, center, cv2.NORMAL_CLONE)

        return self.opencv_to_qpixmap(blended_np)

    def edge_process(self, src_image, dst_image, new_x1, new_y1, width, height):
        """根据选择的融合模式处理边缘"""

        # Convert QPixmap to OpenCV format (numpy array)
        src_np = self.qpixmap_to_opencv(src_image)  # 需要移动的区域
        dst_np = self.qpixmap_to_opencv(dst_image)  # 目标图像

        # 解决 dst_np 只读问题
        dst_np = dst_np.copy()

        # 目标图像尺寸
        h_dst, w_dst, _ = dst_np.shape

        # 获取融合方式
        if self.original_fusion.isChecked():  # 直接覆盖
            dst_np[new_y1:new_y1 + height, new_x1:new_x1 + width] = src_np[:height, :width]
            return self.opencv_to_qpixmap(dst_np)

        elif self.feathering_fusion.isChecked():  # 羽化融合
            return self.feather_blend(src_np, dst_np, new_x1, new_y1, width, height)

        elif self.poisson_fusion.isChecked():  # 泊松融合
            return self.poisson_blend(src_np, dst_np, new_x1, new_y1, width, height)

        return self.opencv_to_qpixmap(dst_np)  # 默认返回目标图像

    def update_image_with_moved_bbox(self):
        """Update image and redraw bounding box with new position"""
        if self.image and hasattr(self, 'selected_bbox') and self.mouse_click_position:
            # Get the bounding box and adjust its position based on mouse offset
            selected_bbox = self.selected_bbox
            if selected_bbox:
                x1, y1 = selected_bbox["points"][0]
                width, height = abs(selected_bbox["points"][1][0] - x1), abs(selected_bbox["points"][1][1] - y1)
                off1, off2 = self.get_image_position_in_label()
                label_x, label_y = self.label.mapTo(self, QPoint(0, 0)).x(), self.label.mapTo(self, QPoint(0, 0)).y()
                new_x1 = (self.mouse_click_position.x() - off1 - label_x) / self.new_width * self.image.width()
                new_y1 = (self.mouse_click_position.y() - off2 - label_y) / self.new_height * self.image.height()
                new_x2 = new_x1 + width
                new_y2 = new_y1 + height

                new_x1 = max(0, min(new_x1, self.image.width() - width))
                new_y1 = max(0, min(new_y1, self.image.height() - height))

                # 限制右下角在 (0,0) 以上，且不超出图像范围
                width = max(1, min(width, self.image.width() - new_x1))
                height = max(1, min(height, self.image.height() - new_y1))
                new_x1, new_y1, width, height = map(int, [new_x1, new_y1, width, height])
                if self.selected_bbox_is_slice:
                   image_content = self.slice_image
                # Extract the region to be moved (bounding box)
                else:
                    image_content = self.image.copy(QRect(int(x1), int(y1), int(width), int(height)))

                # Perform seamless cloning to update the image with the moved bounding box
                self.update_image = self.edge_process(image_content, self.image, new_x1, new_y1, width, height)

                self.show_image()

                # Return new bounding box coordinates
                return new_x1, new_y1, new_x2, new_y2, selected_bbox["label"]

    def show_image(self):
        """ Adjust image size and draw the bounding box on the image """
        if self.image is not None:
            if self.update_image is not None:
                show_img = self.update_image.copy()
            else:
                show_img = self.image.copy()

            available_size = self.scroll_area.viewport().size()

            if available_size.width() == 0 or available_size.height() == 0:
                return

            image_ratio = self.image.width() / self.image.height()
            available_ratio = available_size.width() / available_size.height()

            if image_ratio > available_ratio:
                self.new_width = available_size.width()
                self.new_height = self.new_width / image_ratio
            else:
                self.new_height = available_size.height()
                self.new_width = self.new_height * image_ratio

            resized_image = show_img.scaled(int(self.new_width), int(self.new_height), Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)

            annotated_image = resized_image.copy()
            painter = QPainter(annotated_image)

            x_scale = self.new_width / self.image.width()
            y_scale = self.new_height / self.image.height()

            for annotation in self.annotations:
                x1, y1 = annotation["points"][0]
                x2, y2 = annotation["points"][1]

                scaled_x1 = int(x1 * x_scale)
                scaled_y1 = int(y1 * y_scale)
                scaled_x2 = int(x2 * x_scale)
                scaled_y2 = int(y2 * y_scale)

                # Highlight the selected bounding box
                if hasattr(self, "selected_bbox") and self.selected_bbox == annotation:
                    pen = QPen(QColor(0, 255, 0))  # Selected box (green)
                    pen.setWidth(4)
                else:
                    pen = QPen(QColor(255, 0, 0))  # Default box (red)
                    pen.setWidth(2)

                painter.setPen(pen)
                painter.drawRect(scaled_x1, scaled_y1, scaled_x2 - scaled_x1, scaled_y2 - scaled_y1)

            painter.end()
            self.label.setPixmap(annotated_image)

    def resizeEvent(self, event):
        """ 窗口大小变化时更新图像大小 """
        self.show_image()
        super().resizeEvent(event)

    def show_previous_image(self):
        """ 显示上一张图像 """
        if self.image_files:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            self.load_image(os.path.join(os.path.dirname(self.image_path), self.image_files[self.current_image_index]),
                            os.path.dirname(self.image_path))

    def show_next_image(self):
        """ 显示下一张图像 """
        if self.image_files:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self.load_image(os.path.join(os.path.dirname(self.image_path), self.image_files[self.current_image_index]),
                            os.path.dirname(self.image_path))

    def save_image(self):
        """ 保存当前图像 """
        if self.update_image:
            current_image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
            self.update_image.save(current_image_path)
            annotation_path = os.path.join(self.folder_path,
                                           f"{os.path.splitext(self.image_files[self.current_image_index])[0]}.json")
            with open(annotation_path, "w") as json_file:
                json.dump(self.anno_data, json_file, indent=4)

    def switch_image_from_list(self):
        """ 从列表切换图像 """
        if self.image_list_widget.selectedItems():
            selected_item = self.image_list_widget.selectedItems()[0]
            selected_image_name = selected_item.text()
            self.load_image(os.path.join(os.path.dirname(self.image_path), selected_image_name),
                            os.path.dirname(self.image_path))


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = ImageWindow()
#     window.show()
#
#     sys.exit(app.exec_())
