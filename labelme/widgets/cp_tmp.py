import sys
import os
import json
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QImage, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, \
    QFileDialog, QPushButton, QListWidget, QSplitter, QCheckBox, QButtonGroup, QSpacerItem, QSizePolicy, QMenu, QAction, \
    QListWidgetItem, QGridLayout, QComboBox,QLineEdit,QMessageBox,QSpinBox, QDockWidget,QHeaderView,QListView
from PyQt5.QtGui import QPen, QColor, QKeySequence, QCursor
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QShortcut
import cv2
import numpy as np
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import QRect


class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据增强")
        self.setGeometry(100, 100, 1400, 1000)

        self.image = None
        self.update_image = None
        self.zoom_scale = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.annotations = []
        self.selected_bbox = None
        self.selected_bbox_is_slice = False
        self.mouse_click_position = None
        self.is_moving_bbox = False
        self.anno_data = None
        self.slice_folder = None
        self.slice_image = None
        self.slice_image_list = []
        self.folder_path = None
        self.image_files = []
        self.current_image_index = -1
        self.annotation_format = None
        self.init_ui()

    def init_ui(self):
        self.create_menu_bar()

        # Center image view + fusion + annotation
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.scroll_area = QScrollArea()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(True)

        self.annotation_list_widget = QTableWidget()
        self.annotation_list_widget.setColumnCount(5)
        self.annotation_list_widget.setHorizontalHeaderLabels(["cls", "x1", "y1", "x2", "y2"])
        self.annotation_list_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.annotation_list_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.annotation_list_widget.cellPressed.connect(self.handle_cell_pressed)
        self.annotation_list_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fusion_widget = self.create_fusion_widget()

        layout.addWidget(self.scroll_area)

        self.setCentralWidget(central_widget)

        # Fusion + Annotation dock
        self.fusion_dock = QDockWidget("融合与标注", self)
        self.fusion_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.fusion_dock.setWidget(self.create_fusion_annotation_widget())
        self.addDockWidget(Qt.RightDockWidgetArea, self.fusion_dock)

        # Slice dock
        self.slice_dock = QDockWidget("切片图像", self)
        self.slice_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.slice_dock.setWidget(self.create_slice_widget())
        self.addDockWidget(Qt.LeftDockWidgetArea, self.slice_dock)

        # File list dock
        self.file_dock = QDockWidget("图像文件", self)
        self.file_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.file_dock.setWidget(self.create_file_widget())
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.fusion_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.fusion_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.fusion_dock)

    def action(self, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False, enabled=True):
        act = QAction(text, self)
        if icon is not None:
            act.setIcon(QIcon(icon))
        if shortcut is not None:
            act.setShortcut(QKeySequence(shortcut))
        if tip is not None:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        if slot is not None:
            act.triggered.connect(slot)
        act.setCheckable(checkable)
        act.setEnabled(enabled)
        return act

    def create_menu_bar(self):
        self.actions = {}

        menubar = self.menuBar()
        file_menu = menubar.addMenu("打开")
        window_menu = menubar.addMenu("窗口")

        # 加载文件夹
        self.actions["load_folder"] = self.action(
            "&加载文件夹", self.load_folder, None, None, "加载图像所在文件夹"
        )
        file_menu.addAction(self.actions["load_folder"])

        # 加载切片文件夹
        self.actions["load_slice_folder"] = self.action(
            "&加载切片文件夹", self.load_slice_folder, None, None, "加载图像切片所在文件夹"
        )
        file_menu.addAction(self.actions["load_slice_folder"])

        # 恢复默认窗口布局
        self.actions["reset_dock"] = self.action(
            "&恢复默认窗口", self.restore_default_docks, None, None, "恢复初始窗口布局"
        )
        window_menu.addAction(self.actions["reset_dock"])

        # 上一张图像
        self.actions["prev_img"] = self.action(
            "&上一张", self.show_previous_image, "A", None, "上一张图像"
        )
        menubar.addAction(self.actions["prev_img"])

        # 下一张图像
        self.actions["next_img"] = self.action(
            "&下一张", self.show_next_image, "D", None, "下一张图像"
        )
        menubar.addAction(self.actions["next_img"])

        # 保存图像与标注
        self.actions["save"] = self.action(
            "&保存", self.save_image, "S", None, "保存图像标注", enabled=True
        )
        menubar.addAction(self.actions["save"])

    def restore_default_docks(self):
        self.slice_dock.show()
        self.file_dock.show()
        self.fusion_dock.show()
        self.addDockWidget(Qt.LeftDockWidgetArea, self.slice_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

    def create_fusion_widget(self):
        layout = QHBoxLayout()
        self.original_fusion = QCheckBox("原始")
        self.feathering_fusion = QCheckBox("羽化")
        self.poisson_fusion = QCheckBox("泊松融合")

        self.fusion_group = QButtonGroup()
        self.fusion_group.addButton(self.original_fusion)
        self.fusion_group.addButton(self.feathering_fusion)
        self.fusion_group.addButton(self.poisson_fusion)
        self.fusion_group.setExclusive(True)
        self.original_fusion.setChecked(True)

        self.feather_radius_spin = QSpinBox()
        self.feather_radius_spin.setRange(1, 100)
        self.feather_radius_spin.setValue(3)
        self.feather_radius_spin.setSuffix(" px")
        self.feather_radius_spin.setEnabled(False)
        self.feathering_fusion.toggled.connect(lambda checked: self.feather_radius_spin.setEnabled(checked))

        layout.addWidget(self.original_fusion)
        layout.addWidget(self.feathering_fusion)
        layout.addWidget(self.feather_radius_spin)
        layout.addWidget(self.poisson_fusion)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def create_slice_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 0, 0, 0)
        layout.setSpacing(5)
        # self.load_slice_button = QPushButton("加载切片文件夹")
        # layout.addWidget(self.load_slice_button)
        self.slice_image_list_widget = QListWidget()
        self.slice_image_list_widget.setMovement(QListView.Static)
        self.slice_image_list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        layout.addWidget(self.slice_image_list_widget)
        layout.addLayout(size_control_layout)
        return widget

    def create_fusion_annotation_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.fusion_widget)
        layout.addWidget(self.annotation_list_widget)
        return widget

    def create_file_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.image_list_widget = QListWidget()
        self.image_list_widget.clicked.connect(self.switch_image_from_list)
        self.image_list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_list_widget)

        # btn_layout = QHBoxLayout()
        # self.prev_button = QPushButton("上一张")
        # prev_shortcut = QShortcut(QKeySequence("a"), self)
        # prev_shortcut.activated.connect(self.show_previous_image)
        # self.prev_button.clicked.connect(self.show_previous_image)
        # self.next_button = QPushButton("下一张")
        # self.next_button.clicked.connect(self.show_next_image)
        # next_shortcut = QShortcut(QKeySequence("d"), self)
        # next_shortcut.activated.connect(self.show_next_image)
        # self.save_button = QPushButton("保存")
        # self.save_button.clicked.connect(self.save_image)
        # save_shortcut = QShortcut(QKeySequence("s"), self)
        # save_shortcut.activated.connect(self.save_image)
        # # self.load_folder_button = QPushButton("加载", self)
        # # self.load_folder_button.clicked.connect(self.load_folder)
        # # btn_layout.addWidget(self.load_folder_button)
        # btn_layout.addWidget(self.prev_button)
        # btn_layout.addWidget(self.next_button)
        # btn_layout.addWidget(self.save_button)
        # layout.addLayout(btn_layout)

        # zoom_layout = QHBoxLayout()
        # zoom_in_button = QPushButton("放大")
        # zoom_out_button = QPushButton("缩小")
        # zoom_layout.addWidget(zoom_in_button)
        # zoom_layout.addWidget(zoom_out_button)
        # layout.addLayout(zoom_layout)

        return widget

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
        self.load_annotations(os.path.splitext(image_path)[0] + '.json', folder_path,image_path)
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

    def load_annotations(self, json_path, folder_path,image_path):
        """加载JSON格式的标注信息"""
        self.annotation_list_widget.clear()
        self.annotations.clear()
        self.annotation_list_widget.setRowCount(0)
        if os.path.exists(json_path):
            self.annotation_format = "json"
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
        elif os.path.exists(os.path.splitext(json_path)[0] + ".txt"):
            self.annotation_format = "txt"
            self.anno_data = {
                "shapes": []
            }
            txt_path = os.path.splitext(json_path)[0] + ".txt"
            from PIL import Image
            image_path = None
            for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                candidate = os.path.join(folder_path, os.path.splitext(json_path)[0] + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            img = Image.open(image_path)
            img_w, img_h = img.size
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = parts[0]
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = (xc - bw / 2) * img_w
                    y1 = (yc - bh / 2) * img_h
                    x2 = (xc + bw / 2) * img_w
                    y2 = (yc + bh / 2) * img_h
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    self.anno_data["shapes"].append({
                        "label": cls_id,
                        "points": [[x1, y1], [x2, y2]],
                        "shape_type": "rectangle",
                        "flags": {}
                    })
                    row = self.annotation_list_widget.rowCount()
                    self.annotation_list_widget.insertRow(row)
                    self.annotation_list_widget.setItem(row, 0, QTableWidgetItem(cls_id))
                    self.annotation_list_widget.setItem(row, 1, QTableWidgetItem(f"{x1:.1f}"))
                    self.annotation_list_widget.setItem(row, 2, QTableWidgetItem(f"{y1:.1f}"))
                    self.annotation_list_widget.setItem(row, 3, QTableWidgetItem(f"{w:.1f}"))
                    self.annotation_list_widget.setItem(row, 4, QTableWidgetItem(f"{h:.1f}"))

                    self.annotations.append({"label": cls_id, "points": [(x1, y1), (x2, y2)]})
        else:
            self.annotation_format = "json"
            img = cv2.imread(image_path)
            image_height, image_width, _ = img.shape
            self.anno_data = {
                "version": "5.3.0",
                "flags": {},
                "shapes": [],
                "lineColor": [0, 255, 0, 128],
                "fillColor": [255, 0, 0, 128],
                "imagePath": os.path.basename(image_path),
                "imageData": None,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "imagePass": None
            }
            self.annotations.clear()

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
        border_size = max(3, min(width, height) // 20)  # 取较小边长的 5% 作为羽化区域

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
                base_width = available_size.width()
                base_height = base_width / image_ratio
            else:
                base_height = available_size.height()
                base_width = base_height * image_ratio

            self.new_width = base_width * self.zoom_scale
            self.new_height = base_height * self.zoom_scale

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

    def select_list_item(self, image_name):
        """在图像列表中选中给定文件名的项"""
        matching_items = self.image_list_widget.findItems(image_name, Qt.MatchExactly)
        if matching_items:
            self.image_list_widget.setCurrentItem(matching_items[0])
    def show_previous_image(self):
        """ 显示上一张图像 """
        self.annotations.clear()
        if self.image_files:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            selected_image_name = self.image_files[self.current_image_index]
            image_dir = os.path.dirname(self.image_path)
            self.load_image(os.path.join(image_dir, selected_image_name), image_dir)

            # 同步选中项
            self.select_list_item(selected_image_name)

    def show_next_image(self):
        """ 显示下一张图像 """
        self.annotations.clear()
        if self.image_files:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            selected_image_name = self.image_files[self.current_image_index]
            image_dir = os.path.dirname(self.image_path)
            self.load_image(os.path.join(image_dir, selected_image_name), image_dir)

            # 同步选中项
            self.select_list_item(selected_image_name)

    def switch_image_from_list(self):
        """ 从列表切换图像 """
        if self.image_list_widget.selectedItems():
            selected_item = self.image_list_widget.selectedItems()[0]
            selected_image_name = selected_item.text()
            image_dir = os.path.dirname(self.image_path)
            full_path = os.path.join(image_dir, selected_image_name)

            # 更新 current_image_index
            if selected_image_name in self.image_files:
                self.current_image_index = self.image_files.index(selected_image_name)

            self.load_image(full_path, image_dir)

    def zoom_in(self):
        """ 放大图像 """
        self.zoom_scale = min(self.zoom_scale * 1.25, self.max_zoom)
        self.show_image()

    def zoom_out(self):
        """ 缩小图像 """
        self.zoom_scale = max(self.zoom_scale * 0.8, self.min_zoom)
        self.show_image()

    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:  # Ctrl + 滚轮缩放
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()

    def save_image(self):
        """根据原始标注类型保存图像与标注（json 或 yolo txt）"""
        if self.update_image:
            # 保存图像
            current_image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
            self.update_image.save(current_image_path)

            # 获取基本文件名
            base_name = os.path.splitext(self.image_files[self.current_image_index])[0]

            # 获取图像尺寸（YOLO需要归一化）
            from PIL import Image
            img = Image.open(current_image_path)
            img_w, img_h = img.size

            # 保存标注（判断来源格式）
            if self.annotation_format == "json":
                annotation_path = os.path.join(self.folder_path, f"{base_name}.json")
                with open(annotation_path, "w") as json_file:
                    json.dump(self.anno_data, json_file, indent=4)

            elif self.annotation_format == "txt":
                yolo_txt_path = os.path.join(self.folder_path, f"{base_name}.txt")
                with open(yolo_txt_path, "w") as txt_file:
                    for shape in self.anno_data.get("shapes", []):
                        if shape["shape_type"] != "rectangle":
                            continue
                        label = shape["label"]
                        x1, y1 = shape["points"][0]
                        x2, y2 = shape["points"][1]

                        # 转为YOLO格式
                        cx = (x1 + x2) / 2 / img_w
                        cy = (y1 + y2) / 2 / img_h
                        w = abs(x2 - x1) / img_w
                        h = abs(y2 - y1) / img_h

                        class_id = int(label) if label.isdigit() else 0  # 可加入 label 映射
                        txt_file.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.show()

    sys.exit(app.exec_())
