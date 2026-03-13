import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QCheckBox,
    QScrollArea, QMessageBox, QSpinBox, QFormLayout, QGroupBox, QSplitter
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QLineEdit
import random
class ImagePreviewDialog(QWidget):
    def __init__(self, image_list, current_index):
        super().__init__()
        self.setWindowTitle("图像预览")
        self.resize(800, 600)

        self.image_list = image_list
        self.current_index = current_index

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        self.prev_button = QPushButton("⬅ 上一张")
        self.next_button = QPushButton("下一张 ➡")
        self.prev_button.clicked.connect(self.show_prev)
        self.next_button.clicked.connect(self.show_next)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addStretch()
        button_layout.addWidget(self.next_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.load_image()

    def load_image(self):
        image_path = self.image_list[self.current_index]
        self.setWindowTitle(f"图像预览 - {os.path.basename(image_path)}")
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def resizeEvent(self, event):
        self.load_image()

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def show_next(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.show_prev()
        elif event.key() == Qt.Key_Right:
            self.show_next()
        elif event.key() == Qt.Key_Escape:
            self.close()


class ImageWidget(QWidget):
    def __init__(self, image_path, max_width=150, max_height=150,
                 image_list=None, index_in_page=0, selected_files=None):
        super().__init__()
        self.image_path = image_path
        self.image_list = image_list
        self.index_in_page = index_in_page
        self.selected_files = selected_files

        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self.update_selected)
        img_name = os.path.basename(image_path)
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        label_text = img_name
        summary = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    cls_counts = {}
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            cls = int(parts[0])
                            cls_counts[cls] = cls_counts.get(cls, 0) + 1
                    if cls_counts:
                        summary = ' | ' + ', '.join(f"{k}: {v}" for k, v in sorted(cls_counts.items()))
            except Exception:
                summary = " | [解析失败]"

        self.label = QLabel(f"{img_name}{summary}")
        self.image_label = QLabel()

        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            max_width, max_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setCursor(Qt.PointingHandCursor)

        info_layout = QHBoxLayout()
        info_layout.addWidget(self.checkbox)
        info_layout.addWidget(self.label)
        info_layout.addStretch()  # 可选：右侧对齐

        # 整体垂直布局
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(info_layout)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)

    def update_selected(self, state):
        if self.selected_files is not None:
            if state == Qt.Checked:
                self.selected_files[self.image_path] = True
            elif self.image_path in self.selected_files:
                del self.selected_files[self.image_path]
    def is_checked(self):
        return self.checkbox.isChecked()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 左键切换复选框状态
            self.checkbox.setChecked(not self.checkbox.isChecked())
        elif event.button() == Qt.RightButton:
            # 右键打开大图预览
            if self.image_list:
                self.preview_window = ImagePreviewDialog(self.image_list, self.index_in_page)
                self.preview_window.show()



class ImageBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像分页浏览器（支持标注操作 + 原图预览）")
        self.resize(1200, 800)
        self.selected_files = {}  # 用于跨页记录选中图像
        self.image_paths = []
        self.image_widgets = []

        self.page = 0
        self.per_page = 20  # 默认显示 4x5
        self.current_folder = None  # 记录当前图像所在文件夹
        self.last_active_image = None  # 最后处理图像路径

        # 主界面容器
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # ============ 使用 QSplitter 实现左右可调宽度 =============
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：缩略图显示区
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.grid_widget)

        self.prev_button = QPushButton("上一页")
        self.next_button = QPushButton("下一页")
        self.page_label = QLabel("第 0 页 / 共 0 页")
        self.prev_button.clicked.connect(self.prev_page)
        self.next_button.clicked.connect(self.next_page)
        self.shortcut_prev = QShortcut(QKeySequence("A"), self)
        self.shortcut_prev.activated.connect(self.prev_page)

        self.shortcut_next = QShortcut(QKeySequence("D"), self)
        self.shortcut_next.activated.connect(self.next_page)

        self.goto_input = QLineEdit()
        self.goto_input.setFixedWidth(50)
        self.goto_input.setPlaceholderText("跳转页")
        self.goto_input.setAlignment(Qt.AlignCenter)
        self.goto_input.returnPressed.connect(self.goto_page)
        self.goto_button = QPushButton("跳转")
        self.goto_button.clicked.connect(self.goto_page)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.page_label)
        nav_layout.addWidget(self.goto_input)
        nav_layout.addWidget(self.goto_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_button)

        left_layout.addWidget(self.scroll_area)
        left_layout.addLayout(nav_layout)

        splitter.addWidget(left_container)

        # 右侧：操作区
        right_container = QWidget()
        control_layout = QVBoxLayout(right_container)

        self.load_button = QPushButton("加载图像文件夹")
        self.copy_button = QPushButton("复制选中")
        self.move_button = QPushButton("移动选中")
        self.delete_button = QPushButton("删除选中")

        self.load_button.clicked.connect(self.load_images)
        self.copy_button.clicked.connect(self.copy_selected)
        self.move_button.clicked.connect(self.move_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.selected_label = QLabel("已选中图像: 0")
        control_layout.addWidget(self.selected_label)
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.copy_button)
        control_layout.addWidget(self.move_button)
        control_layout.addWidget(self.delete_button)

        self.include_txt_checkbox = QCheckBox("同时操作同名标注（.txt）")
        self.include_txt_checkbox.setChecked(True)
        control_layout.addWidget(self.include_txt_checkbox)

        settings_group = QGroupBox("分页设置")
        form_layout = QFormLayout()

        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, 10)
        self.col_spin.setValue(4)

        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 10)
        self.row_spin.setValue(2)

        self.col_spin.valueChanged.connect(self.update_per_page)
        self.row_spin.valueChanged.connect(self.update_per_page)

        form_layout.addRow("列数:", self.col_spin)
        form_layout.addRow("行数:", self.row_spin)
        settings_group.setLayout(form_layout)

        control_layout.addWidget(settings_group)
        control_layout.addStretch()
        self.select_all_button = QPushButton("选中当前页全部")
        self.select_half_button = QPushButton("选中当前页偶数索引")
        self.select_odd_button = QPushButton("选中当前页奇数索引")
        self.select_random_button = QPushButton("随机选中当前页 3/4")

        self.select_all_button.clicked.connect(self.select_current_page_all)
        self.select_half_button.clicked.connect(self.select_current_page_half)
        self.select_odd_button.clicked.connect(self.select_current_page_odd)
        self.select_random_button.clicked.connect(self.select_current_page_random)

        control_layout.addWidget(self.select_all_button)
        control_layout.addWidget(self.select_half_button)
        control_layout.addWidget(self.select_odd_button)
        control_layout.addWidget(self.select_random_button)

        splitter.addWidget(right_container)

        # 设置右侧区域宽度范围（可根据需求调整）
        right_container.setMinimumWidth(200)
        right_container.setMaximumWidth(400)

        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)

    def select_current_page_all(self):
        for widget in self.image_widgets:
            widget.checkbox.setChecked(True)
        self.update_selected_count()

    def select_current_page_half(self):
        for i, widget in enumerate(self.image_widgets):
            widget.checkbox.setChecked(i % 2 == 0)
        self.update_selected_count()

    def select_current_page_odd(self):
        for i, widget in enumerate(self.image_widgets):
            widget.checkbox.setChecked(i % 2 == 1)
        self.update_selected_count()

    def select_current_page_random(self):
        total = len(self.image_widgets)
        count = int(total * 0.75)
        indices = random.sample(range(total), count)
        for i, widget in enumerate(self.image_widgets):
            widget.checkbox.setChecked(i in indices)
        self.update_selected_count()

    def record_last_unselected_image(self):
        self.last_active_image = None
        for widget in reversed(self.image_widgets):  # 从最后一张往前找
            if not widget.is_checked():
                self.last_active_image = widget.image_path
                break

    def custom_sort_key(self,file_names):
        import re
        name,_ = os.path.splitext(file_names)
        p,_num = name.rsplit('_',1)
        try:
            num= int(_num)
        except:
            num = float('inf')
        return (p.lower(),num)
    def load_images(self, folder=None):
        if not isinstance(folder, (str, os.PathLike)) or not os.path.isdir(folder):
            folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
            if not folder or not os.path.isdir(folder):
                return

        self.current_folder = folder  # 保存当前路径
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".png")

        self.image_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(valid_exts)
        ]
        self.image_paths= sorted(self.image_paths,key = self.custom_sort_key)

        self.page = 0
        self.selected_files.clear()
        self.update_per_page()

    def goto_page(self):
        text = self.goto_input.text()
        try:
            page = int(text) - 1
            total_pages = max(1, (len(self.image_paths) + self.per_page - 1) // self.per_page)
            if 0 <= page < total_pages:
                self.page = page
                self.update_page()
            else:
                QMessageBox.warning(self, "页码错误", f"请输入 1 到 {total_pages} 之间的数字")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字页码")

    def update_per_page(self):
        old_per_page = self.per_page
        old_page = self.page
        old_first_index = old_page * old_per_page

        self.per_page = self.col_spin.value() * self.row_spin.value()

        # 计算新页码，保证当前显示的图像连续
        if self.per_page > 0:
            self.page = old_first_index // self.per_page
        else:
            self.page = 0

        self.update_page()

    def update_page(self):
        for w in self.image_widgets:
            w.setParent(None)
        self.image_widgets.clear()

        total_pages = max(1, (len(self.image_paths) + self.per_page - 1) // self.per_page)
        start = self.page * self.per_page
        end = min(start + self.per_page, len(self.image_paths))
        current_page_paths = self.image_paths[start:end]

        scroll_width = self.scroll_area.viewport().width()
        scroll_height = self.scroll_area.viewport().height()
        col_count = self.col_spin.value()
        row_count = self.row_spin.value()
        cell_w = scroll_width // col_count - 10
        cell_h = scroll_height // row_count - 10

        for i, img_path in enumerate(current_page_paths):
            is_selected = self.selected_files.get(img_path, False)
            widget = ImageWidget(
                img_path, cell_w, cell_h,
                image_list=current_page_paths,
                index_in_page=i,
                selected_files=self.selected_files
            )
            widget.checkbox.setChecked(is_selected)
            self.image_widgets.append(widget)
            self.grid_layout.addWidget(widget, i // col_count, i % col_count)

        self.page_label.setText(f"第 {self.page + 1} 页 / 共 {total_pages} 页")
        self.prev_button.setEnabled(self.page > 0)
        self.next_button.setEnabled(self.page < total_pages - 1)
        self.update_selected_count()

    def update_selected_count(self):
        count = len(self.selected_files)
        self.selected_label.setText(f"已选中图像: {count}")
    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.update_page()

    def next_page(self):
        total_pages = max(1, (len(self.image_paths) + self.per_page - 1) // self.per_page)
        if self.page < total_pages - 1:
            self.page += 1
            self.update_page()

    def get_selected_paths(self):
        return list(self.selected_files.keys())

    def copy_selected(self):
        dst = QFileDialog.getExistingDirectory(self, "选择目标文件夹")
        if not dst:
            return
        for path in self.get_selected_paths():
            shutil.copy(path, dst)
            if self.include_txt_checkbox.isChecked():
                txt_path = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(txt_path):
                    shutil.copy(txt_path, dst)
        QMessageBox.information(self, "完成", "已复制选中图像及标注")
        self.selected_files.clear()

    def move_selected(self):
        dst = QFileDialog.getExistingDirectory(self, "选择目标文件夹")
        self.record_last_unselected_image()
        if not dst:
            return
        for path in self.get_selected_paths():
            shutil.move(path, dst)
            if self.include_txt_checkbox.isChecked():
                txt_path = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(txt_path):
                    shutil.move(txt_path, dst)
        QMessageBox.information(self, "完成", "已移动选中图像及标注")
        self.reload_current_folder()
        self.selected_files.clear()

    def delete_selected(self):
        reply = QMessageBox.question(self, "确认删除", "确定要删除选中的图像及标注？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.record_last_unselected_image()
            for path in self.get_selected_paths():
                os.remove(path)
                if self.include_txt_checkbox.isChecked():
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
            QMessageBox.information(self, "完成", "已删除选中图像及标注")
            self.reload_current_folder()
            self.selected_files.clear()
            self.update_page()

    def reload_current_folder(self):
        if self.current_folder:
            # 重新加载文件夹
            self.load_images(folder=self.current_folder)

            # 如果之前有焦点图像，则跳转到其新索引所在页
            if self.last_active_image in self.image_paths:
                new_index = self.image_paths.index(self.last_active_image)
                self.page = new_index // self.per_page
            else:
                self.page = min(self.page, max(0, len(self.image_paths) // self.per_page))

            self.update_page()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageBrowser()
    window.show()
    sys.exit(app.exec_())
