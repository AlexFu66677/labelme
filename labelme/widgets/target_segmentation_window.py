from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtCore import Qt


class TargetSegmentationWindow(QtWidgets.QDialog):
    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super(TargetSegmentationWindow, self).__init__(parent)
        self.setWindowTitle(self.tr("目标分割"))
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.Tool, True)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(360, 390)

        self.prompt = QtWidgets.QLineEdit(self)
        self.prompt.setPlaceholderText(self.tr("输入文本提示，例如 person.car"))
        self.mode = QtWidgets.QComboBox(self)
        self.mode.addItem("SAM3(speed)", "speed")
        self.mode.addItem("SAM3(accuracy)", "accuracy")
        self.confidence = QtWidgets.QDoubleSpinBox(self)
        self.confidence.setRange(0.01, 1.0)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.25)
        self.nms = QtWidgets.QDoubleSpinBox(self)
        self.nms.setRange(0.01, 1.0)
        self.nms.setSingleStep(0.05)
        self.nms.setValue(0.70)
        self.image_size = QtWidgets.QSpinBox(self)
        self.image_size.setRange(32, 4096)
        self.image_size.setSingleStep(32)
        self.image_size.setValue(1008)
        self.process_all = QtWidgets.QCheckBox(self.tr("处理全部图像"), self)
        self.keep_existing = QtWidgets.QCheckBox(self.tr("保留已有标签"), self)
        self.run_button = QtWidgets.QPushButton(self.tr("开始分割"), self)
        self.stop_button = QtWidgets.QPushButton(self.tr("停止"), self)
        self.stop_button.setEnabled(False)

        intro = QtWidgets.QLabel(
            self.tr("使用 SAM3 文本提示分割当前图像，并保存为多边形标签。"),
            self,
        )
        intro.setWordWrap(True)

        group = QtWidgets.QGroupBox(self.tr("文本提示分割"), self)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setSpacing(6)
        group_layout.addWidget(self.mode)
        group_layout.addWidget(self.prompt)

        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(QtWidgets.QLabel(self.tr("置信度阈值"), self))
        conf_layout.addWidget(self.confidence)
        nms_layout = QtWidgets.QHBoxLayout()
        nms_layout.addWidget(QtWidgets.QLabel(self.tr("NMS 阈值"), self))
        nms_layout.addWidget(self.nms)
        image_size_layout = QtWidgets.QHBoxLayout()
        image_size_layout.addWidget(QtWidgets.QLabel(self.tr("图像尺寸"), self))
        image_size_layout.addWidget(self.image_size)
        group_layout.addLayout(conf_layout)
        group_layout.addLayout(nms_layout)
        group_layout.addLayout(image_size_layout)
        group_layout.addWidget(self.process_all)
        group_layout.addWidget(self.keep_existing)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        group_layout.addLayout(button_layout)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(intro)
        layout.addWidget(group)
        layout.addStretch()

    def options(self):
        return {
            "prompt": self.prompt.text().strip(),
            "strategy": self.mode.currentData(),
            "conf": self.confidence.value(),
            "iou": self.nms.value(),
            "imgsz": self.image_size.value(),
            "process_all": self.process_all.isChecked(),
            "keep_existing": self.keep_existing.isChecked(),
        }

    def set_running(self, running):
        """Match the detection panel's control state while inference runs."""
        for widget in (
            self.prompt,
            self.mode,
            self.confidence,
            self.nms,
            self.image_size,
            self.process_all,
            self.keep_existing,
        ):
            widget.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def closeEvent(self, event):
        self.closed.emit()
        super(TargetSegmentationWindow, self).closeEvent(event)
