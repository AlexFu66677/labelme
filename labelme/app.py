# -*- coding: utf-8 -*-

import functools
import copy
import html
import ast
import math
import os
import os.path as osp
import re
import sys
import webbrowser

import shutil
import cv2
import PIL.Image
import PIL.ImageEnhance
import onnxruntime as ort
import numpy as np
from qtpy.QtWidgets import QFileDialog
import json
from labelme.widgets import QCWidget
from labelme.widgets import Selectonnx
from labelme.widgets import Slice_dataset
from labelme.widgets import Concat_dataset
from labelme.widgets import Create_dataset_Dialog
from labelme.widgets import Yolo_Vis_Dialog
from labelme.widgets import Video_slice_Dialog
from labelme.widgets import Data_augmentation_Dialog
from labelme.widgets import copy_paste_augmentation_Dialog
import imgviz
import natsort
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from labelme import PY2
from labelme import __appname__
from labelme.ai import MODELS
from labelme.ai import Text2LabelMODELS
from labelme.ai import VisualPromptMODELS
from labelme.ai import GroundingDINO
from labelme.ai import Sam3Inference
from labelme.ai.tracking import track_boxes_csrt
from labelme.ai.tracking import track_boxes_template_matching
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.logger import logger
from labelme.shape import Shape
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import FileListWidget
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ObjectDetectionWindow
from labelme.widgets import TrackingPredictionWindow
from labelme.widgets import TargetSegmentationWindow
from labelme.widgets import SequenceWarpingWindow
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
from labelme.widgets import Dataset_analysis_Dialog
from labelme.ai import obb_nms, xywhr2xyxyxyxy, resize_with_padding, calculate_iou
from labelme.ai import SIFT_Homography, optical_flow_Homography
from labelme.ai import preprocess, detect_postprocess, obb_detect_postprocess, obb_detect_postprocess_adjust, \
    obb_detect_preprocess_adjust
import os
import json
import numpy as np

import labelme.utils
from labelme.yolo_io import load_yolo_shapes, save_yolo_shapes
from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()
SEQUENCE_WARPING_MAX_SIDE = 1280


class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
            self,
            config=None,
            filename=None,
            output=None,
            output_file=None,
            output_dir=None,
    ):
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config
        self.device = self._config["device"]
        self.sam3_inference = Sam3Inference(self.device)
        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False
        # Category selected by the user is inherited by subsequently opened images.
        self._category_override = None
        self._advanced_auto_apply = False
        self._advanced_code = None
        self._advanced_instances = {}
        self._advanced_dialog = None
        self.annotation_format = "json"
        self.annotation_path = None

        self._noSelectionSlot = False

        self._copied_shapes = None

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. " "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.annotationInfo = QtWidgets.QLabel()
        self.annotationInfo.setAlignment(QtCore.Qt.AlignCenter)  # 文本居中
        self.annotationInfo.setText("Current: 0 | Total: 0")  # 默认文本

        self.fileListWidget = FileListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        self.fileListWidget.deleteClicked.connect(self.deleteImageAndAnnotation)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.annotationInfo)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self._mouse_position = None
        self.canvas.mouseMoved.connect(self.updateStatusBar)

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.shapeMoved.connect(self.updateStatusBar)
        self.canvas.shapeMoved.connect(self.batchMoveSelectedShapes)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scrollArea)
        self.rotate_state = 0

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        rotate = action(
            self.tr("&旋转"),
            self.Rotate,
            shortcuts["rotate"],
            "rotate",
            self.tr("图像旋转"),
            enabled=False,
        )
        select_onnx = action(
            self.tr("&选择模型"),
            self.select_onnx,
            shortcuts["select_onnx"],
            "select_onnx",
            self.tr("选择模型"),
            enabled=False,
        )
        object_shortcut = shortcuts["object"]
        if isinstance(object_shortcut, (list, tuple)):
            object_shortcut = [
                key for key in object_shortcut if str(key).upper() != "W"
            ]
            object_shortcut = object_shortcut or None
        elif object_shortcut is not None and str(object_shortcut).upper() == "W":
            object_shortcut = None
        object = action(
            self.tr("&目标检测"),
            self.toggleObjectDetectionPanel,
            object_shortcut,
            "object",
            self.tr("目标检测"),
            enabled=False,
            checkable=True,
        )
        target_segmentation = action(
            self.tr("&目标分割"),
            self.toggleTargetSegmentationPanel,
            None,
            "segmentation",
            self.tr("使用 SAM3 文本提示生成分割标注"),
            enabled=False,
            checkable=True,
        )
        sequence_warping = action(
            self.tr("&标注映射"),
            self.toggleSequenceWarpingPanel,
            shortcuts["sequence_warping"],
            "warping",
            self.tr("标注映射"),
            enabled=False,
            checkable=True,
        )
        tracking_prediction = action(
            self.tr("&跟踪预测"),
            self.toggleTrackingPredictionPanel,
            None,
            "objects",
            self.tr("根据上一帧的检测框预测当前帧标注"),
            enabled=False,
            checkable=True,
        )
        batch_process = action(
            self.tr("&批量处理"),
            self.showBatchProcessDialog,
            None,
            "objects",
            self.tr("设置批量处理的帧范围"),
            enabled=False,
        )
        image_pass = action(
            self.tr("&复核通过"),
            self.Image_pass,
            shortcuts["image_pass"],
            "image_pass",
            self.tr("复核通过"),
            enabled=False,
        )
        image_unpass = action(
            self.tr("&复核不通过"),
            self.Image_unpass,
            shortcuts["image_unpass"],
            "image_unpass",
            self.tr("复核不通过"),
            enabled=False,
        )
        create_dataset = action(
            self.tr("&数据集生成"),
            self.create_dataset,
            shortcuts["create_dataset"],
            "create_dataset",
            self.tr("数据集生成"),
        )
        dataset_analysis = action(
            self.tr("&数据集分析"),
            self.dataset_analysis,
            shortcuts["dataset_analysis"],
            "dataset_analysis",
            self.tr("数据集分析"),
        )
        yolovis = action(
            self.tr("&yolo可视化"),
            self.Yolovis,
            shortcuts["yolo"],
            "yolovis",
            self.tr("yolo可视化"),
        )
        video_slice = action(
            self.tr("&视频切分"),
            self.Video_slice,
            shortcuts["video_slice"],
            "video_slice",
            self.tr("视频切分"),
        )
        data_augmentation = action(
            self.tr("&数据增强"),
            self.Data_augmentation,
            shortcuts["data_augmentation"],
            "data_augmentation",
            self.tr("数据增强"),
        )
        set_category = action(
            self.tr("&高级配置"),
            self.advanced_config,
            None,
            "labels",
            self.tr("使用 Python 代码修改当前图像和标注"),
            enabled=False,
        )
        apply_category = action(
            self.tr("应用固定类别"),
            self.apply_category,
            shortcuts.get("set_category") or "L",
            "labels",
            self.tr("将当前图像所有标注改为已设置的固定类别"),
            enabled=False,
        )
        copy_paste_augmentation = action(
            self.tr("&复制粘贴增强"),
            self.copy_paste_augmentation,
            shortcuts["copy_paste_augmentation"],
            "copy_paste_augmentation",
            self.tr("复制粘贴增强"),
        )
        slice_dataset = action(
            self.tr("&数据集分割"),
            self.Slice_Dataset,
            shortcuts["create_dataset"],
            "slice",
            self.tr("数据分割"),
        )
        concat_dataset = action(
            self.tr("&数据集合并"),
            self.Concat_Dataset,
            shortcuts["create_dataset"],
            "concat",
            self.tr("数据合并"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "opendir",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        self.delete_image_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(shortcuts.get("delete_image", "Shift+Delete")),
            self,
        )
        self.delete_image_shortcut.setContext(Qt.ApplicationShortcut)
        self.delete_image_shortcut.activated.connect(
            self.deleteCurrentImageAndAnnotation
        )

        confirmDelete = action(
            self.tr("删除时确认"),
            self.setDeleteConfirmationEnabled,
            tip=self.tr("删除图片和标注文件前显示确认框"),
            checkable=True,
            checked=True,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createRotateMode = action(
            self.tr("创建旋转框"),
            lambda: self.toggleDrawMode(False, createMode="rotate"),
            shortcuts["create_rotate"],
            "objects",
            self.tr("Start drawing rotate rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "objects",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText(),
                device=self.device
            )
            if self.canvas.createMode == "ai_polygon"
            else None
        )
        createAiMaskMode = action(
            self.tr("Create AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            None,
            "objects",
            self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText(),
                device=self.device
            )
            if self.canvas.createMode == "ai_mask"
            else None
        )
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "copy",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        self.QCResult = QCWidget()
        QCResult = QtWidgets.QWidgetAction(self)
        QCResultBoxLayout = QtWidgets.QVBoxLayout()
        QCResultBoxLayout.addWidget(self.QCResult)
        QCResultLabel = QtWidgets.QLabel("复核结果")
        QCResultLabel.setAlignment(Qt.AlignCenter)
        QCResultLabel.setFont(QtGui.QFont(None, 10))
        QCResultBoxLayout.addWidget(QCResultLabel)
        QCResult.setDefaultWidget(QtWidgets.QWidget())
        QCResult.defaultWidget().setLayout(QCResultBoxLayout)

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            rotate=rotate,
            sequence_warping=sequence_warping,
            tracking_prediction=tracking_prediction,
            batch_process=batch_process,
            select_onnx=select_onnx,
            object=object,
            target_segmentation=target_segmentation,
            image_pass=image_pass,
            image_unpass=image_unpass,
            QCResult=QCResult,
            create_dataset=create_dataset,
            dataset_analysis=dataset_analysis,
            slice_dataset=slice_dataset,
            data_augmentation=data_augmentation,
            copy_paste_augmentation=copy_paste_augmentation,
            concat_dataset=concat_dataset,
            yolovis=yolovis,
            video_slice=video_slice,
            set_category=set_category,
            apply_category=apply_category,
            close=close,
            deleteFile=deleteFile,
            confirmDelete=confirmDelete,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createRotateMode=createRotateMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            tool=(),
            left_tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                copy,
                paste,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
                None,
                set_category,
                apply_category,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createRectangleMode,
                createRotateMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createRotateMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                createAiMaskMode,
                editMode,
                brightnessContrast,
                batch_process,
                tracking_prediction,
                target_segmentation,
                set_category,
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            # help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                confirmDelete,
                None,
                quit,
            ),
        )
        # utils.addActions(self.menus.help, (help,))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )
        self.text2label_model = None

        self.check_all_infer = QtWidgets.QCheckBox("推理全部图像", self)
        self.check_save_existing_label = QtWidgets.QCheckBox("保留已有标签", self)

        self.check_Lucas_Kanade = QtWidgets.QCheckBox("L-K", self)
        self.check_SIFT = QtWidgets.QCheckBox("SIFT", self)
        self.check_SIFT.setChecked(True)
        self.homography_group = QtWidgets.QButtonGroup(self)
        self.homography_group.addButton(self.check_Lucas_Kanade)
        self.homography_group.addButton(self.check_SIFT)
        self.homography_group.setExclusive(True)  # 互斥模式
        self.check_center = QtWidgets.QCheckBox("中心映射", self)
        self.check_edge = QtWidgets.QCheckBox("边缘映射", self)
        self.check_edge.setChecked(True)
        self.warp_method_group = QtWidgets.QButtonGroup(self)
        self.warp_method_group.addButton(self.check_center)
        self.warp_method_group.addButton(self.check_edge)
        self.warp_method_group.setExclusive(True)  # 互斥模式
        self.homography_thres = 0.3
        self.homography_thres_text = QtWidgets.QLineEdit(self)
        self.homography_thres_text.setText(str(self.homography_thres))
        self.homography_thres_text.setPlaceholderText("IOU(0-1)")
        self.homography_thres_text.setFixedWidth(80)
        self.homography_thres_button = QtWidgets.QPushButton("设置iou", self)
        self.keep_current_label = QtWidgets.QCheckBox("保留当前", self)
        self.keep_current_label.setFixedWidth(80)
        self.undo_shapes_button = QtWidgets.QPushButton("映射撤消", self)
        homography_thres_layout = QtWidgets.QHBoxLayout()
        homography_thres_layout.addWidget(self.homography_thres_text)
        homography_thres_layout.addWidget(self.homography_thres_button)

        self.object_conf_thres = 0.25
        self.object_conf_thres_text = QtWidgets.QLineEdit(self)
        self.object_conf_thres_text.setText(str(self.object_conf_thres))
        self.object_conf_thres_text.setPlaceholderText("Conf(0-1)")
        self.homography_thres_button.clicked.connect(self.update_homography_threshold)
        self.object_conf_thres_text.editingFinished.connect(self.update_object_conf_threshold)
        self.undo_shapes_button.clicked.connect(self.undo_shapes)
        self.pre_shape = []
        self.object_detection_enabled = False
        self.sequence_warping_enabled = False
        self.sequence_warping_window = SequenceWarpingWindow(self)
        self.sequence_warping_window.closed.connect(
            self.onSequenceWarpingWindowClosed
        )
        self.sequence_warping_apply_button = QtWidgets.QPushButton(
            self.tr("开始映射"),
            self.sequence_warping_window,
        )
        self.sequence_warping_apply_button.clicked.connect(self.sequence_warping)
        sequence_warping_hint = QtWidgets.QLabel(
            self.tr("Keep this window open to adjust mapped targets in the main view."),
            self.sequence_warping_window,
        )
        sequence_warping_hint.setWordWrap(True)
        sequence_warping_layout = QtWidgets.QVBoxLayout()
        sequence_warping_layout.addWidget(self.check_Lucas_Kanade)
        sequence_warping_layout.addWidget(self.check_SIFT)
        sequence_warping_layout.addSpacing(4)
        sequence_warping_layout.addWidget(self.check_center)
        sequence_warping_layout.addWidget(self.check_edge)
        sequence_warping_layout.addSpacing(4)
        sequence_warping_layout.addLayout(homography_thres_layout)
        sequence_warping_layout.addWidget(self.keep_current_label)
        sequence_warping_layout.addWidget(self.undo_shapes_button)
        sequence_warping_layout.addWidget(self.sequence_warping_apply_button)
        sequence_warping_layout.addWidget(sequence_warping_hint)
        self.sequence_warping_window.setLayout(sequence_warping_layout)

        # Tracking implementations can be added later with
        # registerTrackingAlgorithm(name, runner). A runner receives
        # (previous_image, current_image, previous_boxes) and returns current
        # boxes as [label, [x1, y1], [x2, y2]].
        self.tracking_algorithms = {}
        self.tracking_prediction_enabled = False
        self.tracking_prediction_window = TrackingPredictionWindow(self)
        self.tracking_prediction_window.closed.connect(
            self.onTrackingPredictionWindowClosed
        )
        self.tracking_algorithm_combo = QtWidgets.QComboBox(
            self.tracking_prediction_window
        )
        self.tracking_algorithm_combo.addItem(self.tr("暂无可用算法"), None)
        self.tracking_keep_existing = QtWidgets.QCheckBox(
            self.tr("保留已有标签"), self.tracking_prediction_window
        )
        self.template_match_threshold = QtWidgets.QDoubleSpinBox(
            self.tracking_prediction_window
        )
        self.template_match_threshold.setRange(0.0, 1.0)
        self.template_match_threshold.setSingleStep(0.05)
        self.template_match_threshold.setDecimals(2)
        self.template_match_threshold.setValue(0.8)
        self.template_nms_threshold = QtWidgets.QDoubleSpinBox(
            self.tracking_prediction_window
        )
        self.template_nms_threshold.setRange(0.0, 1.0)
        self.template_nms_threshold.setSingleStep(0.05)
        self.template_nms_threshold.setDecimals(2)
        self.template_nms_threshold.setValue(0.3)
        self.tracking_start_button = QtWidgets.QPushButton(
            self.tr("开始"), self.tracking_prediction_window
        )
        self.tracking_start_button.clicked.connect(self.runTrackingPrediction)
        tracking_layout = QtWidgets.QVBoxLayout()
        tracking_layout.setContentsMargins(12, 12, 12, 12)
        tracking_layout.setSpacing(8)
        tracking_layout.addWidget(QtWidgets.QLabel(
            self.tr("跟踪算法"), self.tracking_prediction_window
        ))
        tracking_layout.addWidget(self.tracking_algorithm_combo)
        template_threshold_layout = QtWidgets.QHBoxLayout()
        self.template_match_threshold_label = QtWidgets.QLabel(
            self.tr("匹配阈值"), self.tracking_prediction_window
        )
        template_threshold_layout.addWidget(self.template_match_threshold_label)
        template_threshold_layout.addWidget(self.template_match_threshold)
        tracking_layout.addLayout(template_threshold_layout)
        template_nms_layout = QtWidgets.QHBoxLayout()
        self.template_nms_threshold_label = QtWidgets.QLabel(
            self.tr("NMS 阈值"), self.tracking_prediction_window
        )
        template_nms_layout.addWidget(self.template_nms_threshold_label)
        template_nms_layout.addWidget(self.template_nms_threshold)
        tracking_layout.addLayout(template_nms_layout)
        tracking_layout.addWidget(self.tracking_keep_existing)
        tracking_layout.addWidget(self.tracking_start_button)
        tracking_layout.addStretch()
        self.tracking_prediction_window.setLayout(tracking_layout)
        self.registerTrackingAlgorithm(self.tr("CSRT"), track_boxes_csrt)
        self.registerTrackingAlgorithm(
            self.tr("模板匹配"), self._runTemplateMatching
        )
        self.tracking_algorithm_combo.currentIndexChanged.connect(
            self._updateTrackingParameterState
        )
        self._updateTrackingParameterState()

        # 批量模式仅记录帧范围；实际操作仍在主画布上完成。
        self.batch_processing_enabled = False
        self.batch_start_index = 0
        self.batch_end_index = 0
        self.batch_process_dialog = QtWidgets.QDialog(self)
        self.batch_process_dialog.setWindowTitle(self.tr("批量处理"))
        self.batch_process_dialog.setModal(False)
        self.batch_start_spin = QtWidgets.QSpinBox(self.batch_process_dialog)
        self.batch_end_spin = QtWidgets.QSpinBox(self.batch_process_dialog)
        self.batch_start_spin.setPrefix(self.tr("第 "))
        self.batch_start_spin.setSuffix(self.tr(" 张"))
        self.batch_end_spin.setPrefix(self.tr("第 "))
        self.batch_end_spin.setSuffix(self.tr(" 张"))
        self.batch_process_start_button = QtWidgets.QPushButton(
            self.tr("开始批量处理"), self.batch_process_dialog
        )
        self.batch_process_start_button.clicked.connect(self.startBatchProcessing)
        self.batch_process_stop_button = QtWidgets.QPushButton(
            self.tr("停止批量处理"), self.batch_process_dialog
        )
        self.batch_process_stop_button.clicked.connect(self.stopBatchProcessing)
        self.batch_process_stop_button.setEnabled(False)
        self.batch_process_status = QtWidgets.QLabel(self.batch_process_dialog)
        batch_layout = QtWidgets.QVBoxLayout(self.batch_process_dialog)
        batch_layout.addWidget(QtWidgets.QLabel(self.tr("选择处理范围（包含起止帧）：")))
        batch_layout.addWidget(self.batch_start_spin)
        batch_layout.addWidget(self.batch_end_spin)
        batch_layout.addWidget(self.batch_process_start_button)
        batch_layout.addWidget(self.batch_process_stop_button)
        batch_layout.addWidget(self.batch_process_status)

        self.object_detection_window = ObjectDetectionWindow(self)
        self.object_detection_window.closed.connect(
            self.onObjectDetectionWindowClosed
        )
        self.target_segmentation_window = TargetSegmentationWindow(self)
        self.target_segmentation_window.closed.connect(
            self.onTargetSegmentationWindowClosed
        )
        self.target_segmentation_window.run_button.clicked.connect(
            self.runTargetSegmentation
        )
        self.target_segmentation_window.stop_button.clicked.connect(
            self.stopTargetSegmentation
        )
        self.target_segmentation_inference = Sam3Inference(self.device)
        self.target_segmentation_stop_requested = False
        self.object_detection_select_model_button = QtWidgets.QPushButton(
            self.tr("选择 ONNX 模型"), self.object_detection_window
        )
        self.object_detection_select_model_button.clicked.connect(self.select_onnx)
        self.object_detection_select_model_button.setProperty("variant", "primary")
        self.object_detection_onnx_path = QtWidgets.QLineEdit(self.object_detection_window)
        self.object_detection_onnx_path.setReadOnly(True)
        self.object_detection_onnx_path.setPlaceholderText(self.tr("未选择目标检测模型"))
        self.object_detection_run_button = QtWidgets.QPushButton(
            self.tr("开始"), self.object_detection_window
        )
        self.object_detection_run_button.clicked.connect(self.object_detection)
        self.object_detection_infer_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(shortcuts.get("object_detection_infer", "W")),
            self,
        )
        self.object_detection_infer_shortcut.setContext(Qt.ApplicationShortcut)
        self.object_detection_infer_shortcut.setEnabled(False)
        self.object_detection_infer_shortcut.activated.connect(
            self._runObjectDetectionShortcut
        )
        self.object_detection_stop_button = QtWidgets.QPushButton(
            self.tr("停止"), self.object_detection_window
        )
        self.object_detection_stop_button.clicked.connect(
            self.requestStopAiAnnotation
        )
        self.object_detection_stop_button.setEnabled(False)
        self.text2label_stop_button = QtWidgets.QPushButton(
            self.tr("停止"), self.object_detection_window
        )
        self.text2label_stop_button.clicked.connect(
            self.requestStopAiAnnotation
        )
        self.text2label_stop_button.setEnabled(False)
        self.check_all_infer_dino = QtWidgets.QCheckBox("处理全部图像", self.object_detection_window)
        self.check_save_existing_label_dino = QtWidgets.QCheckBox("保留已有标签", self.object_detection_window)
        self.text2label_conf_thres = 0.3
        self.text2label_conf_thres_text = QtWidgets.QLineEdit(self.object_detection_window)
        self.text2label_conf_thres_text.setText(str(self.text2label_conf_thres))
        self.text2label_conf_thres_text.setPlaceholderText("Conf(0-1)")
        self.text2label_conf_thres_text.editingFinished.connect(self.update_text2label_conf_threshold)
        self.text2label_nms_thres = 0.7
        self.text2label_nms_thres_text = QtWidgets.QLineEdit(self.object_detection_window)
        self.text2label_nms_thres_text.setText(str(self.text2label_nms_thres))
        self.text2label_nms_thres_text.setPlaceholderText("NMS(0-1)")
        self.text2label_nms_thres_text.editingFinished.connect(self.update_text2label_nms_threshold)
        self.text2label_imgsz_w = 1200
        self.text2label_imgsz_h = 800
        self.text2label_imgsz_w_text = QtWidgets.QLineEdit(self.object_detection_window)
        self.text2label_imgsz_w_text.setText(str(self.text2label_imgsz_w))
        self.text2label_imgsz_w_text.setPlaceholderText("宽")
        self.text2label_imgsz_w_text.editingFinished.connect(self.update_text2label_imgsz)
        self.text2label_imgsz_h_text = QtWidgets.QLineEdit(self.object_detection_window)
        self.text2label_imgsz_h_text.setText(str(self.text2label_imgsz_h))
        self.text2label_imgsz_h_text.setPlaceholderText("高")
        self.text2label_imgsz_h_text.editingFinished.connect(self.update_text2label_imgsz)
        self._selectAiVisualPromptModelComboBox = QtWidgets.QComboBox(self.object_detection_window)
        self.sam3_label_text = QtWidgets.QLineEdit(self.object_detection_window)
        self.sam3_conf_thres = 0.25
        self.sam3_conf_thres_text = QtWidgets.QLineEdit(self.object_detection_window)
        self.sam3_conf_thres_text.setText(str(self.sam3_conf_thres))
        self.sam3_conf_thres_text.setPlaceholderText("Conf(0-1)")
        self.sam3_conf_thres_text.editingFinished.connect(self.update_sam3_conf_threshold)
        self.sam3_imgsz = 640
        self.sam3_imgsz_text = QtWidgets.QLineEdit(self.object_detection_window)
        self.sam3_imgsz_text.setText(str(self.sam3_imgsz))
        self.sam3_imgsz_text.setPlaceholderText("图像尺寸")
        self.sam3_imgsz_text.editingFinished.connect(self.update_sam3_imgsz)
        self.sam3_single_target_check = QtWidgets.QCheckBox("仅标注单一目标", self.object_detection_window)
        self.check_save_existing_label_sam3 = QtWidgets.QCheckBox("保留已有标签", self.object_detection_window)
        self.sam3_run_button = QtWidgets.QPushButton("开始", self.object_detection_window)
        self.sam3_run_button.clicked.connect(self.start_sam3_annotation)
        self.sam3_pause_button = QtWidgets.QPushButton("停止", self.object_detection_window)
        self.sam3_pause_button.clicked.connect(self.pause_sam3_annotation)
        self.sam3_pause_button.setEnabled(False)
        self.ai_annotation_stop_requested = False
        self.ai_annotation_is_running = False
        self.text2label_model_type = None
        self.text2label_sam_strategy = None
        self.text2label_model_path = None
        self.sam3_placeholder_config = {}
        self.sam3_annotation_active = False
        self.ai_annotation_controls = {
            "object_detection": {
                "run_all": self.check_all_infer,
                "keep_existing": self.check_save_existing_label,
            },
            "grounding_dino": {
                "run_all": self.check_all_infer_dino,
                "keep_existing": self.check_save_existing_label_dino,
            },
            "sam3": {
                "keep_existing": self.check_save_existing_label_sam3,
            },
        }

        selectAiWidget = QtWidgets.QWidget()
        selectAiLayout = QtWidgets.QVBoxLayout(selectAiWidget)
        self._selectAiText2LabelModelComboBox = QtWidgets.QComboBox()
        selectAiModelLabel = QtWidgets.QLabel(self.tr("SAM:"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        self._selectAiModelComboBox = QtWidgets.QComboBox()
        aiModelLayout = QtWidgets.QHBoxLayout()
        aiModelLayout.addWidget(selectAiModelLabel)
        aiModelLayout.addWidget(self._selectAiModelComboBox)
        selectAiLayout.addLayout(aiModelLayout)
        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(selectAiWidget)

        text2label_model_names = [model.name for model in Text2LabelMODELS]
        self._selectAiText2LabelModelComboBox.addItems(text2label_model_names)
        # 禁止自动加载模型 提升软件整体初始化速度
        text_model_index = 0
        text_default = self._config["ai"]["text_default"]
        if text_default == "SAM3":
            text_default = "SAM3(accuracy)"
        if text_default in text2label_model_names:
            text_model_index = text2label_model_names.index(text_default)
        self._selectAiText2LabelModelComboBox.setCurrentIndex(text_model_index)
        self._selectAiText2LabelModelComboBox.currentIndexChanged.connect(
            lambda: self.init_text2lable_model(
                Text2LabelMODELS[
                    text2label_model_names.index(self._selectAiText2LabelModelComboBox.currentText())].config_path,
                Text2LabelMODELS[
                    text2label_model_names.index(self._selectAiText2LabelModelComboBox.currentText())].model_path,
                self.device
            )
        )
        visual_prompt_model_names = [model.name for model in VisualPromptMODELS]
        self._selectAiVisualPromptModelComboBox.addItems(visual_prompt_model_names)
        visual_prompt_model_index = 0
        self._selectAiVisualPromptModelComboBox.setCurrentIndex(visual_prompt_model_index)
        self._selectAiVisualPromptModelComboBox.currentIndexChanged.connect(
            lambda: self.init_sam3_model(
                VisualPromptMODELS[
                    visual_prompt_model_names.index(self._selectAiVisualPromptModelComboBox.currentText())
                ].config_path,
                VisualPromptMODELS[
                    visual_prompt_model_names.index(self._selectAiVisualPromptModelComboBox.currentText())
                ].model_path,
                self.device,
            )
        )
        self.Text2Label_Text = QtWidgets.QLineEdit(self)
        self.Text2Label_detect_button = QtWidgets.QPushButton("开始", self)
        self.Text2Label_Text.setAlignment(QtCore.Qt.AlignCenter)
        self.Text2Label_detect_button.clicked.connect(self.Run_Text2Label)
        Text2Label_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(shortcuts["Text2Label"]), self)
        Text2Label_shortcut.activated.connect(self.Text2Label_detect_button.click)
        self.sam3_toggle_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(shortcuts["visual_prompt_toggle"]), self
        )
        self.sam3_toggle_shortcut.activated.connect(self.toggle_sam3_annotation_shortcut)
        model_names = [model.name for model in MODELS]
        self._selectAiModelComboBox.addItems(model_names)
        if self._config["ai"]["default"] in model_names:
            model_index = model_names.index(self._config["ai"]["default"])
        else:
            logger.warning(
                "Default AI model is not found: %r",
                self._config["ai"]["default"],
            )
            model_index = 0
        self._selectAiModelComboBox.setCurrentIndex(model_index)
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText(),
                device=self.device
            )
            if self.canvas.createMode in ["ai_polygon", "ai_mask"]
            else None
        )

        detection_panel_layout = QtWidgets.QVBoxLayout()
        detection_panel_layout.setContentsMargins(12, 12, 12, 12)
        detection_panel_layout.setSpacing(8)

        detection_intro = QtWidgets.QLabel(
            self.tr("在当前图像或整个序列上执行检测，并保存标签。"),
            self.object_detection_window,
        )
        detection_intro.setWordWrap(True)
        detection_panel_layout.addWidget(detection_intro)

        model_group = QtWidgets.QGroupBox(self.tr("目标检测"), self.object_detection_window)
        model_layout = QtWidgets.QVBoxLayout()
        model_layout.setSpacing(6)
        self.object_conf_thres_text.setFixedWidth(80)
        object_conf_layout = QtWidgets.QHBoxLayout()
        object_conf_layout.addWidget(QtWidgets.QLabel("置信度阈值", self.object_detection_window))
        object_conf_layout.addWidget(self.object_conf_thres_text)
        model_layout.addWidget(self.object_detection_select_model_button)
        model_layout.addWidget(self.object_detection_onnx_path)
        model_layout.addLayout(object_conf_layout)
        model_layout.addWidget(self.check_all_infer)
        model_layout.addWidget(self.check_save_existing_label)
        object_detection_button_layout = QtWidgets.QHBoxLayout()
        object_detection_button_layout.addWidget(self.object_detection_run_button)
        object_detection_button_layout.addWidget(self.object_detection_stop_button)
        model_layout.addLayout(object_detection_button_layout)
        model_group.setLayout(model_layout)
        detection_panel_layout.addWidget(model_group)

        dino_group = QtWidgets.QGroupBox(self.tr("文本提示"), self.object_detection_window)
        dino_layout = QtWidgets.QVBoxLayout()
        dino_layout.setSpacing(6)
        self.Text2Label_Text.setPlaceholderText(self.tr("输入文本提示，例如 car.person"))
        self.text2label_conf_thres_text.setFixedWidth(80)
        self.text2label_nms_thres_text.setFixedWidth(80)
        self.text2label_imgsz_w_text.setFixedWidth(80)
        self.text2label_imgsz_h_text.setFixedWidth(80)
        dino_conf_layout = QtWidgets.QHBoxLayout()
        dino_conf_layout.addWidget(QtWidgets.QLabel("置信度阈值", self.object_detection_window))
        dino_conf_layout.addWidget(self.text2label_conf_thres_text)
        dino_nms_layout = QtWidgets.QHBoxLayout()
        dino_nms_layout.addWidget(QtWidgets.QLabel("NMS 阈值", self.object_detection_window))
        dino_nms_layout.addWidget(self.text2label_nms_thres_text)
        dino_imgsz_layout = QtWidgets.QHBoxLayout()
        dino_imgsz_layout.addWidget(QtWidgets.QLabel("图像尺寸", self.object_detection_window))
        dino_imgsz_layout.addWidget(QtWidgets.QLabel("长", self.object_detection_window))
        dino_imgsz_layout.addWidget(self.text2label_imgsz_w_text)
        dino_imgsz_layout.addWidget(QtWidgets.QLabel("宽", self.object_detection_window))
        dino_imgsz_layout.addWidget(self.text2label_imgsz_h_text)
        dino_layout.addWidget(self._selectAiText2LabelModelComboBox)
        dino_layout.addWidget(self.Text2Label_Text)
        dino_layout.addLayout(dino_conf_layout)
        dino_layout.addLayout(dino_nms_layout)
        dino_layout.addLayout(dino_imgsz_layout)
        dino_layout.addWidget(self.check_all_infer_dino)
        dino_layout.addWidget(self.check_save_existing_label_dino)
        dino_button_layout = QtWidgets.QHBoxLayout()
        dino_button_layout.addWidget(self.Text2Label_detect_button)
        dino_button_layout.addWidget(self.text2label_stop_button)
        dino_layout.addLayout(dino_button_layout)
        dino_group.setLayout(dino_layout)
        detection_panel_layout.addWidget(dino_group)

        sam3_group = QtWidgets.QGroupBox(self.tr("视觉提示"), self.object_detection_window)
        sam3_layout = QtWidgets.QVBoxLayout()
        sam3_layout.setSpacing(6)
        self.sam3_label_text.setPlaceholderText(self.tr("设置标签名，检测结果统一使用该标签"))
        self.sam3_conf_thres_text.setFixedWidth(80)
        self.sam3_imgsz_text.setFixedWidth(80)
        sam3_conf_layout = QtWidgets.QHBoxLayout()
        sam3_conf_layout.addWidget(QtWidgets.QLabel("置信度阈值", self.object_detection_window))
        sam3_conf_layout.addWidget(self.sam3_conf_thres_text)
        sam3_imgsz_layout = QtWidgets.QHBoxLayout()
        sam3_imgsz_layout.addWidget(QtWidgets.QLabel("图像尺寸", self.object_detection_window))
        sam3_imgsz_layout.addWidget(self.sam3_imgsz_text)
        sam3_layout.addWidget(self._selectAiVisualPromptModelComboBox)
        sam3_layout.addWidget(self.sam3_label_text)
        sam3_layout.addLayout(sam3_conf_layout)
        sam3_layout.addLayout(sam3_imgsz_layout)
        sam3_layout.addWidget(self.sam3_single_target_check)
        sam3_layout.addWidget(self.check_save_existing_label_sam3)
        sam3_button_layout = QtWidgets.QHBoxLayout()
        sam3_button_layout.addWidget(self.sam3_run_button)
        sam3_button_layout.addWidget(self.sam3_pause_button)
        sam3_layout.addLayout(sam3_button_layout)
        sam3_group.setLayout(sam3_layout)
        detection_panel_layout.addWidget(sam3_group)
        detection_panel_layout.addStretch()
        self.object_detection_window.setLayout(detection_panel_layout)
        self.init_sam3_model(
            VisualPromptMODELS[visual_prompt_model_index].config_path,
            VisualPromptMODELS[visual_prompt_model_index].model_path,
            self.device,
        )
        self._refreshDetectionPanelState()

        self.tools = self.toolbar("Tools")
        self.lefttools = self.lefttoolbar("LeftTools")

        self.actions.lefttool = (
            object,
            target_segmentation,
            sequence_warping,
            tracking_prediction,
            batch_process,
            # rotate,
            # video_object,
            fitWindow,
            # zoomIn,
            # zoomOut,
            create_dataset,
            dataset_analysis,
            slice_dataset,
            concat_dataset,
            yolovis,
            video_slice,
            data_augmentation,
            set_category,
            # copy_paste_augmentation,
            None,
            # image_pass,
            # image_unpass,
            # QCResult,
        )

        self.actions.tool = (
            open_,
            opendir,
            openPrevImg,
            openNextImg,
            save,
            deleteFile,
            None,
            createRectangleMode,
            editMode,
            duplicate,
            delete,
            undo,
            None,
            zoom,
            None,
            selectAiModel,
            None,
        )

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warn(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        confirm_delete = self.settings.value("confirmDelete", True)
        if isinstance(confirm_delete, str):
            confirm_delete = confirm_delete.lower() not in ("false", "0", "no")
        self.actions.confirmDelete.setChecked(bool(confirm_delete))
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        self.center()
        self.onnx_path = ''
        self.net = None
        self.label_list = {}

        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    def lefttoolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sLeftToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        # action1 = QAction("Action 1", self)
        # action2 = QAction("Action 2", self)
        # # 将动作添加到工具栏
        # toolbar.addAction(action1)
        # toolbar.addAction(action2)

        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def center(self):
        screen = QtWidgets.QDesktopWidget().availableGeometry()
        window_size = self.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        self.move(x, y)

    def populateModeActions(self):
        lefttool, tool, menu = self.actions.lefttool, self.actions.tool, self.actions.menu
        self.lefttools.clear()
        utils.addActions(self.lefttools, lefttool)
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createRotateMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.createAiPolygonMode,
            self.actions.createAiMaskMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = self._currentAnnotationFile()
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.rotate_state = 0
        self.actions.rotate.setEnabled(False)
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createRotateMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)
        self.actions.createAiMaskMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)
        self._applyAnnotationModeRestrictions()

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        # self.closeSequenceWarpingPanel()
        # self.closeObjectDetectionPanel()
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.annotation_format = "json"
        self.annotation_path = None
        self.otherData = None
        self.tmpimageData = None
        self.rotatevalue = None
        self.imagePass = None
        self.QCResult.setText(None)
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "rotate": self.actions.createRotateMode,
            "circle": self.actions.createCircleMode,
            "point": self.actions.createPointMode,
            "line": self.actions.createLineMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
        }
        self.actions.rotate.setEnabled(False)
        if edit or createMode != "rectangle":
            self.sam3_annotation_active = False
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
            self.actions.rotate.setEnabled(False)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)
        self.actions.rotate.setEnabled(False)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(True)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(False)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        self.canvas.storeShapes()
        for item in items:
            self._update_item(
                item=item,
                text=text if edit_text else None,
                flags=flags if edit_flags else None,
                group_id=group_id if edit_group_id else None,
                description=description if edit_description else None,
            )

    def _update_item(self, item, text, flags, group_id, description):
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        shape = item.shape()

        if text is not None:
            if text != shape.label and self.annotation_format == "txt":
                shape.other_data.pop("yolo_class_id", None)
            shape.label = text
        if flags is not None:
            shape.flags = flags
        if group_id is not None:
            shape.group_id = group_id
        if description is not None:
            shape.description = description

        self._update_shape_color(shape)
        if shape.group_id is None:
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                )
            )
        else:
            item.setText("{} ({})".format(shape.label, shape.group_id))
        self.setDirty()
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            if item is None:
                continue
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)
        self.updateStatusBar()

    def updateStatusBar(self, mouse_position=None):
        """Show the selected shape's bounds, or the current mouse position."""
        if mouse_position is not None:
            self._mouse_position = mouse_position

        if len(self.canvas.selectedShapes) == 1:
            rect = self.canvas.selectedShapes[0].boundingRect()
            center = rect.center()
            self.status(
                "Selected shape: center=({:g}, {:g}), width={:g}, height={:g}".format(
                    center.x(), center.y(), rect.width(), rect.height()
                )
            )
        elif self._mouse_position is not None:
            self.status(
                "Mouse is at: x={:g}, y={:g}".format(
                    self._mouse_position.x(), self._mouse_position.y()
                )
            )

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (
                self._config["shape_color"] == "manual"
                and self._config["label_colors"]
                and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):
        try:
            for shape in shapes:
                item = self.labelList.findItemByShape(shape)
                if item is not None:
                    self.labelList.removeItem(item)
        except:
            return

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            if flags == None:
                flags = {}
            description = shape.get("description", "")
            group_id = shape["group_id"]
            other_data = shape["other_data"]
            direction = shape.get("direction", 0)
            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                direction=direction,
                mask=shape["mask"],
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def saveLabels(self, filename, imagePass=None):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            if data["shape_type"] == "rotate":
                data["direction"] = s.direction
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        if self.annotation_format == "txt" or str(filename).lower().endswith(".txt"):
            try:
                save_yolo_shapes(
                    filename, shapes, self.image.width(), self.image.height(),
                    self._config.get("labels") or [],
                )
                self.annotation_format = "txt"
                self.annotation_path = filename
                items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
                if len(items) == 1:
                    items[0].setCheckState(Qt.Checked)
                return True
            except (OSError, ValueError) as e:
                self.errorMessage(self.tr("Error saving YOLO label data"), str(e))
                return False
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
                imagePass=imagePass,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def save_AI_obb_labels(self, filename, res, imagePass=None):
        lf = LabelFile()
        self.setDirty()
        self.loadFile(self.filename)

        def format_shape(s):
            if self.label_list and (type(s[0]) != type('1')):
                data = dict(
                    label=self.label_list[s[0]],
                    points=[[p[0], p[1]] for p in s[1:5]],
                    shape_type="rotate",
                    direction=s[5],
                    flags={},
                    description="",
                    group_id=None,
                    mask=None,
                )
            elif type(s[0]) == type('1'):
                data = dict(
                    label=s[0],
                    points=[[p[0], p[1]] for p in s[1:5]],
                    shape_type="rotate",
                    direction=s[5],
                    flags={},
                    description="",
                    group_id=None,
                    mask=None,
                )
            return data

        shapes = [format_shape(item) for item in res]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            if self.check_save_existing_label.isChecked():
                if self.labelFile:
                    if self.labelFile.shapes:
                        # 如果不为空，将新 shapes 追加到现有的 shapes 中
                        existing_shapes = self.labelFile.shapes
                        for existing_shape in existing_shapes:
                            if existing_shape['mask'] is not None:
                                existing_shape['mask'] = utils.img_arr_to_b64(existing_shape['mask'].astype(np.uint8))
                        shapes = existing_shapes + shapes
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
                imagePass=imagePass,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(
                self.imagePath, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("保存标签发生错误"), self.tr("<b>%s</b>") % e
            )
            return False

    def save_AI_Labels(self, filename, res, imagePass=None, keep_existing=None):
        lf = LabelFile()
        self.setDirty()
        self.loadFile(self.filename)

        def format_shape(s):
            if self.label_list and (type(s[0]) != type('1')):
                data = dict(
                    label=self.label_list[s[0]],
                    points=[[p[0], p[1]] for p in s[1:5]],
                    shape_type="rectangle",
                    flags={},
                    description="",
                    group_id=None,
                    mask=None,

                )
            elif type(s[0]) == type('1'):
                data = dict(
                    label=s[0],
                    points=[[p[0], p[1]] for p in s[1:5]],
                    shape_type="rectangle",
                    flags={},
                    description="",
                    group_id=None,
                    mask=None,
                )
            return data

        shapes = [format_shape(item) for item in res]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            if keep_existing is None:
                keep_existing = self.check_save_existing_label.isChecked()
            if keep_existing:
                if self.labelFile:
                    if self.labelFile.shapes:
                        # 如果不为空，将新 shapes 追加到现有的 shapes 中
                        existing_shapes = self.labelFile.shapes
                        for existing_shape in existing_shapes:
                            if existing_shape['mask'] is not None:
                                existing_shape['mask'] = utils.img_arr_to_b64(existing_shape['mask'].astype(np.uint8))
                        shapes = existing_shapes + shapes
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
                imagePass=imagePass,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(
                self.imagePath, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("保存标签发生错误"), self.tr("<b>%s</b>") % e
            )
            return False

    def save_AI_PolygonLabels(self, filename, polygons, keep_existing=False):
        """Save AI segmentation results as Labelme polygon shapes."""
        shapes = [
            {
                "label": polygon["label"],
                "points": polygon["points"],
                "shape_type": "polygon",
                "flags": {},
                "description": "",
                "group_id": None,
                "mask": None,
            }
            for polygon in polygons
        ]
        if keep_existing and self.labelFile and self.labelFile.shapes:
            existing_shapes = []
            for source_shape in copy.deepcopy(self.labelFile.shapes):
                other_data = source_shape.pop("other_data", {})
                shape = source_shape
                if shape.get("mask") is not None and not isinstance(shape["mask"], str):
                    shape["mask"] = utils.img_arr_to_b64(
                        shape["mask"].astype(np.uint8)
                    )
                shape.update(other_data)
                existing_shapes.append(shape)
            shapes = existing_shapes + shapes

        flags = {}
        for index in range(self.flag_widget.count()):
            item = self.flag_widget.item(index)
            flags[item.text()] = item.checkState() == Qt.Checked

        image_path = osp.relpath(self.imagePath, osp.dirname(filename))
        image_data = self.imageData if self._config["store_data"] else None
        if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
            os.makedirs(osp.dirname(filename))
        label_file = LabelFile()
        label_file.save(
            filename=filename,
            shapes=shapes,
            imagePath=image_path,
            imageData=image_data,
            imageHeight=self.image.height(),
            imageWidth=self.image.width(),
            otherData=self.otherData,
            flags=flags,
            imagePass=None,
        )
        self.labelFile = label_file
        # The annotation file has already been written. Marking the window dirty
        # here makes loadFile() auto-save the old canvas over the new polygons.
        self.setClean()
        items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
        if len(items) == 1:
            items[0].setCheckState(Qt.Checked)
        return True

    def duplicateSelectedShape(self):
        added_shapes = self.canvas.duplicateSelectedShapes()
        self.labelList.clearSelection()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()
        if self.batch_processing_enabled and self._copied_shapes:
            self._batchPasteShapes(self._copied_shapes)

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if (
                self.sam3_annotation_active
                and self.canvas.createMode == "rectangle"
                and self.canvas.shapes
                and self.canvas.shapes[-1].shape_type == "rectangle"
        ):
            prompt_shape = self.canvas.shapes[-1]
            prompt_box = self._get_prompt_box_from_shape(prompt_shape)
            try:
                self._run_sam3_box_detection(prompt_box)
            except Exception as e:
                self.loadFile(self.filename)
                self.errorMessage(
                    self.tr("SAM3 推理失败"),
                    self.tr("{}").format(e),
                )
            return

        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
                self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)

        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
        label_file = self._annotationFileForImage(filename)
        self.annotation_path = label_file if osp.isfile(label_file) else None
        self.annotation_format = "txt" if label_file.lower().endswith(".txt") else "json"
        if (self.annotation_format == "json" and QtCore.QFile.exists(label_file)
                and LabelFile.is_label_file(label_file)):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.tmpimageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self.otherData = self.labelFile.otherData
            try:
                data = self.labelFile.imagePass
            except:
                data = None
            self.QCResult.setText(str(data))
        else:
            self.imageData = LabelFile.load_image_file(filename)
            self.tmpimageData = self.imageData
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        yolo_shapes = []
        if self.annotation_format == "txt" and osp.isfile(label_file):
            try:
                yolo_shapes = load_yolo_shapes(
                    label_file, image.width(), image.height(),
                    self._config.get("labels") or [],
                )
            except OSError as e:
                self.errorMessage(self.tr("Error opening YOLO label file"), str(e))
                return False
        total_files = len(self.imageList)  # 总标注数量
        if len(self.imageList) != 0:
            current_index = self.imageList.index(filename) + 1  # 当前标注索引 (从1开始)
            self.annotationInfo.setText(f"Current: {current_index} | Total: {total_files} ")
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        elif self.annotation_format == "txt":
            self.loadLabels(yolo_shapes)
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        if self._advanced_auto_apply and self._advanced_code and len(self.labelList):
            try:
                self._execute_advanced_code(self._advanced_code)
            except Exception as exc:
                logger.error("自动应用高级配置失败: %s", exc)
        self.canvas.setEnabled(True)
        self.actions.rotate.setEnabled(False)
        self.actions.select_onnx.setEnabled(True)
        self.actions.object.setEnabled(True)
        self.actions.target_segmentation.setEnabled(True)
        self.actions.sequence_warping.setEnabled(True)
        self.actions.tracking_prediction.setEnabled(True)
        self.actions.batch_process.setEnabled(True)
        self._applyAnnotationModeRestrictions()
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self._refreshDetectionPanelState()
        self._applyAnnotationModeRestrictions()
        self.canvas.setFocus()
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        return True

    def Rotate(self, _value=False):

        img = utils.img_data_to_pil(self.tmpimageData)
        img = img.transpose(PIL.Image.ROTATE_90)
        self.rotate_state += 90
        if (self.rotate_state == 360):
            self.rotate_state = 0
        img_data = utils.img_pil_to_data(img)
        qimage = QtGui.QImage.fromData(img_data)

        self.remLabels(self.canvas.deleteallShape())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

        self.canvas.setEnabled(True)
        if (self.rotate_state == 0):
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)
            self.canvas.setEnabled(True)
            self.actions.save.setEnabled(True)
            self.actions.saveAs.setEnabled(True)
            self.actions.delete.setEnabled(True)
            self.actions.deleteFile.setEnabled(True)
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(True)
            self.actions.undo.setEnabled(True)
            self.actions.select_onnx.setEnabled(True)
            self.actions.object.setEnabled(True)
            self.actions.target_segmentation.setEnabled(True)
            self.actions.sequence_warping.setEnabled(True)
            self.actions.tracking_prediction.setEnabled(True)
        else:
            self.canvas.setEnabled(False)
            self.actions.save.setEnabled(False)
            self.actions.saveAs.setEnabled(False)
            self.actions.delete.setEnabled(False)
            self.actions.deleteFile.setEnabled(False)
            self.actions.openNextImg.setEnabled(False)
            self.actions.openPrevImg.setEnabled(False)
            self.actions.createMode.setEnabled(False)
            self.actions.editMode.setEnabled(False)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(False)
            self.actions.select_onnx.setEnabled(False)
            self.actions.object.setEnabled(False)
            self.actions.target_segmentation.setEnabled(False)
            self.actions.sequence_warping.setEnabled(False)
            self.actions.tracking_prediction.setEnabled(False)
            self.closeObjectDetectionPanel()
            self.target_segmentation_window.close()
            self.closeSequenceWarpingPanel()
            self.closeTrackingPredictionPanel()
        if self.labelFile:
            shapes = self.labelFile.shapes
            s = []
            for shape in shapes:
                label = shape["label"]
                points = shape["points"]
                shape_type = shape["shape_type"]
                description = shape.get("description", "")
                group_id = shape["group_id"]
                flags = shape["flags"]
                other_data = shape["other_data"]

                if not points:
                    continue
                shape = Shape(
                    label=label,
                    shape_type=shape_type,
                    group_id=group_id,
                    description=description,
                )
                for x, y in points:
                    if (self.rotate_state == 0):
                        shape.addPoint(QtCore.QPointF(x, y))
                    if (self.rotate_state == 90):
                        shape.addPoint(QtCore.QPointF(y, self.image.width() - x))
                    if (self.rotate_state == 180):
                        shape.addPoint(QtCore.QPointF(self.image.width() - x, self.image.height() - y))
                    if (self.rotate_state == 270):
                        shape.addPoint(QtCore.QPointF(self.image.height() - y, x))
                default_flags = {}
                if self._config["label_flags"]:
                    for pattern, keys in self._config["label_flags"].items():
                        if re.match(pattern, label):
                            for key in keys:
                                default_flags[key] = False
                shape.flags = default_flags
                shape.flags.update(flags)
                shape.other_data = other_data
                s.append(shape)
            self.loadShapes(s)
        self.canvas.loadPixmap(
            QtGui.QPixmap.fromImage(qimage), clear_shapes=False
        )
        self.tmpimageData = img_data

    def select_onnx(self, _value=False):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select ONNX File', '', 'ONNX Files (*.onnx);;All Files (*)',
                                                   options=options)
        if not file_name:
            return False
        self.onnx_path = file_name
        so = ort.SessionOptions()
        so.log_severity_level = 4
        if self.device == "cpu":
            self.net = ort.InferenceSession(self.onnx_path, so, providers=['CPUExecutionProvider'])
        elif self.device == "cuda":
            self.net = ort.InferenceSession(self.onnx_path, so, providers=['CUDAExecutionProvider'])
        else:
            self.errorMessage(
                self.tr("Invalid providers"),
                self.tr("Invalid infer providers '{}'").format(
                    self._config["device"]
                ),
            )
            return False
        self.actions.object.setEnabled(True)
        self._refreshDetectionPanelState()
        self.actions.rotate.setEnabled(False)

    def init_sam3_model(self, config_path, model_path, device, _value=False):
        self.sam3_model_file = model_path
        self.sam3_inference.set_visual_model(model_path)
        self._refreshDetectionPanelState()
        return True

    def init_text2lable_model(self, config_path, model_path, device, _value=False):
        if not model_path:
            self.text2label_model = None
            self.text2label_model_type = None
            self.text2label_sam_strategy = None
            self.text2label_model_path = None
            self.sam3_inference.set_text_model(None)
            self._refreshDetectionPanelState()
            return False
        self.text2label_model_path = model_path
        if model_path.lower().endswith(".onnx"):
            self.sam3_inference.set_text_model(None)
            self.text2label_model = GroundingDINO(
                model_path,
                self.text2label_conf_thres,
                "./ai/seg_model/vocab.txt",
                device,
                0.1,
            )
            self.text2label_model.size = [self.text2label_imgsz_w, self.text2label_imgsz_h]
            self.text2label_model_type = "grounding_dino"
            self.text2label_sam_strategy = None
        else:
            self.text2label_model = None
            self.text2label_model_type = "sam3_text"
            self.text2label_sam_strategy = (
                config_path if config_path in ("speed", "accuracy") else "accuracy"
            )
            self.sam3_inference.set_text_model(
                model_path, self.text2label_sam_strategy
            )
        self.actions.rotate.setEnabled(False)
        self._refreshDetectionPanelState()
        return self.text2label_model if self.text2label_model is not None else True

    def _text2label_model_ready(self):
        if self.text2label_model_type == "grounding_dino":
            return self.text2label_model is not None
        if self.text2label_model_type == "sam3_text":
            return bool(self.text2label_model_path)
        return False

    def _refreshDetectionPanelState(self):
        if hasattr(self, "object_detection_onnx_path"):
            self.object_detection_onnx_path.setText(
                getattr(self, "onnx_path", "") or ""
            )
        if hasattr(self, "object_detection_run_button"):
            self.object_detection_run_button.setEnabled(bool(getattr(self, "net", None)))
        if hasattr(self, "object_detection_infer_shortcut"):
            self.object_detection_infer_shortcut.setEnabled(
                self.object_detection_window.isVisible()
                and bool(getattr(self, "net", None))
                and not self.ai_annotation_is_running
            )
        if hasattr(self, "sam3_run_button"):
            self.sam3_run_button.setEnabled(bool(getattr(self, "sam3_model_file", None)))
        if hasattr(self, "Text2Label_detect_button"):
            self.Text2Label_detect_button.setEnabled(self._text2label_model_ready())
        self._updateSam3PanelState()

    def _updateSam3PanelState(self, _value=None):
        if not hasattr(self, "sam3_run_button"):
            return
        self.sam3_run_button.setEnabled(
            bool(getattr(self, "sam3_model_file", None)) and not self.ai_annotation_is_running)
        if hasattr(self, "sam3_pause_button"):
            self.sam3_pause_button.setEnabled(self.sam3_annotation_active and not self.ai_annotation_is_running)

    def _parse_sam_text_prompts(self, raw_text):
        return self.sam3_inference.parse_text_prompts(raw_text)

    def _get_ai_annotation_options(self, algorithm_name):
        controls = self.ai_annotation_controls[algorithm_name]
        return {
            "run_all_images": controls["run_all"].isChecked() if "run_all" in controls else False,
            "keep_existing_labels": controls["keep_existing"].isChecked(),
        }

    def _set_ai_annotation_running(self, running):
        self.ai_annotation_is_running = running
        self.object_detection_stop_button.setEnabled(running)
        self.text2label_stop_button.setEnabled(running)
        self.object_detection_run_button.setEnabled(
            bool(getattr(self, "net", None)) and not running
        )
        self.object_detection_infer_shortcut.setEnabled(
            self.object_detection_window.isVisible()
            and bool(getattr(self, "net", None))
            and not running
        )
        self.Text2Label_detect_button.setEnabled(self._text2label_model_ready() and not running)
        self.sam3_run_button.setEnabled(
            bool(getattr(self, "sam3_model_file", None)) and not running
        )

    def requestStopAiAnnotation(self):
        if self.ai_annotation_is_running:
            self.ai_annotation_stop_requested = True

    def _runObjectDetectionShortcut(self):
        if (
                self.object_detection_window.isVisible()
                and getattr(self, "net", None) is not None
                and self.object_detection_run_button.isEnabled()
        ):
            self.object_detection_run_button.click()

    def _run_ai_annotation(self, algorithm_name, single_runner, reload_current=False):
        options = self._get_ai_annotation_options(algorithm_name)
        self.ai_annotation_stop_requested = False
        self._set_ai_annotation_running(True)
        try:
            if options["run_all_images"]:
                start_index = self.imageList.index(self.filename)
                for i in range(start_index, len(self.imageList)):
                    QtWidgets.QApplication.processEvents()
                    if self.ai_annotation_stop_requested:
                        break
                    single_runner(options)
                    QtWidgets.QApplication.processEvents()
                    if self.ai_annotation_stop_requested:
                        break
                    self.openNextImg()
            else:
                single_runner(options)
                if reload_current:
                    self.loadFile(self.filename)
        finally:
            self.ai_annotation_stop_requested = False
            self._set_ai_annotation_running(False)

    def toggleObjectDetectionPanel(self, checked=False):
        self.object_detection_enabled = checked
        if checked:
            self.object_detection_window.show()
            self.object_detection_window.raise_()
            self.object_detection_window.activateWindow()
        else:
            self.object_detection_window.hide()
        self._refreshDetectionPanelState()

    def toggleTargetSegmentationPanel(self, checked=False):
        if checked:
            self.target_segmentation_window.show()
            self.target_segmentation_window.raise_()
            self.target_segmentation_window.activateWindow()
        else:
            self.target_segmentation_window.hide()

    def onTargetSegmentationWindowClosed(self):
        self.target_segmentation_stop_requested = True
        self.actions.target_segmentation.setChecked(False)

    @staticmethod
    def _targetSegmentationModel(strategy):
        model_name = "SAM3({})".format(strategy)
        for model in Text2LabelMODELS:
            if model.name == model_name:
                return model.model_path
        raise RuntimeError("Target segmentation model is not configured: {}".format(model_name))

    def stopTargetSegmentation(self, _value=False):
        self.target_segmentation_stop_requested = True
        self.status(self.tr("正在停止目标分割……"))

    def _targetSegmentationShouldStop(self):
        QtWidgets.QApplication.processEvents()
        return self.target_segmentation_stop_requested

    def _runTargetSegmentationForCurrent(self, options, prompts):
        image = cv2.imdecode(
            np.fromfile(self.filename, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError(self.tr("无法读取当前原始图像。"))
        polygons = self.target_segmentation_inference.run_text_segmentation(
            image=image,
            prompts=prompts,
            conf=options["conf"],
            iou=options["iou"],
            imgsz=options["imgsz"],
            should_stop=self._targetSegmentationShouldStop,
        )
        if polygons and not self.target_segmentation_stop_requested:
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file = osp.join(self.output_dir, osp.basename(label_file))
            self.save_AI_PolygonLabels(
                label_file,
                polygons,
                keep_existing=options["keep_existing"],
            )
        return polygons

    def runTargetSegmentation(self, _value=False):
        options = self.target_segmentation_window.options()
        prompts = self.target_segmentation_inference.parse_text_prompts(
            options["prompt"]
        )
        if not self.filename:
            self.errorMessage(self.tr("未打开图像"), self.tr("请先打开一张图像。"))
            return
        if not prompts:
            self.errorMessage(
                self.tr("无效文本提示"),
                self.tr("请输入至少一个文本提示，例如 person.car。"),
            )
            return

        model_path = self._targetSegmentationModel(options["strategy"])
        if (
                self.target_segmentation_inference.text_model_path != model_path
                or self.target_segmentation_inference.text_strategy != options["strategy"]
        ):
            self.target_segmentation_inference.set_text_model(
                model_path, options["strategy"]
            )

        self.target_segmentation_stop_requested = False
        self.target_segmentation_window.set_running(True)
        QtWidgets.QApplication.processEvents()
        try:
            image_count = len(self.imageList) if options["process_all"] else 1
            start_index = self.imageList.index(self.filename)
            if options["process_all"]:
                image_count -= start_index
            total_polygons = 0
            processed_images = 0
            for index in range(image_count):
                if self._targetSegmentationShouldStop():
                    break
                polygons = self._runTargetSegmentationForCurrent(options, prompts)
                total_polygons += len(polygons)
                processed_images += 1
                self.status(
                    self.tr("目标分割：{}/{} 张，已生成 {} 个多边形。").format(
                        processed_images, image_count, total_polygons
                    )
                )
                if options["process_all"] and index < image_count - 1:
                    self.openNextImg()
                    QtWidgets.QApplication.processEvents()
            if self.target_segmentation_stop_requested:
                self.status(self.tr("目标分割已停止。"))
                return
            if total_polygons == 0:
                self.status(self.tr("目标分割未检测到符合条件的目标。"))
                self.errorMessage(
                    self.tr("没有分割结果"),
                    self.tr("SAM3 未返回有效 Mask，请尝试降低置信度或更换文本提示。"),
                )
                return
            self.loadFile(self.filename)
            self.status(
                self.tr("目标分割完成：处理 {} 张图像，生成 {} 个多边形。").format(
                    processed_images, total_polygons
                )
            )
        except Exception as error:
            self.errorMessage(self.tr("目标分割失败"), self.tr("{}").format(error))
        finally:
            self.target_segmentation_stop_requested = False
            self.target_segmentation_window.set_running(False)
            self.target_segmentation_window.prompt.clearFocus()
            self.activateWindow()
            self.canvas.setFocus(QtCore.Qt.OtherFocusReason)

    def onObjectDetectionWindowClosed(self):
        self.object_detection_enabled = False
        self.sam3_annotation_active = False
        if hasattr(self, "_selectAiText2LabelModelComboBox"):
            self._selectAiText2LabelModelComboBox.setCurrentIndex(0)
        self.text2label_model = None
        self.text2label_model_type = None
        self.text2label_sam_strategy = None
        self.text2label_model_path = None
        self.sam3_inference.reset()
        self.actions.object.setChecked(False)
        self._refreshDetectionPanelState()
        # closed is emitted before QDialog finishes hiding, so force the
        # visibility-dependent shortcut off here.
        self.object_detection_infer_shortcut.setEnabled(False)

    def closeObjectDetectionPanel(self):
        if self.object_detection_window.isVisible():
            self.object_detection_window.close()
        else:
            self.onObjectDetectionWindowClosed()

    def get_sam3_annotation_options(self):
        options = self._get_ai_annotation_options("sam3")
        options.update(
            {
                "model_path": getattr(self, "sam3_model_file", ""),
                "annotation_mode": "目标检测",
                "prompt_mode": "方框提示",
                "single_target_only": self.sam3_single_target_check.isChecked(),
                "label_text": self.sam3_label_text.text().strip(),
                "imgsz": self.sam3_imgsz,
            }
        )
        return options

    def _get_default_sam3_label(self, single_target_only):
        custom_label = self.sam3_label_text.text().strip()
        if custom_label:
            return custom_label
        items = self.uniqLabelList.selectedItems()
        if items:
            selected = items[0].data(Qt.UserRole)
            if selected:
                return selected
        return "sam3_object" if single_target_only else "sam3_concept"

    # def _show_sam_mask_overlay(self, result):
    #     if not hasattr(self, "canvas"):
    #         return
    #     if result is None or result.masks is None or len(result.masks) == 0:
    #         self.canvas.clearSamMaskOverlay()
    #         return
    #     mask_data = result.masks.data.detach().cpu().numpy()
    #     self.canvas.setSamMaskOverlay(mask_data)

    def _run_sam3_box_detection(self, prompt_box):
        options = self.get_sam3_annotation_options()
        default_label = self._get_default_sam3_label(options["single_target_only"])
        image = cv2.imdecode(np.fromfile(self.filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("无法读取当前图像")
        labels = self.sam3_inference.run_box_prompt(
            image=image,
            prompt_box=prompt_box,
            default_label=default_label,
            single_target_only=options["single_target_only"],
            conf=self.sam3_conf_thres,
            imgsz=options["imgsz"],
        )
        label_file = osp.splitext(self.imagePath)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))
        self.save_AI_Labels(label_file, labels, keep_existing=options["keep_existing_labels"])
        self.loadFile(self.filename)
        # self._show_sam_mask_overlay(result)

    def _get_prompt_box_from_shape(self, shape):
        xs = [point.x() for point in shape.points[:2]]
        ys = [point.y() for point in shape.points[:2]]
        x1, x2 = sorted(xs)
        y1, y2 = sorted(ys)
        return [float(x1), float(y1), float(x2), float(y2)]

    def start_sam3_annotation(self, _value=False):
        self.update_sam3_conf_threshold()
        options = self.get_sam3_annotation_options()
        self.sam3_placeholder_config = options
        if not options["model_path"]:
            self.errorMessage(self.tr("未选择模型"), self.tr("请先选择 SAM3 模型文件。"))
            return
        self.sam3_annotation_active = True
        self.toggleDrawMode(False, createMode="rectangle")
        self._updateSam3PanelState()
        self.status(self.tr("SAM3 方框提示已开启：请在图像上绘制一个矩形提示框。"))

    def toggle_sam3_annotation_shortcut(self):
        if self.sam3_annotation_active:
            self.pause_sam3_annotation()
        else:
            self.start_sam3_annotation()

    def pause_sam3_annotation(self, _value=False):
        self.sam3_annotation_active = False
        self.setEditMode()
        self._updateSam3PanelState()
        self.status(self.tr("SAM3 已暂停：现在可以手动调整标注或补标目标。"))

    def toggleSequenceWarpingPanel(self, checked=False):
        self.sequence_warping_enabled = checked
        if checked:
            self.sequence_warping_window.show()
            self.sequence_warping_window.raise_()
            self.sequence_warping_window.activateWindow()
        else:
            self.sequence_warping_window.hide()

    def onSequenceWarpingWindowClosed(self):
        self.sequence_warping_enabled = False
        self.actions.sequence_warping.setChecked(False)

    def closeSequenceWarpingPanel(self):
        if self.sequence_warping_window.isVisible():
            self.sequence_warping_window.close()
        else:
            self.onSequenceWarpingWindowClosed()

    def toggleTrackingPredictionPanel(self, checked=False):
        self.tracking_prediction_enabled = checked
        if checked:
            self.tracking_prediction_window.show()
            self.tracking_prediction_window.raise_()
            self.tracking_prediction_window.activateWindow()
        else:
            self.tracking_prediction_window.hide()

    def onTrackingPredictionWindowClosed(self):
        self.tracking_prediction_enabled = False
        self.actions.tracking_prediction.setChecked(False)

    def closeTrackingPredictionPanel(self):
        if self.tracking_prediction_window.isVisible():
            self.tracking_prediction_window.close()
        else:
            self.onTrackingPredictionWindowClosed()

    def registerTrackingAlgorithm(self, name, runner):
        """Register runner(previous_image, current_image, previous_boxes)."""
        if not name or not callable(runner):
            raise ValueError("Tracking algorithm requires a name and callable runner")
        self.tracking_algorithms[name] = runner
        if (
                self.tracking_algorithm_combo.count() == 1
                and self.tracking_algorithm_combo.itemData(0) is None
        ):
            self.tracking_algorithm_combo.clear()
        existing = self.tracking_algorithm_combo.findData(name)
        if existing < 0:
            self.tracking_algorithm_combo.addItem(name, name)

    def _runTemplateMatching(
            self, previous_image, current_image, previous_boxes
    ):
        return track_boxes_template_matching(
            previous_image,
            current_image,
            previous_boxes,
            score_threshold=self.template_match_threshold.value(),
            nms_threshold=self.template_nms_threshold.value(),
        )

    def _updateTrackingParameterState(self, _index=None):
        enabled = (
                self.tracking_algorithm_combo.currentData() == self.tr("模板匹配")
        )
        for widget in (
                self.template_match_threshold_label,
                self.template_match_threshold,
                self.template_nms_threshold_label,
                self.template_nms_threshold,
        ):
            widget.setEnabled(enabled)

    @staticmethod
    def _trackingBoxesFromShapes(shapes):
        boxes = []
        for shape in shapes:
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            if x1 < x2 and y1 < y2:
                boxes.append([shape.get("label", "object"), [x1, y1], [x2, y2]])
        return boxes

    @staticmethod
    def _normalizeTrackingBoxes(boxes):
        normalized = []
        for box in boxes or []:
            if not isinstance(box, (list, tuple)) or len(box) < 3:
                raise ValueError("跟踪结果必须为 [标签, 左上点, 右下点]")
            label, point1, point2 = box[:3]
            if len(point1) != 2 or len(point2) != 2:
                raise ValueError("跟踪框坐标必须是二维坐标")
            x1, x2 = sorted((float(point1[0]), float(point2[0])))
            y1, y2 = sorted((float(point1[1]), float(point2[1])))
            if x1 < x2 and y1 < y2:
                normalized.append([str(label), [x1, y1], [x2, y2]])
        return normalized

    def runTrackingPrediction(self, _value=False):
        algorithm_name = self.tracking_algorithm_combo.currentData()
        runner = self.tracking_algorithms.get(algorithm_name)
        if runner is None:
            self.errorMessage(self.tr("跟踪预测"), self.tr("当前没有可用的跟踪算法。"))
            return
        if not self.filename or self.filename not in self.imageList:
            self.errorMessage(self.tr("跟踪预测"), self.tr("请先打开当前帧图像。"))
            return
        current_index = self.imageList.index(self.filename)
        if current_index == 0:
            self.errorMessage(self.tr("跟踪预测"), self.tr("当前图像没有上一帧。"))
            return

        previous_filename = self.imageList[current_index - 1]
        previous_label_file = self._labelFileForImage(previous_filename)
        if not osp.isfile(previous_label_file):
            self.errorMessage(self.tr("跟踪预测"), self.tr("上一帧没有标注文件。"))
            return
        try:
            with open(previous_label_file, "r", encoding="utf-8") as stream:
                previous_shapes = json.load(stream).get("shapes") or []
            previous_boxes = self._trackingBoxesFromShapes(previous_shapes)
            if not previous_boxes:
                self.errorMessage(self.tr("跟踪预测"), self.tr("上一帧没有可跟踪的检测框。"))
                return
            previous_image = cv2.imdecode(
                np.fromfile(previous_filename, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            current_image = cv2.imdecode(
                np.fromfile(self.filename, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if previous_image is None or current_image is None:
                raise ValueError("无法读取上一帧或当前帧图像")
            predicted_boxes = self._normalizeTrackingBoxes(
                runner(previous_image, current_image, previous_boxes)
            )
        except (OSError, ValueError, TypeError) as error:
            self.errorMessage(self.tr("跟踪预测失败"), str(error))
            return
        except Exception as error:
            logger.exception("Tracking prediction failed")
            self.errorMessage(self.tr("跟踪预测失败"), str(error))
            return

        if not predicted_boxes:
            self.errorMessage(self.tr("跟踪预测"), self.tr("算法没有返回有效检测框。"))
            return
        label_file = self._labelFileForImage(self.filename)
        if self.labelFile is not None:
            self.pre_shape = self.labelFile.shapes
        if self.save_AI_Labels(
                label_file,
                predicted_boxes,
                keep_existing=self.tracking_keep_existing.isChecked(),
        ):
            self.loadFile(self.filename)
            self.status(self.tr("跟踪预测完成，已保存当前帧标注。"))

    def update_homography_threshold(self):
        # Update the homography threshold from the QLineEdit input
        try:
            self.homography_thres = float(self.homography_thres_text.text())
            print(f"Updated Homography Threshold: {self.homography_thres}")
        except ValueError:
            print("Invalid input for Homography Threshold")

    def undo_shapes(self):
        try:
            polygon = []
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.pre_shape is not None:
                current_label = self.pre_shape
                for box1 in current_label:
                    polygon.append([box1['label'], box1['points'][0], box1['points'][1]])
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.pre_shape = []
            self.save_AI_Labels(label_file, polygon)
            self.loadFile(self.filename)
        except Exception as e:
            self.errorMessage(
                self.tr("An error occurred:"),
                self.tr("{}").format(
                    e
                ),
            )

    def update_object_conf_threshold(self):
        # Update the object confidence threshold from the QLineEdit input
        try:
            self.object_conf_thres = float(self.object_conf_thres_text.text())
        except ValueError:
            print("Invalid input for Object Confidence Threshold")

    def update_text2label_conf_threshold(self):
        try:
            self.text2label_conf_thres = float(self.text2label_conf_thres_text.text())
            if self.text2label_model_type == "grounding_dino" and self.text2label_model is not None:
                self.text2label_model.box_threshold = self.text2label_conf_thres
        except ValueError:
            print("Invalid input for Text2Label Confidence Threshold")

    def update_text2label_nms_threshold(self):
        try:
            value = float(self.text2label_nms_thres_text.text())
            if not 0.0 <= value <= 1.0:
                raise ValueError
            self.text2label_nms_thres = value
        except ValueError:
            self.text2label_nms_thres_text.setText(str(self.text2label_nms_thres))
            print("Invalid input for Text2Label NMS Threshold")

    def update_text2label_imgsz(self):
        try:
            new_w = int(self.text2label_imgsz_w_text.text())
            new_h = int(self.text2label_imgsz_h_text.text())
            size_changed = (new_w != self.text2label_imgsz_w) or (new_h != self.text2label_imgsz_h)
            self.text2label_imgsz_w = new_w
            self.text2label_imgsz_h = new_h
            if self.text2label_model_type == "grounding_dino" and self.text2label_model is not None:
                self.text2label_model.size = [self.text2label_imgsz_w, self.text2label_imgsz_h]
            elif self.text2label_model_type == "sam3_text" and size_changed:
                self.sam3_inference.set_text_model(
                    self.text2label_model_path, self.text2label_sam_strategy
                )
        except ValueError:
            print("Invalid input for Text2Label Image Size")

    def update_sam3_conf_threshold(self):
        try:
            self.sam3_conf_thres = float(self.sam3_conf_thres_text.text())
        except ValueError:
            print("Invalid input for SAM3 Confidence Threshold")

    def update_sam3_imgsz(self):
        try:
            self.sam3_imgsz = int(self.sam3_imgsz_text.text())
        except ValueError:
            print("Invalid input for SAM3 Image Size")

    def _limit_sequence_warping_images(self, pre_image, cur_image):
        max_side = max(
            pre_image.shape[0],
            pre_image.shape[1],
            cur_image.shape[0],
            cur_image.shape[1],
        )
        if max_side <= SEQUENCE_WARPING_MAX_SIDE:
            return pre_image, cur_image

        scale = float(SEQUENCE_WARPING_MAX_SIDE) / float(max_side)
        pre_size = (
            max(1, int(round(pre_image.shape[1] * scale))),
            max(1, int(round(pre_image.shape[0] * scale))),
        )
        cur_size = (
            max(1, int(round(cur_image.shape[1] * scale))),
            max(1, int(round(cur_image.shape[0] * scale))),
        )
        return (
            cv2.resize(pre_image, pre_size, interpolation=cv2.INTER_AREA),
            cv2.resize(cur_image, cur_size, interpolation=cv2.INTER_AREA),
        )

    def _warp_bbox_with_homography(self, points, H, image_width, image_height):
        x0, y0 = points[0]
        x1, y1 = points[1]
        left = min(x0, x1)
        right = max(x0, x1)
        top = min(y0, y1)
        bottom = max(y0, y1)
        box_corners = np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(box_corners, H).reshape(-1, 2)
        if not np.all(np.isfinite(warped_corners)):
            return None

        min_corner = warped_corners.min(axis=0)
        max_corner = warped_corners.max(axis=0)
        if (
                max_corner[0] < 0
                or max_corner[1] < 0
                or min_corner[0] >= image_width
                or min_corner[1] >= image_height
        ):
            return None

        min_corner[0] = np.clip(min_corner[0], 0, image_width - 1)
        min_corner[1] = np.clip(min_corner[1], 0, image_height - 1)
        max_corner[0] = np.clip(max_corner[0], 0, image_width - 1)
        max_corner[1] = np.clip(max_corner[1], 0, image_height - 1)

        points0 = [int(round(min_corner[0])), int(round(min_corner[1]))]
        points1 = [int(round(max_corner[0])), int(round(max_corner[1]))]
        if points0[0] >= points1[0] or points0[1] >= points1[1]:
            return None
        return points0, points1

    def sequence_warping(self, _value=False):
        if not self.sequence_warping_enabled:
            return
        # Navigation can trigger warping automatically.  A previous image is
        # allowed to have no label file (or an empty one), so treat that as a
        # no-op instead of trying to open a missing JSON file.
        if not self.filename or self.filename not in self.imageList:
            return
        index = self.imageList.index(self.filename)
        if index == 0:
            return 0
        else:
            pre_image_name = self.imageList[index - 1]
            pre_image = cv2.imdecode(np.fromfile(pre_image_name, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            cur_image = cv2.imdecode(np.fromfile(self.filename, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if pre_image is None or cur_image is None:
                logger.warning("Sequence warping skipped because an image could not be read.")
                return
            json_path = os.path.splitext(pre_image_name)[0] + '.json'
            if self.output_dir:
                json_path = osp.join(self.output_dir, osp.basename(json_path))
            if not osp.isfile(json_path):
                logger.info("Sequence warping skipped because previous image has no annotation file: %s", json_path)
                return
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Sequence warping skipped because previous annotation could not be read: %s", exc)
                return
            shapes = data.get('shapes') or []
            if not shapes:
                logger.info("Sequence warping skipped because previous annotation has no shapes.")
                return
            image_height, image_width = cur_image.shape[:2]
            H = None
            if self.check_Lucas_Kanade.isChecked():
                H = optical_flow_Homography(pre_image, cur_image)
            if self.check_SIFT.isChecked():
                H = SIFT_Homography(pre_image, cur_image)
            if H is None:
                logger.warning("Sequence warping skipped because homography estimation failed.")
                return
            polygon = []
            for shape in shapes:
                src_point0 = np.array([[shape['points'][0][0], shape['points'][0][1]]], dtype=np.float32)
                src_point1 = np.array([[shape['points'][1][0], shape['points'][1][1]]], dtype=np.float32)
                class_id = shape['label']
                if self.check_center.isChecked():
                    center_x = (shape['points'][0][0] + shape['points'][1][0]) / 2
                    center_y = (shape['points'][0][1] + shape['points'][1][1]) / 2
                    label_w = shape['points'][1][0] - shape['points'][0][0]
                    label_h = shape['points'][1][1] - shape['points'][0][1]
                    src_center = np.array([[center_x, center_y]], dtype=np.float32)
                    center = cv2.perspectiveTransform(src_center.reshape(-1, 1, 2), H).tolist()
                    points0 = [int(center[0][0][0] - (label_w / 2)), int(center[0][0][1]) - (label_h / 2)]
                    points1 = [int(center[0][0][0] + (label_w / 2)), int(center[0][0][1]) + (label_h / 2)]
                if self.check_edge.isChecked():
                    warped_box = self._warp_bbox_with_homography(
                        shape['points'], H, image_width, image_height
                    )
                    if warped_box is None:
                        continue
                    points0, points1 = warped_box
                if 0 <= points0[0] < image_width and 0 <= points0[1] < image_height and 0 <= points1[
                    0] < image_width and 0 <= points1[1] < image_height:
                    # 确保坐标在图像范围内
                    points0[0] = max(0, min(points0[0], image_width - 1))
                    points0[1] = max(0, min(points0[1], image_height - 1))
                    points1[0] = max(0, min(points1[0], image_width - 1))
                    points1[1] = max(0, min(points1[1], image_height - 1))
                    polygon.append([class_id, points0, points1])
            if self.keep_current_label.isChecked():
                if self.labelFile is not None and self.labelFile.shapes:
                    current_label = self.labelFile.shapes
                    polygon1 = polygon.copy()
                    for box1 in current_label:
                        ious = [calculate_iou([box1['points'][0], box1['points'][1]], [box2[1], box2[2]]) for box2 in
                                polygon1]
                        if all(iou < self.homography_thres for iou in ious):
                            polygon.append([box1['label'], box1['points'][0], box1['points'][1]])

            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            if self.labelFile is not None:
                self.pre_shape = self.labelFile.shapes
            self.save_AI_Labels(label_file, polygon)
            self.loadFile(self.filename)

    def object_detection_single(self, options=None, _value=False):
        image = labelme.utils.img_qt_to_arr(self.image)
        input_tensors = self.net.get_inputs()
        self.label_list = ast.literal_eval(self.net.get_modelmeta().custom_metadata_map['names'])  # 该 API 会返回列表
        task = self.net.get_modelmeta().custom_metadata_map['task']
        label_file = osp.splitext(self.imagePath)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))
        blob, ratio, image_height, image_width = preprocess(image, input_tensors)
        # blob, ratio, image_height, image_width,img_r,img,image_show= obb_detect_preprocess_adjust(image, input_tensors)
        inputs = {self.net.get_inputs()[0].name: blob}
        pred = self.net.run(None, inputs)[0]
        if self.labelFile is not None:
            self.pre_shape = self.labelFile.shapes
        if task == "obb":
            polygon = obb_detect_postprocess(pred, self.object_conf_thres, ratio, image_width, image_height)
            # polygon = obb_detect_postprocess_adjust(pred, self.object_conf_thres, ratio, image_width, image_height,img_r,img,image_show)
            self.save_AI_obb_labels(label_file, polygon)
        elif task == "detect":
            polygon = detect_postprocess(pred, self.object_conf_thres, ratio, image_width, image_height)
            self.save_AI_Labels(
                label_file,
                polygon,
                keep_existing=options["keep_existing_labels"] if options else None,
            )

    def object_detection(self, _value=False):
        self.update_object_conf_threshold()
        self._run_ai_annotation(
            "object_detection",
            self.object_detection_single,
            reload_current=True,
        )

    def Text2Label_single(self, options=None, _value=False):
        image = labelme.utils.img_qt_to_arr(self.image)
        if self.labelFile is not None:
            self.pre_shape = self.labelFile.shapes
        if image.ndim == 3 and image.shape[-1] > 3:
            image = image[:, :, :3]
        if image.ndim == 4 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        polygon = []
        if self.text2label_model_type == "sam3_text":
            prompts = self._parse_sam_text_prompts(self.Text2Label_Text.text())
            if not prompts:
                self.errorMessage(
                    self.tr("无效文本提示"),
                    self.tr("请输入至少一个文本提示，例如 car.person。"),
                )
                return
            self.update_text2label_conf_threshold()
            self.update_text2label_nms_threshold()
            self.update_text2label_imgsz()
            if self.text2label_imgsz_w != self.text2label_imgsz_h:
                self.errorMessage(
                    self.tr("无效图像尺寸"),
                    self.tr("SAM3 文本提示模型要求图像尺寸为方形，请将宽和高设置为相同值。"),
                )
                return

            detections = self.sam3_inference.run_text_prompt(
                image=image,
                prompts=prompts,
                conf=self.text2label_conf_thres,
                iou=self.text2label_nms_thres,
                imgsz=self.text2label_imgsz_w,
                should_stop=lambda: self.ai_annotation_stop_requested,
            )
            polygon.extend(
                [
                    detection["label"],
                    [int(round(detection["box"][0])), int(round(detection["box"][1]))],
                    [int(round(detection["box"][2])), int(round(detection["box"][3]))],
                ]
                for detection in detections
            )
        else:
            self.update_text2label_conf_threshold()
            self.update_text2label_nms_threshold()
            self.update_text2label_imgsz()
            boxes_filt, pred_phrases = self.text2label_model.detect(
                image,
                self.Text2Label_Text.text(),
                size=[self.text2label_imgsz_w, self.text2label_imgsz_h],
                box_threshold=self.text2label_conf_thres,
                nms_threshold=self.text2label_nms_thres,
            )
            H, W = image.shape[0], image.shape[1]
            for box, label in zip(boxes_filt, pred_phrases):
                box = box * np.array([W, H, W, H])
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]
                x0, y0, x1, y1 = box
                polygon.append([label, [int(x0), int(y0)], [int(x1), int(y1)]])
        label_file = osp.splitext(self.imagePath)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))
        self.save_AI_Labels(
            label_file,
            polygon,
            keep_existing=options["keep_existing_labels"] if options else None,
        )
        self.loadFile(self.filename)
        # if self.text2label_model_type == "sam3_text":
        #     self._show_sam_mask_overlay(result)

    def Run_Text2Label(self, _value=False):
        if not self._text2label_model_ready():
            self.errorMessage(
                self.tr("未加载模型"),
                self.tr("请先在文本提示中选择并加载模型。"),
            )
            return
        self._run_ai_annotation(
            "grounding_dino",
            self.Text2Label_single,
        )

    def Image_pass(self, _value=False):
        if not os.path.exists(self.lastOpenDir + '/image'):
            os.makedirs(self.lastOpenDir + '/image')
        if not os.path.exists(self.lastOpenDir + '/json'):
            os.makedirs(self.lastOpenDir + '/json')
        label_file = osp.splitext(self.filename)[0] + ".json"
        label_outdir = osp.join(self.lastOpenDir + '/json', str(os.path.split(label_file)[1]))
        image_outdir = osp.join(self.lastOpenDir + '/image', str(os.path.split(self.filename)[1]))
        self.saveLabels(label_file, True)
        if osp.exists(self.lastOpenDir + '/image'):
            shutil.copyfile(self.filename, image_outdir)
            shutil.copyfile(label_file, label_outdir)

    def Image_unpass(self, _value=False):
        label_file = osp.splitext(self.filename)[0] + ".json"
        self.saveLabels(label_file, False)

    def create_dataset(self, _value=False):
        dialog = Create_dataset_Dialog()
        dialog.exec_()

    def dataset_analysis(self, _value=False):
        dialog = Dataset_analysis_Dialog()
        dialog.exec_()

    def advanced_config(self, _value=False):
        """Open a small Python editor for custom per-image annotation edits."""
        if self.filename is None or not self.imageData:
            self.errorMessage(self.tr("高级配置"), self.tr("请先打开图像"))
            return
        if self._advanced_dialog is not None:
            try:
                if self._advanced_dialog.isVisible():
                    self._advanced_dialog.raise_()
                    self._advanced_dialog.activateWindow()
                    return
            except RuntimeError:
                # The Qt object was deleted after closing; create a new dialog.
                self._advanced_dialog = None
        dialog = QtWidgets.QDialog(self)
        self._advanced_dialog = dialog
        dialog.setModal(False)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_: setattr(self, "_advanced_dialog", None))
        dialog.setWindowTitle(self.tr("高级配置 - Python 标注处理"))
        dialog.resize(760, 520)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel(
            self.tr("可用变量：image（PIL 图像）、shapes（当前标注列表）、np。直接修改 shape.label/points 等属性。")
        ))
        presets = {
            "按面积设置类别": "for shape in shapes:\n    pts = [(float(p.x()), float(p.y())) for p in shape.points]\n    area = abs((max(x for x, y in pts)-min(x for x, y in pts)) * (max(y for x, y in pts)-min(y for x, y in pts))) if len(pts) >= 2 else 0\n    if area < 100:\n        shape.label = '5'",
            "所有目标统一类别": "for shape in shapes:\n    shape.label = '5'",
            "扩展检测框 5 像素": "for shape in shapes:\n    if len(shape.points) >= 2:\n        x1, y1 = shape.points[0].x(), shape.points[0].y()\n        x2, y2 = shape.points[1].x(), shape.points[1].y()\n        shape.points = [[x1-5, y1-5], [x2+5, y2+5]]",
            "Mask/阈值示例": "# threshold: maximum difference from the brightest grayscale pixel\nthreshold = 20\npadding = 10\nif threshold < 0:\n    raise ValueError(\"threshold must be non-negative\")\ngray = np.asarray(image.convert(\"L\"), dtype=np.float64)\nheight, width = gray.shape\nfor shape in shapes:\n    if shape.shape_type != \"rectangle\" or len(shape.points) < 2:\n        continue\n    xs = [float(p.x()) for p in shape.points]\n    ys = [float(p.y()) for p in shape.points]\n    left = max(0, int(np.floor(min(xs))) - padding)\n    top = max(0, int(np.floor(min(ys))) - padding)\n    right = min(width, int(np.ceil(max(xs))) + padding)\n    bottom = min(height, int(np.ceil(max(ys))) + padding)\n    region = gray[top:bottom, left:right]\n    if region.size == 0:\n        continue\n    mask = region.max() - region <= threshold\n    yy, xx = np.nonzero(mask)\n    if xx.size:\n        shape.points = [\n            QtCore.QPointF(float(left + xx.min()), float(top + yy.min())),\n            QtCore.QPointF(float(left + xx.max() + 1), float(top + yy.max() + 1)),\n        ]",
        }
        instance_row = QtWidgets.QHBoxLayout()
        instance_combo = QtWidgets.QComboBox(dialog)
        instance_combo.addItems(list(presets))
        new_instance = QtWidgets.QPushButton(self.tr("新建空白实例"), dialog)
        save_instance = QtWidgets.QPushButton(self.tr("保存实例"), dialog)
        auto_apply = QtWidgets.QCheckBox(self.tr("打开图像时自动应用"), dialog)
        instance_row.addWidget(QtWidgets.QLabel(self.tr("实例:")))
        instance_row.addWidget(instance_combo, 1)
        instance_row.addWidget(new_instance)
        instance_row.addWidget(save_instance)
        instance_row.addWidget(auto_apply)
        layout.addLayout(instance_row)
        editor = QtWidgets.QPlainTextEdit(dialog)
        editor.setPlainText(presets[instance_combo.currentText()])
        new_instance.clicked.connect(lambda: self._create_blank_advanced_instance(instance_combo, editor))
        instance_combo.currentTextChanged.connect(lambda name: editor.setPlainText(
            self._advanced_instances.get(name, presets.get(name, ""))))
        save_instance.clicked.connect(lambda: self._save_advanced_instance(
            dialog, instance_combo, editor, auto_apply))
        auto_apply.toggled.connect(lambda checked: self._set_advanced_auto_apply(
            checked, editor))

        # Keep the full example available when no preset is selected.
        editor.setToolTip(self.tr("代码将在当前进程中执行，请仅运行可信代码"))
        """editor.setPlainText(
            "# image: 当前帧 PIL.Image；shapes: 当前帧标注对象列表\n"
            "# 示例：将面积小于 100 的标注类别设置为 5\n"
            "for shape in shapes:\n"
            "    points = [(float(p.x()), float(p.y())) for p in shape.points]\n"
            "    if len(points) >= 2:\n"
            "        area = abs((max(x for x, y in points) - min(x for x, y in points)) *\n"
            "                   (max(y for x, y in points) - min(y for x, y in points)))\n"
            "        if area < 100:\n"
            "            shape.label = '5'\n"
        )"""
        layout.addWidget(editor, 1)
        buttons = QtWidgets.QHBoxLayout()
        run_button = QtWidgets.QPushButton(self.tr("运行"), dialog)
        close_button = QtWidgets.QPushButton(self.tr("关闭"), dialog)
        buttons.addWidget(run_button)
        buttons.addStretch()
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        def run_code():
            self.setEnabled(False)
            try:
                code = editor.toPlainText()
                self._execute_advanced_code(code)
                if auto_apply.isChecked():
                    self._advanced_code = code
                    self._advanced_auto_apply = True
                self.status(self.tr("高级配置运行完成"))
            except Exception as exc:
                QtWidgets.QMessageBox.critical(dialog, self.tr("运行失败"), str(exc))
            finally:
                self.setEnabled(True)
                dialog.setEnabled(True)

        run_button.clicked.connect(run_code)
        QtWidgets.QShortcut(QtGui.QKeySequence("L"), dialog).activated.connect(run_code)
        close_button.clicked.connect(dialog.accept)
        dialog.show()

    def _save_advanced_instance(self, dialog, combo, editor, auto_apply):
        name, ok = QtWidgets.QInputDialog.getText(dialog, self.tr("保存实例"), self.tr("实例名称："))
        name = str(name).strip()
        if not ok or not name:
            return
        self._advanced_instances[name] = editor.toPlainText()
        if combo.findText(name) < 0:
            combo.addItem(name)
        combo.setCurrentText(name)
        if auto_apply.isChecked():
            self._advanced_code = editor.toPlainText()
            self._advanced_auto_apply = True

    def _create_blank_advanced_instance(self, combo, editor):
        name = self.tr("空白实例")
        suffix = 1
        while combo.findText(name) >= 0:
            suffix += 1
            name = self.tr("空白实例 {}").format(suffix)
        combo.addItem(name)
        combo.setCurrentText(name)
        editor.setPlainText("")

    def _set_advanced_auto_apply(self, checked, editor):
        self._advanced_auto_apply = bool(checked)
        self._advanced_code = editor.toPlainText() if checked else None

    def _execute_advanced_code(self, code):
        image = utils.img_data_to_pil(self.imageData)
        shapes = [item.shape() for item in self.labelList]
        exec(code, {"__builtins__": __builtins__, "image": image,
                    "shapes": shapes, "np": __import__("numpy"), "QtCore": QtCore})
        items = list(self.labelList)
        for item in items:
            shape = item.shape()
            if shape.points and isinstance(shape.points[0], (list, tuple)):
                shape.points = [QtCore.QPointF(float(p[0]), float(p[1])) for p in shape.points]
            shape.label = str(shape.label)
        # Normalize all boxes before updates trigger repaint or autosave.
        for item in items:
            self._update_item(item, item.shape().label, None, None, None)
        self.canvas.update()
        self.setDirty()

    def set_category(self, _value=False):
        """Set every annotation in the current image to one category."""
        if self.filename is None or not len(self.labelList):
            self.errorMessage(self.tr("设置类别"), self.tr("当前图像没有可修改的标注"))
            return

        # Use configured labels as suggestions, while allowing a new label.
        labels = [str(label) for label in (self._config.get("labels") or [])]
        labels.extend(
            item.shape().label for item in self.labelList if item.shape().label not in labels
        )
        category, ok = QtWidgets.QInputDialog.getItem(
            self,
            self.tr("设置类别"),
            self.tr("将当前图像所有标注设置为："),
            labels,
            0,
            True,
        )
        category = str(category).strip()
        if not ok or not category:
            return

        configured_id = None
        if category in (self._config.get("labels") or []):
            configured_id = list(self._config["labels"]).index(category)
        if self.annotation_format == "txt" and configured_id is None:
            try:
                configured_id = int(category)
            except ValueError:
                self.errorMessage(
                    self.tr("设置类别"),
                    self.tr("YOLO TXT 需要在配置 labels 中定义该类别，或输入数字类别 ID。"),
                )
                return
        self._category_override = category
        # Apply immediately to the current image as confirmation that the
        # category was accepted; later images require the shortcut explicitly.
        if len(self.labelList):
            self._apply_category_override(category, configured_id)

    def apply_category(self, _value=False):
        """Apply the previously selected fixed category to this image."""
        if not self._category_override:
            self.errorMessage(self.tr("应用固定类别"), self.tr("请先设置固定类别"))
            return
        if not len(self.labelList):
            self.errorMessage(self.tr("应用固定类别"), self.tr("当前图像没有可修改的标注"))
            return
        self._apply_category_override(self._category_override)

    def _apply_category_override(self, category, configured_id=None):
        """Apply the inherited category to all current shapes."""
        if configured_id is None and category in (self._config.get("labels") or []):
            configured_id = list(self._config["labels"]).index(category)
        if self.annotation_format == "txt" and configured_id is None:
            try:
                configured_id = int(category)
            except (TypeError, ValueError):
                return
        for item in list(self.labelList):
            shape = item.shape()
            self._update_item(item, category, None, None, None)
            # A stale YOLO class id would otherwise override the changed text
            # when saving TXT annotations.
            if self.annotation_format == "txt":
                if configured_id is None:
                    shape.other_data.pop("yolo_class_id", None)
                else:
                    shape.other_data["yolo_class_id"] = configured_id
        self.canvas.update()

    def Yolovis(self, _value=False):
        dialog = Yolo_Vis_Dialog()
        dialog.exec_()

    def Video_slice(self, _value=False):
        dialog = Video_slice_Dialog()
        dialog.exec_()

    def Data_augmentation(self, _value=False):
        dialog = Data_augmentation_Dialog()
        dialog.exec_()

    def copy_paste_augmentation(self, _value=False):
        # TODO
        dialog = copy_paste_augmentation_Dialog()
        dialog.exec_()

    def Slice_Dataset(self, _value=False):
        dialog = Slice_dataset()
        dialog.exec_()

    def Concat_Dataset(self, _value=False):
        dialog = Concat_dataset()
        dialog.exec_()

    def resizeEvent(self, event):
        if (
                self.canvas
                and not self.image.isNull()
                and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        # value = 329
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def setDeleteConfirmationEnabled(self, enabled):
        self.settings.setValue("confirmDelete", bool(enabled))

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        self.settings.setValue(
            "confirmDelete", self.actions.confirmDelete.isChecked()
        )
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
                Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
                Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)
            if self.sequence_warping_enabled:
                self.sequence_warping()

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()
        self._refreshDetectionPanelState()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.annotation_format == "txt" and self.annotation_path:
            self._saveFile(self.annotation_path)
        elif self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        suffix = ".txt" if self.annotation_format == "txt" else LabelFile.suffix
        filters = self.tr("Label files (*%s)") % suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.closeObjectDetectionPanel()
        self.target_segmentation_window.close()
        self.closeSequenceWarpingPanel()
        self.closeTrackingPredictionPanel()
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.annotation_path:
            return self.annotation_path
        if self.annotation_format == "txt":
            return self._currentAnnotationFile()
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        if self.filename is None:
            return

        label_file = self.getLabelFile()
        if not osp.exists(label_file):
            return
        try:
            os.remove(label_file)
        except OSError as e:
            self.errorMessage(self.tr("Error deleting annotation file"), str(e))
            return
        logger.info("Label file is removed: %s", label_file)

        filename = self.filename
        images = self.imageList
        if filename in images:
            row = images.index(filename)
            self.fileListWidget.item(row).setCheckState(Qt.Unchecked)
            if row + 1 < len(images):
                filename = images[row + 1]

        self.setClean()
        if filename == label_file:
            self.closeFile()
        else:
            self.loadFile(filename)

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        # yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        # msg = self.tr(
        #     "You are about to permanently delete {} polygons, " "proceed anyway?"
        # ).format(len(self.canvas.selectedShapes))
        # if yes == QtWidgets.QMessageBox.warning(
        #         self, self.tr("Attention"), msg, yes | no, yes
        # ):
        selected_shapes = [shape.copy() for shape in self.canvas.selectedShapes]
        self.remLabels(self.canvas.deleteSelected())
        self.setDirty()
        if self.batch_processing_enabled and selected_shapes:
            self._batchDeleteOverlappingShapes(selected_shapes)
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        moves = []
        if self.batch_processing_enabled:
            for old_shape, new_shape in zip(self.canvas.selectedShapes, self.canvas.selectedShapesCopy):
                if old_shape.points and new_shape.points:
                    moves.append((old_shape.copy(), new_shape.points[0].x() - old_shape.points[0].x(),
                                  new_shape.points[0].y() - old_shape.points[0].y()))
        self.canvas.endMove(copy=False)
        self.setDirty()
        self._batchMoveShapes(moves)

    # Batch processing -------------------------------------------------
    # These helpers operate on label JSON files directly so the visible frame,
    # its undo history and its selection are not disturbed by batch updates.
    def showBatchProcessDialog(self, _value=False):
        total = len(self.imageList)
        if not total:
            return
        for spin in (self.batch_start_spin, self.batch_end_spin):
            spin.setRange(1, total)
        current = self.imageList.index(self.filename) + 1 if self.filename in self.imageList else 1
        if not self.batch_processing_enabled:
            self.batch_start_spin.setValue(current)
            self.batch_end_spin.setValue(current)
            self.batch_process_status.setText(self.tr("尚未开始"))
            self.batch_process_start_button.setText(self.tr("开始批量处理"))
        self.batch_process_dialog.show()
        self.batch_process_dialog.raise_()
        self.batch_process_dialog.activateWindow()

    def startBatchProcessing(self):
        start = self.batch_start_spin.value()
        end = self.batch_end_spin.value()
        if start > end:
            QtWidgets.QMessageBox.warning(self, self.tr("批量处理"), self.tr("起始帧不能大于结束帧。"))
            return
        self.batch_start_index = start - 1
        self.batch_end_index = end - 1
        self.batch_processing_enabled = True
        self.batch_process_start_button.setEnabled(False)
        self.batch_process_stop_button.setEnabled(True)
        self.batch_process_status.setText(
            self.tr("已启用：第 {} 张到第 {} 张").format(start, end)
        )

    def stopBatchProcessing(self):
        self.batch_processing_enabled = False
        self.batch_process_start_button.setEnabled(True)
        self.batch_process_stop_button.setEnabled(False)
        self.batch_process_status.setText(self.tr("已停止；后续操作仅作用于当前帧。"))

    def _batchImageFilenames(self):
        if not self.batch_processing_enabled:
            return []
        return self.imageList[self.batch_start_index:self.batch_end_index + 1]

    def _batchLabelFilename(self, image_filename):
        basename = osp.splitext(osp.basename(image_filename))[0] + LabelFile.suffix
        return osp.join(self.output_dir, basename) if self.output_dir else osp.splitext(image_filename)[
                                                                               0] + LabelFile.suffix

    def _shapeToData(self, shape):
        data = shape.other_data.copy()
        data.update(dict(
            label=shape.label,
            points=[(point.x(), point.y()) for point in shape.points],
            group_id=shape.group_id,
            description=shape.description,
            shape_type=shape.shape_type,
            flags=shape.flags,
            mask=None if shape.mask is None else utils.img_arr_to_b64(shape.mask.astype(np.uint8)),
        ))
        if shape.shape_type == "rotate":
            data["direction"] = shape.direction
        return data

    @staticmethod
    def _batchShapeRect(shape):
        points = shape.get("points") if isinstance(shape, dict) else shape.points
        if not points:
            return None
        xy = [(point[0], point[1]) if isinstance(point, (list, tuple)) else (point.x(), point.y()) for point in points]
        xs, ys = zip(*xy)
        return QtCore.QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def _batchOverlaps(self, first, second):
        first_rect, second_rect = self._batchShapeRect(first), self._batchShapeRect(second)
        return first_rect is not None and second_rect is not None and first_rect.intersects(second_rect)

    def _loadBatchLabel(self, image_filename):
        label_filename = self._batchLabelFilename(image_filename)
        if osp.exists(label_filename) and LabelFile.is_label_file(label_filename):
            return LabelFile(label_filename), label_filename
        label_file = LabelFile()
        label_file.shapes, label_file.flags, label_file.otherData, label_file.imagePass = [], {}, {}, None
        label_file.imagePath = osp.relpath(image_filename, osp.dirname(label_filename))
        label_file.imageData = LabelFile.load_image_file(image_filename)
        return label_file, label_filename

    def _saveBatchLabel(self, label_file, label_filename, image_filename):
        image = QtGui.QImage.fromData(label_file.imageData or LabelFile.load_image_file(image_filename))
        label_file.save(
            filename=label_filename, shapes=label_file.shapes,
            imagePath=label_file.imagePath or osp.relpath(image_filename, osp.dirname(label_filename)),
            imageData=label_file.imageData if self._config["store_data"] else None,
            imageHeight=image.height(), imageWidth=image.width(), otherData=label_file.otherData,
            flags=label_file.flags, imagePass=label_file.imagePass,
        )

    def _applyToBatchLabels(self, operation):
        for image_filename in self._batchImageFilenames():
            if image_filename == self.filename:
                continue
            try:
                label_file, label_filename = self._loadBatchLabel(image_filename)
                if operation(label_file.shapes):
                    self._saveBatchLabel(label_file, label_filename, image_filename)
            except (LabelFileError, OSError) as error:
                logger.warning("Batch processing skipped %s: %s", image_filename, error)

    def _batchPasteShapes(self, shapes):
        copied_shapes = [self._shapeToData(shape) for shape in shapes]
        self._applyToBatchLabels(lambda target: target.extend(copy.deepcopy(copied_shapes)) or True)

    def _batchDeleteOverlappingShapes(self, source_shapes):
        def delete(target):
            retained = [shape for shape in target if
                        not any(self._batchOverlaps(shape, source) for source in source_shapes)]
            changed = len(retained) != len(target)
            target[:] = retained
            return changed

        self._applyToBatchLabels(delete)

    def batchMoveSelectedShapes(self):
        if not self.batch_processing_enabled or not self.canvas.selectedShapes or len(self.canvas.shapesBackups) < 2:
            return
        previous_shapes = self.canvas.shapesBackups[-2]
        moves = []
        for shape in self.canvas.selectedShapes:
            try:
                old_shape = previous_shapes[self.canvas.shapes.index(shape)]
            except ValueError:
                continue
            if not old_shape.points or not shape.points:
                continue
            dx = shape.points[0].x() - old_shape.points[0].x()
            dy = shape.points[0].y() - old_shape.points[0].y()
            if dx or dy:
                moves.append((old_shape, dx, dy))
        self._batchMoveShapes(moves)

    def _batchMoveShapes(self, moves):
        if not moves:
            return

        def move(target):
            changed = False
            for target_shape in target:
                for source_shape, dx, dy in moves:
                    if self._batchOverlaps(target_shape, source_shape):
                        target_shape["points"] = [[point[0] + dx, point[1] + dy] for point in target_shape["points"]]
                        changed = True
                        break
            return changed

        self._applyToBatchLabels(move)

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        return self.fileListWidget.imageList

    def _labelFileForImage(self, filename):
        return self._annotationFileForImage(filename)

    def _annotationCandidate(self, filename, suffix):
        path = osp.splitext(filename)[0] + suffix
        if self.output_dir:
            path = osp.join(self.output_dir, osp.basename(path))
        return path

    def _annotationFileForImage(self, filename):
        """Prefer a same-name JSON annotation, then use a same-name TXT."""
        json_file = self._annotationCandidate(filename, ".json")
        txt_file = self._annotationCandidate(filename, ".txt")
        if osp.isfile(json_file):
            return json_file
        if osp.isfile(txt_file):
            return txt_file
        return json_file

    def _currentAnnotationFile(self):
        if self.annotation_path:
            return self.annotation_path
        suffix = ".txt" if self.annotation_format == "txt" else ".json"
        return self._annotationCandidate(self.imagePath or self.filename, suffix)

    def _applyAnnotationModeRestrictions(self):
        """TXT mode supports only ordinary rectangle creation and editing."""
        txt_mode = self.annotation_format == "txt"
        # Keep the user's current edit/rectangle mode while changing images.
        # Only an unsupported drawing mode needs to fall back to rectangle.
        if (txt_mode and not self.canvas.editing()
                and self.canvas.createMode != "rectangle"):
            self.toggleDrawMode(False, createMode="rectangle")
        restricted = (
            self.actions.createMode, self.actions.createRotateMode,
            self.actions.createCircleMode, self.actions.createLineMode,
            self.actions.createPointMode, self.actions.createLineStripMode,
            self.actions.createAiPolygonMode, self.actions.createAiMaskMode,
            self.actions.select_onnx, self.actions.object,
            self.actions.target_segmentation, self.actions.sequence_warping,
            self.actions.tracking_prediction, self.actions.batch_process,
            self.actions.rotate,
        )
        for action in restricted:
            action.setEnabled(not txt_mode)
        if txt_mode:
            self.actions.createRectangleMode.setEnabled(self.canvas.editing())
            self.actions.editMode.setEnabled(not self.canvas.editing())

    def _addFileListItem(self, filename, label_file):
        self.fileListWidget.addFile(
            filename,
            checked=(
                    QtCore.QFile.exists(label_file)
                    and (label_file.lower().endswith(".txt") or LabelFile.is_label_file(label_file))
            ),
        )

    def deleteImageAndAnnotation(self, item):
        filename = str(item.text())
        label_file = self._labelFileForImage(filename)
        if self.actions.confirmDelete.isChecked():
            message_box = QtWidgets.QMessageBox(self)
            message_box.setIcon(QtWidgets.QMessageBox.Warning)
            message_box.setWindowTitle(self.tr("删除确认"))
            message_box.setText(
                self.tr(
                    "确定要永久删除图片“{0}”及其标注文件（如果存在）吗？"
                ).format(osp.basename(filename))
            )
            message_box.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            message_box.setDefaultButton(QtWidgets.QMessageBox.No)
            message_box.button(QtWidgets.QMessageBox.Yes).setText(self.tr("删除"))
            message_box.button(QtWidgets.QMessageBox.No).setText(self.tr("取消"))
            if message_box.exec_() != QtWidgets.QMessageBox.Yes:
                return

        row = self.fileListWidget.row(item)
        was_current = filename == self.filename
        try:
            os.remove(filename)
        except OSError as e:
            self.errorMessage(self.tr("Error deleting file"), str(e))
            return

        annotation_error = None
        if osp.exists(label_file):
            try:
                os.remove(label_file)
            except OSError as e:
                annotation_error = e

        self.fileListWidget.blockSignals(True)
        self.fileListWidget.takeItem(row)
        self.fileListWidget.blockSignals(False)

        if was_current:
            self.resetState()
            self.setClean()
            if self.fileListWidget.count():
                # Removing the current item can leave QListWidget's internal
                # current index unchanged. Reset it first so selecting the item
                # now occupying this row always emits selectionChanged and loads
                # the next image (or the previous image when the last was removed).
                next_row = min(row, self.fileListWidget.count() - 1)
                self.fileListWidget.setCurrentRow(-1)
                self.fileListWidget.clearSelection()
                self.fileListWidget.setCurrentRow(next_row)
                self.fileListWidget.scrollToItem(
                    self.fileListWidget.item(next_row),
                    QtWidgets.QAbstractItemView.EnsureVisible,
                )
            else:
                self.annotationInfo.setText("Current: 0 | Total: 0 ")
                self.toggleActions(False)
                self.canvas.setEnabled(False)
        elif self.filename in self.imageList:
            self.annotationInfo.setText(
                "Current: {} | Total: {} ".format(
                    self.imageList.index(self.filename) + 1, len(self.imageList)
                )
            )

        logger.info("Image file is removed: %s", filename)
        if not annotation_error and not osp.exists(label_file):
            logger.info("Annotation file is removed (if it existed): %s", label_file)
        if annotation_error:
            self.errorMessage(
                self.tr("Error deleting annotation file"), str(annotation_error)
            )

    def deleteCurrentImageAndAnnotation(self):
        """Delete the currently selected image through the file-list workflow."""
        item = self.fileListWidget.currentItem()
        if item is not None:
            self.deleteImageAndAnnotation(item)

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = self._annotationFileForImage(file)
            self._addFileListItem(file, label_file)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)
        self.actions.image_pass.setEnabled(True)
        self.actions.image_unpass.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.fileListWidget.clear()

        filenames = self.scanAllImages(dirpath)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        for filename in filenames:
            label_file = self._annotationFileForImage(filename)
            self._addFileListItem(filename, label_file)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.normpath(osp.join(root, file))
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images
