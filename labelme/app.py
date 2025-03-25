# -*- coding: utf-8 -*-

import functools
import html
import ast
import math
import os
import os.path as osp
import re
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
from labelme.widgets import DatasetDialog
from labelme.widgets import Yolo_Vis_Dialog
from labelme.widgets import Video_slice_Dialog
from labelme.widgets import Data_augmentation_Dialog
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
from labelme.ai import GroundingDINO
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.logger import logger
from labelme.shape import Shape
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
import os
import json
import numpy as np

import labelme.utils
from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()


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

        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
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
        self.canvas.mouseMoved.connect(
            lambda pos: self.status(f"Mouse is at: x={pos.x()}, y={pos.y()}")
        )

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
        object = action(
            self.tr("&目标检测"),
            self.object_detection,
            shortcuts["object"],
            "object",
            self.tr("目标检测"),
            enabled=False,
        )
        sequence_warping = action(
            self.tr("&标注映射"),
            self.sequence_warping,
            shortcuts["sequence_warping"],
            "warping",
            self.tr("标注映射"),
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
        dataset = action(
            self.tr("&数据集生成"),
            self.Dataset,
            shortcuts["dataset"],
            "dataset",
            self.tr("数据集生成"),
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
        slice_dataset = action(
            self.tr("&数据集分割"),
            self.Slice_Dataset,
            shortcuts["dataset"],
            "slice",
            self.tr("数据分割"),
        )
        concat_dataset = action(
            self.tr("&数据集合并"),
            self.Concat_Dataset,
            shortcuts["dataset"],
            "concat",
            self.tr("数据合并"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
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
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
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
            select_onnx=select_onnx,
            object=object,
            image_pass=image_pass,
            image_unpass=image_unpass,
            QCResult=QCResult,
            dataset=dataset,
            slice_dataset=slice_dataset,
            data_augmentation=data_augmentation,
            concat_dataset=concat_dataset,
            yolovis=yolovis,
            video_slice=video_slice,
            close=close,
            deleteFile=deleteFile,
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
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
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
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))
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
        Check_AI_config_layout = QtWidgets.QWidgetAction(self)
        Check_AI_config_layout.setDefaultWidget(QtWidgets.QWidget())
        Check_AI_config_layout.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        Check_AI_config_layout.defaultWidget().layout().addWidget(self.check_all_infer)
        Check_AI_config_layout.defaultWidget().layout().addWidget(self.check_save_existing_label)

        self.check_Lucas_Kanade = QtWidgets.QCheckBox("L-K", self)
        self.check_SIFT = QtWidgets.QCheckBox("SIFT", self)
        self.check_SIFT.setChecked(True)
        self.homography_group = QtWidgets.QButtonGroup(self)
        self.homography_group.addButton(self.check_Lucas_Kanade)
        self.homography_group.addButton(self.check_SIFT)
        self.homography_group.setExclusive(True)  # 互斥模式
        Check_Homography_layout = QtWidgets.QWidgetAction(self)
        Check_Homography_layout.setDefaultWidget(QtWidgets.QWidget())
        Check_Homography_layout.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        Check_Homography_layout.defaultWidget().layout().addWidget(self.check_Lucas_Kanade)
        Check_Homography_layout.defaultWidget().layout().addWidget(self.check_SIFT)
        self.check_center = QtWidgets.QCheckBox("中心映射", self)
        self.check_edge = QtWidgets.QCheckBox("边缘映射", self)
        self.check_edge.setChecked(True)
        self.warp_method_group = QtWidgets.QButtonGroup(self)
        self.warp_method_group.addButton(self.check_center)
        self.warp_method_group.addButton(self.check_edge)
        self.warp_method_group.setExclusive(True)  # 互斥模式
        Check_warp_method_layout = QtWidgets.QWidgetAction(self)
        Check_warp_method_layout.setDefaultWidget(QtWidgets.QWidget())
        Check_warp_method_layout.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        Check_warp_method_layout.defaultWidget().layout().addWidget(self.check_center)
        Check_warp_method_layout.defaultWidget().layout().addWidget(self.check_edge)
        self.homography_thres = 0.3
        self.homography_thres_text = QtWidgets.QLineEdit(self)
        self.homography_thres_text.setPlaceholderText("IOU(0-1)")
        self.homography_thres_text.setFixedWidth(80)
        self.homography_thres_button = QtWidgets.QPushButton("设置iou", self)
        self.keep_current_label = QtWidgets.QCheckBox("保留当前", self)
        self.keep_current_label.setFixedWidth(80)
        self.undo_homography_button = QtWidgets.QPushButton("映射撤消", self)
        homography_thres_layout = QtWidgets.QHBoxLayout()
        homography_thres_layout.addWidget(self.homography_thres_text)
        homography_thres_layout.addWidget(self.homography_thres_button)

        homography_undo_layout = QtWidgets.QHBoxLayout()
        homography_undo_layout.addWidget(self.keep_current_label)
        homography_undo_layout.addWidget(self.undo_homography_button)
        # self.undo_homography_button.setFixedWidth(100)
        # self.homography_thres_button.setFixedWidth(100)
        homography_thres_action = QtWidgets.QWidgetAction(self)
        homography_thres_action.setDefaultWidget(QtWidgets.QWidget())
        homography_thres_action.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        homography_thres_action.defaultWidget().setFixedWidth(180)
        homography_thres_action.defaultWidget().layout().addLayout(homography_thres_layout)
        homography_thres_action.defaultWidget().layout().addLayout(homography_undo_layout)
        self.object_conf_thres = 0.25
        self.object_conf_thres_text = QtWidgets.QLineEdit(self)
        self.object_conf_thres_text.setPlaceholderText("Conf(0-1)")
        self.object_conf_thres_button = QtWidgets.QPushButton("检测阈值", self)
        object_conf_thres_action = QtWidgets.QWidgetAction(self)
        object_conf_thres_action.setDefaultWidget(QtWidgets.QWidget())
        object_conf_thres_action.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        object_conf_thres_action.defaultWidget().setFixedWidth(100)
        object_conf_thres_action.defaultWidget().layout().addWidget(self.object_conf_thres_text)
        object_conf_thres_action.defaultWidget().layout().addWidget(self.object_conf_thres_button)
        self.homography_thres_button.clicked.connect(self.update_homography_threshold)
        self.object_conf_thres_button.clicked.connect(self.update_object_conf_threshold)
        self.undo_homography_button.clicked.connect(self.undo_homography)
        self.pre_homography_shape = []

        # 创建一个主widget和布局
        selectAiWidget = QtWidgets.QWidget()
        selectAiLayout = QtWidgets.QVBoxLayout(selectAiWidget)

        # 第一个 "DINO" 部分
        selectAiText2LabelModelLabel = QtWidgets.QLabel(self.tr("DINO:"))
        selectAiText2LabelModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        self._selectAiText2LabelModelComboBox = QtWidgets.QComboBox()

        # 创建第一个部分的布局
        text2LabelLayout = QtWidgets.QHBoxLayout()
        text2LabelLayout.addWidget(selectAiText2LabelModelLabel)
        text2LabelLayout.addWidget(self._selectAiText2LabelModelComboBox)
        # 添加第一个部分到主布局
        selectAiLayout.addLayout(text2LabelLayout)
        # 第二个 "SAM" 部分
        selectAiModelLabel = QtWidgets.QLabel(self.tr("SAM:"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        self._selectAiModelComboBox = QtWidgets.QComboBox()
        # 创建第二个部分的布局
        aiModelLayout = QtWidgets.QHBoxLayout()
        aiModelLayout.addWidget(selectAiModelLabel)
        aiModelLayout.addWidget(self._selectAiModelComboBox)
        # 添加第二个部分到主布局
        selectAiLayout.addLayout(aiModelLayout)
        # 设置整体布局
        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(selectAiWidget)

        text2label_model_names = [model.name for model in Text2LabelMODELS]
        self._selectAiText2LabelModelComboBox.addItems(text2label_model_names)
        # 禁止自动加载模型 提升软件整体初始化速度
        text_model_index = 0
        if self._config["ai"]["text_default"] in text2label_model_names:
            text_model_index = text2label_model_names.index(self._config["ai"]["text_default"])
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
        Text2Label_Input = QtWidgets.QWidgetAction(self)
        Text2Label_Input.setDefaultWidget(QtWidgets.QWidget())
        Text2Label_Input.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        Text2Label_Input.defaultWidget().setFixedWidth(200)
        self.Text2Label_Text = QtWidgets.QLineEdit(self)
        self.Text2Label_detect_button = QtWidgets.QPushButton("TEXT-LABEL", self)
        self.Text2Label_Text.setAlignment(QtCore.Qt.AlignCenter)
        Text2Label_Input.defaultWidget().layout().addWidget(self.Text2Label_Text)
        Text2Label_Input.defaultWidget().layout().addWidget(self.Text2Label_detect_button)
        self.Text2Label_detect_button.clicked.connect(self.Run_Text2Label)
        Text2Label_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(shortcuts["Text2Label"]), self)
        Text2Label_shortcut.activated.connect(self.Text2Label_detect_button.click)
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

        self.tools = self.toolbar("Tools")
        self.lefttools = self.lefttoolbar("LeftTools")

        self.actions.lefttool = (
            select_onnx,
            object,
            rotate,
            # video_object,
            zoomIn,
            zoomOut,
            dataset,
            slice_dataset,
            concat_dataset,
            yolovis,
            video_slice,
            data_augmentation,
            None,
            image_pass,
            image_unpass,
            QCResult,
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
            fitWindow,
            zoom,
            None,
            sequence_warping,
            Check_Homography_layout,
            Check_warp_method_layout,
            homography_thres_action,
            None,
            Check_AI_config_layout,
            object_conf_thres_action,
            None,
            selectAiModel,
            Text2Label_Input,
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
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
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
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
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
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)

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

    def save_AI_Labels(self, filename, res, imagePass=None):
        lf = LabelFile()
        self.setDirty()
        self.loadFile(self.filename)

        def format_shape(s):
            # doc = {}
            # doc[0] = 'people'
            # doc[1] = 'car'
            # doc[2] = 'tent'
            # doc[3] = 'snadbags'
            # doc[4] = 'cone'
            # doc[5] = 'pipeline'
            # doc[6] = 'tank'
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

    def duplicateSelectedShape(self):
        added_shapes = self.canvas.duplicateSelectedShapes()
        self.labelList.clearSelection()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

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
        # assumes same name, but json extension
        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
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
        total_files = len(self.imageList)  # 总标注数量
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
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        self.actions.rotate.setEnabled(False)
        self.actions.select_onnx.setEnabled(True)
        # self.actions.object.setEnabled(True)
        self.actions.sequence_warping.setEnabled(True)
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
            # self.actions.object.setEnabled(True)
            self.actions.sequence_warping.setEnabled(True)
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
            self.actions.sequence_warping.setEnabled(False)
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
        # dialog = Selectonnx()
        # dialog.exec_()
        # file_dialog = QtWidgets.QFileDialog()
        # file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        # file_dialog.setNameFilter('Text files (*.onnx);;All files (*.*)')
        # self.onnx_path=file_dialog.selectedFiles()[0]
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select ONNX File', '', 'ONNX Files (*.onnx);;All Files (*)',
                                                   options=options)
        self.onnx_path = file_name
        if not self.onnx_path:
            self.errorMessage(
                self.tr("Invalid onnxfile"),
                self.tr("Invalid onnxfile '{}'").format(
                    self.onnx_path
                ),
            )
            return False
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
        self.actions.rotate.setEnabled(False)

    # D:\code\Grounded-Segment-Anything\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py
    # D:\code\Grounded-Segment-Anything\groundingdino_swint_ogc.pth
    def init_text2lable_model(self, config_path, model_path, device, _value=False):
        if not model_path:
            self.errorMessage(
                self.tr("Invalid model_path"),
                self.tr("Invalid model_path '{}'").format(
                    model_path
                ),
            )
            return False
        self.text2label_model = GroundingDINO(model_path, 0.3, "./ai/seg_model/vocab.txt", device, 0.25)
        self.actions.rotate.setEnabled(False)
        return self.text2label_model

    def update_homography_threshold(self):
        # Update the homography threshold from the QLineEdit input
        try:
            self.homography_thres = float(self.homography_thres_text.text())
            print(f"Updated Homography Threshold: {self.homography_thres}")
        except ValueError:
            print("Invalid input for Homography Threshold")

    def undo_homography(self):
        try:
            polygon = []
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.pre_homography_shape is not None:
                current_label = self.pre_homography_shape
                for box1 in current_label:
                    polygon.append([box1['label'], box1['points'][0], box1['points'][1]])
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.pre_homography_shape = []
            self.save_AI_Labels(label_file, polygon)
            self.loadFile(self.filename)
        except ValueError:
            print("Invalid undo")

    def update_object_conf_threshold(self):
        # Update the object confidence threshold from the QLineEdit input
        try:
            self.object_conf_thres = float(self.object_conf_thres_text.text())
            print(f"Updated Object Confidence Threshold: {self.object_conf_thres}")
        except ValueError:
            print("Invalid input for Object Confidence Threshold")

    def sequence_warping(self, _value=False):
        def calculate_iou(box1, box2):
            x1, y1, x2, y2 = box1[0][0], box1[0][1], box1[1][0], box1[1][1]
            x3, y3, x4, y4 = box2[0][0], box2[0][1], box2[1][0], box2[1][1]

            # 计算交集的坐标
            x_left = max(x1, x3)
            y_top = max(y1, y3)
            x_right = min(x2, x4)
            y_bottom = min(y2, y4)

            # 计算交集面积
            if x_right > x_left and y_bottom > y_top:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
            else:
                intersection_area = 0

            # 计算两个框的面积
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x4 - x3) * (y4 - y3)

            # 计算 IoU
            iou = intersection_area / (box1_area + box2_area - intersection_area) if (
                                                                                             box1_area + box2_area - intersection_area) > 0 else 0
            return iou

        index = self.imageList.index(self.filename)
        if index == 0:
            return 0
        else:
            pre_image_name = self.imageList[index - 1]
            pre_image = cv2.imdecode(np.fromfile(pre_image_name, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            cur_image = cv2.imdecode(np.fromfile(self.filename, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            # pre_image = cv2.imread(pre_image_name, cv2.IMREAD_GRAYSCALE)
            # cur_image = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            image_height, image_width = cur_image.shape[:2]
            if self.check_Lucas_Kanade.isChecked():
                feature_params = dict(maxCorners=500, qualityLevel=0.1, minDistance=10, blockSize=10)
                prev_pts = cv2.goodFeaturesToTrack(pre_image, mask=None, **feature_params)
                lk_params = dict(winSize=(20, 20), maxLevel=3,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
                cur_pts, status, err = cv2.calcOpticalFlowPyrLK(pre_image, cur_image, prev_pts, None, **lk_params)
                good_prev_pts = prev_pts[status == 1]
                good_cur_pts = cur_pts[status == 1]

                # 计算单应性矩阵
                H, mask = cv2.findHomography(good_prev_pts, good_cur_pts, cv2.RANSAC, 5.0)
            if self.check_SIFT.isChecked():
                sift = cv2.SIFT_create()
                keypoints1, descriptors1 = sift.detectAndCompute(pre_image, None)
                keypoints2, descriptors2 = sift.detectAndCompute(cur_image, None)
                # 创建FLANN匹配器
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                # 使用KNN匹配特征点
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)
                # 应用比率测试来选择良好的匹配点
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                # 使用RANSAC算法计算单应性矩阵H
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            json_path = os.path.splitext(pre_image_name)[0] + '.json'
            polygon = []
            with open(json_path, 'r') as f:
                data = json.load(f)
            for shape in data['shapes']:
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
                    points0 = cv2.perspectiveTransform(src_point0.reshape(-1, 1, 2), H).tolist()
                    points1 = cv2.perspectiveTransform(src_point1.reshape(-1, 1, 2), H).tolist()
                    points0 = [int(points0[0][0][0]), int(points0[0][0][1])]
                    points1 = [int(points1[0][0][0]), int(points1[0][0][1])]
                if 0 <= points0[0] < image_width and 0 <= points0[1] < image_height and 0 <= points1[
                    0] < image_width and 0 <= points1[1] < image_height:
                    # 确保坐标在图像范围内
                    points0[0] = max(0, min(points0[0], image_width - 1))
                    points0[1] = max(0, min(points0[1], image_height - 1))
                    points1[0] = max(0, min(points1[0], image_width - 1))
                    points1[1] = max(0, min(points1[1], image_height - 1))
                    polygon.append([class_id, points0, points1])
                # polygon.append([class_id, points0, points1])
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.keep_current_label.isChecked():
                if self.labelFile.shapes is not None:
                    current_label = self.labelFile.shapes
                    polygon1 = polygon.copy()
                    for box1 in current_label:
                        ious = [calculate_iou([box1['points'][0], box1['points'][1]], [box2[1], box2[2]]) for box2 in
                                polygon1]
                        if all(iou < self.homography_thres for iou in ious):
                            polygon.append([box1['label'], box1['points'][0], box1['points'][1]])
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.pre_homography_shape = self.labelFile.shapes
            self.save_AI_Labels(label_file, polygon)
            self.loadFile(self.filename)

    def object_detection(self, _value=False):

        def nms(pred, conf_thres, iou_thres):
            conf = pred[..., 4] > conf_thres
            box = pred[conf == True]
            cls_conf = box[..., 5:]
            cls = []
            for i in range(len(cls_conf)):
                cls.append(int(np.argmax(cls_conf[i])))
            total_cls = list(set(cls))
            output_box = []
            for i in range(len(total_cls)):
                clss = total_cls[i]
                cls_box = []
                for j in range(len(cls)):
                    if cls[j] == clss:
                        box[j][5] = clss
                        cls_box.append(box[j][:6])
                cls_box = np.array(cls_box)
                box_conf = cls_box[..., 4]
                box_conf_sort = np.argsort(box_conf)
                max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
                output_box.append(max_conf_box)
                cls_box = np.delete(cls_box, 0, 0)
                while len(cls_box) > 0:
                    max_conf_box = output_box[len(output_box) - 1]
                    del_index = []
                    for j in range(len(cls_box)):
                        current_box = cls_box[j]
                        interArea = getInter(max_conf_box, current_box)
                        iou = getIou(max_conf_box, current_box, interArea)
                        if iou > iou_thres:
                            del_index.append(j)
                    cls_box = np.delete(cls_box, del_index, 0)
                    if len(cls_box) > 0:
                        output_box.append(cls_box[0])
                        cls_box = np.delete(cls_box, 0, 0)
            return output_box

        def obb_nms(boxes, iou_thres):
            remove_flags = [False] * len(boxes)
            keep_boxes = []
            for i, ibox in enumerate(boxes):
                if remove_flags[i]:
                    continue
                keep_boxes.append(ibox)
                for j in range(i + 1, len(boxes)):
                    if remove_flags[j]:
                        continue
                    jbox = boxes[j]
                    if (ibox[6] != jbox[6]):
                        continue
                    if probIou(ibox, jbox) > iou_thres:
                        remove_flags[j] = True
            return keep_boxes

        def xywhr2xyxyxyxy(center):
            cos, sin = (np.cos, np.sin)
            ctr = center[..., :2]
            w, h, angle = (center[..., i: i + 1] for i in range(2, 5))
            cos_value, sin_value = cos(angle), sin(angle)
            vec1 = [w / 2 * cos_value, w / 2 * sin_value]
            vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
            vec1 = np.concatenate(vec1, axis=-1)
            vec2 = np.concatenate(vec2, axis=-1)
            pt1 = ctr + vec1 + vec2
            pt2 = ctr + vec1 - vec2
            pt3 = ctr - vec1 - vec2
            pt4 = ctr - vec1 + vec2

            # return [pt1, pt2, pt3, pt4]
            return np.stack([pt1, pt2, pt3, pt4], axis=-2)

        def getIou(box1, box2, inter_area):
            box1_area = box1[2] * box1[3]
            box2_area = box2[2] * box2[3]
            union = box1_area + box2_area - inter_area
            iou = inter_area / union
            return iou

        def covariance_matrix(obb):
            # Extract elements
            w, h, r = obb[2:5]
            a = (w ** 2) / 12
            b = (h ** 2) / 12

            # Calculate cosine and sine using NumPy
            cos_r = np.cos(r)
            sin_r = np.sin(r)

            # Calculate covariance matrix elements
            a_val = a * cos_r ** 2 + b * sin_r ** 2
            b_val = a * sin_r ** 2 + b * cos_r ** 2
            c_val = (a - b) * sin_r * cos_r

            return a_val, b_val, c_val

        def probIou(obb1, obb2, eps=1e-7):
            a1, b1, c1 = covariance_matrix(obb1)
            a2, b2, c2 = covariance_matrix(obb2)
            x1, y1 = obb1[:2]
            x2, y2 = obb2[:2]
            # Calculate terms for Bhattacharyya distance
            t1 = ((a1 + a2) * ((y1 - y2) ** 2) + (b1 + b2) * ((x1 - x2) ** 2)) / \
                 ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)

            t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / \
                 ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
            t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / \
                        (4 * np.sqrt(a1 * b1 - c1 ** 2) * np.sqrt(a2 * b2 - c2 ** 2) + eps) + eps)
            # Bhattacharyya distance calculation
            bd = 0.25 * t1 + 0.5 * t2 + 0.5 * t3
            hd = np.sqrt(1.0 - np.exp(-np.clip(bd, eps, 100.0)) + eps)

            # Extract the x, y coordinates of both
            return 1 - hd

        def getInter(box1, box2):
            box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                                 box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
            box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
                                                 box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
            if box1_x1 > box2_x2 or box1_x2 < box2_x1:
                return 0
            if box1_y1 > box2_y2 or box1_y2 < box2_y1:
                return 0
            x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
            x_list = np.sort(x_list)
            x_inter = x_list[2] - x_list[1]
            y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
            y_list = np.sort(y_list)
            y_inter = y_list[2] - y_list[1]
            inter = x_inter * y_inter
            return inter

        def resize_with_padding(image, target_size, fill_color=114):
            original_size = image.shape[:2]  # 原始尺寸 (height, width)
            ratio = min(target_size[1] / original_size[1], target_size[0] / original_size[0])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))  # new_size (height, width)
            resized_image = cv2.resize(image, (new_size[1], new_size[0]))
            new_image = np.full((target_size[0], target_size[1], 3), fill_color, dtype=np.uint8)
            new_image[0:new_size[0], 0:new_size[1]] = resized_image
            return new_image, ratio

        if self.check_all_infer.isChecked():
            start_index = self.imageList.index(self.filename)  # 获取当前文件的位置
            for i in range(start_index, len(self.imageList)):  # 打开下一个图像
                image = labelme.utils.img_qt_to_arr(self.image)
                input_tensors = self.net.get_inputs()
                self.label_list = ast.literal_eval(self.net.get_modelmeta().custom_metadata_map['names'])
                task = self.net.get_modelmeta().custom_metadata_map['task']
                for input_tensor in input_tensors:
                    input_info = {
                        "name": input_tensor.name,
                        "type": input_tensor.type,
                        "shape": input_tensor.shape,
                    }
                img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                image_height, image_width = img.shape[:2]
                img, ratio = resize_with_padding(img, (input_info["shape"][2], input_info["shape"][3]))
                img = img / 255
                img = img.astype(np.float32)
                blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
                inputs = {self.net.get_inputs()[0].name: blob}
                pred = self.net.run(None, inputs)[0]
                if task == "detect":
                    polygon = []
                    pred = np.squeeze(pred)
                    pred = np.transpose(pred, (1, 0))
                    pred_class = pred[..., 4:]
                    pred_conf = np.max(pred_class, axis=-1)
                    pred = np.insert(pred, 4, pred_conf, axis=-1)
                    result = nms(pred, self.object_conf_thres, 0.45)

                    for detection in result:
                        x_center, y_center, w, h, score, class_id = detection
                        # ymin = ymin + 2
                        # xmin = xmin + 1

                        detect = [int((x_center - w / 2) / ratio), int((y_center - h / 2) / ratio),
                                  int((x_center + w / 2) / ratio), int((y_center + h / 2) / ratio)]

                        points0 = [detect[0], detect[1]]
                        points1 = [detect[2], detect[3]]
                        points0[0] = max(0, min(points0[0], image_width - 1))
                        points0[1] = max(0, min(points0[1], image_height - 1))
                        points1[0] = max(0, min(points1[0], image_width - 1))
                        points1[1] = max(0, min(points1[1], image_height - 1))
                        polygon.append([class_id, points0, points1])

                    label_file = osp.splitext(self.imagePath)[0] + ".json"
                    if self.output_dir:
                        label_file_without_path = osp.basename(label_file)
                        label_file = osp.join(self.output_dir, label_file_without_path)
                    self.save_AI_Labels(label_file, polygon)
                    self.openNextImg()
                elif task == "obb":
                    polygon = []
                    pred = np.transpose(pred, (0, 2, 1))
                    conf_thres = self.object_conf_thres
                    iou_thres = 0.45
                    boxes = []
                    for item in pred[0]:
                        cx, cy, w, h = item[:4]
                        angle = item[-1]
                        label = item[4:-1].argmax()
                        confidence = item[4 + label]
                        if confidence < conf_thres:
                            continue
                        boxes.append([cx, cy, w, h, angle, confidence, label])
                    boxes = np.array(boxes)
                    boxes = sorted(boxes.tolist(), key=lambda x: x[5], reverse=True)
                    boxes = obb_nms(np.array(boxes), iou_thres)
                    confs = [box[5] for box in boxes]
                    classes = [int(box[6]) for box in boxes]
                    if len(boxes) != 0:
                        xyxy_boxes = xywhr2xyxyxyxy(np.array(boxes)[..., :5])
                    else:
                        xyxy_boxes = []
                    for i, box in enumerate(xyxy_boxes):
                        box = box / ratio
                        box[:, 0] = np.clip(box[:, 0], 0, image_width - 1)  # 限制 x 坐标范围
                        box[:, 1] = np.clip(box[:, 1], 0, image_height - 1)  # 限制 y 坐标范围
                        polygon.append([classes[i], box[0], box[1], box[2], box[3], boxes[i][4]])
                    label_file = osp.splitext(self.imagePath)[0] + ".json"
                    if self.output_dir:
                        label_file_without_path = osp.basename(label_file)
                        label_file = osp.join(self.output_dir, label_file_without_path)
                    self.save_AI_obb_labels(label_file, polygon)
                    self.openNextImg()
        else:
            image = labelme.utils.img_qt_to_arr(self.image)
            input_tensors = self.net.get_inputs()
            self.label_list = ast.literal_eval(self.net.get_modelmeta().custom_metadata_map['names'])  # 该 API 会返回列表
            task = self.net.get_modelmeta().custom_metadata_map['task']
            for input_tensor in input_tensors:  # 因为可能有多个输入，所以为列表
                input_info = {
                    "name": input_tensor.name,
                    "type": input_tensor.type,
                    "shape": input_tensor.shape,
                }
            img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            image_show = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            image_height, image_width = img.shape[:2]
            img, ratio = resize_with_padding(img, (input_info["shape"][2], input_info["shape"][3]))
            img_r = img.copy()
            img = img / 255
            img = img.astype(np.float32)
            blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            inputs = {self.net.get_inputs()[0].name: blob}
            pred = self.net.run(None, inputs)[0]
            if task == "obb":
                polygon = []
                pred = np.transpose(pred, (0, 2, 1))
                conf_thres = self.object_conf_thres
                iou_thres = 0.45
                boxes = []
                for item in pred[0]:
                    cx, cy, w, h = item[:4]
                    angle = item[-1]
                    label = item[4:-1].argmax()
                    confidence = item[4 + label]
                    if confidence < conf_thres:
                        continue
                    boxes.append([cx, cy, w, h, angle, confidence, label])
                boxes = np.array(boxes)
                boxes = sorted(boxes.tolist(), key=lambda x: x[5], reverse=True)
                boxes = obb_nms(np.array(boxes), iou_thres)
                confs = [box[5] for box in boxes]
                classes = [int(box[6]) for box in boxes]
                # if len(boxes) != 0:
                #     xyxy_boxes = xywhr2xyxyxyxy(np.array(boxes)[..., :5])
                # else:
                #     xyxy_boxes = []
                xyxy_boxes = []
                if len(boxes) != 0:
                    for i, box in enumerate(boxes):
                        cx, cy, w, h = box[0], box[1], box[2] + 2, box[3] + 2
                        if min(box[2], box[3]) / max(box[2], box[3]) > 0.8:
                            radius = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                            # 创建一个圆形掩码
                            mask = np.zeros_like(img, dtype=np.uint8)
                            cv2.circle(mask, (int(cx), int(cy)), int(radius), (255, 255, 255), -1)
                            # 提取圆形区域
                            circular_region = cv2.bitwise_and(img_r, mask)
                            # 灰度转换和边缘检测
                            gray = cv2.cvtColor(circular_region, cv2.COLOR_BGR2GRAY)
                            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
                            lines = lsd.detect(gray)[0]  # 输入灰度图像
                            # 筛选与长或宽最接近的直线
                            best_line = None
                            min_diff = float('inf')
                            if lines is not None:
                                for idx, line in enumerate(lines):
                                    x1, y1, x2, y2 = map(int, line[0])
                                    cv2.line(img_r, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色线，粗细2
                                    center_dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / np.sqrt(
                                        (y2 - y1) ** 2 + (x2 - x1) ** 2)
                                    # expected_dist = (box[2] + box[3]) / 4
                                    if abs(center_dist - box[2] / 2) > 5 and abs(center_dist - box[3] / 2) > 5:
                                        continue
                                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    diff = abs(box[2] + box[3] - 2 * length)

                                    if diff < min_diff:
                                        min_diff = diff
                                        best_line = (x1, y1, x2, y2)
                                if best_line is not None:
                                    x1, y1, x2, y2 = best_line
                                    cv2.line(img_r, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 绿色线，粗细2
                                    dx, dy = x2 - x1, y2 - y1
                                    A = dx ** 2 + dy ** 2
                                    B = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
                                    C = (x1 - cx) ** 2 + (y1 - cy) ** 2 - radius ** 2
                                    D = B ** 2 - 4 * A * C
                                    t1 = (-B + np.sqrt(D)) / (2 * A)
                                    t2 = (-B - np.sqrt(D)) / (2 * A)
                                    x1, y1, x2, y2 = x1 + t1 * dx, y1 + t1 * dy, x1 + t2 * dx, y1 + t2 * dy
                                    sym_x1, sym_y1 = 2 * cx - x1, 2 * cy - y1
                                    sym_x2, sym_y2 = 2 * cx - x2, 2 * cy - y2
                                    points = [np.array((x1, y1)), np.array((x2, y2)), np.array((sym_x1, sym_y1)),
                                              np.array((sym_x2, sym_y2))]
                                    xyxy_boxes.append(points)
                                else:
                                    xyxy_boxes.append(xywhr2xyxyxyxy(np.array(box)[:5]))
                            else:
                                xyxy_boxes.append(xywhr2xyxyxyxy(np.array(box)[:5]))
                        else:
                            xyxy_boxes.append(xywhr2xyxyxyxy(np.array(box)[:5]))
                for i, box in enumerate(xyxy_boxes):
                    box = np.array(box) / ratio
                    box[:, 0] = np.clip(box[:, 0], 0, image_width - 1)  # 限制 x 坐标范围
                    box[:, 1] = np.clip(box[:, 1], 0, image_height - 1)  # 限制 y 坐标范围
                    polygon.append([classes[i], box[0], box[1], box[2], box[3], boxes[i][4]])

                label_file = osp.splitext(self.imagePath)[0] + ".json"
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(self.output_dir, label_file_without_path)
                output_img_path = osp.splitext(self.imagePath)[0] + "_labeled.png"
                if self.output_dir:
                    output_img_path_without_path = osp.basename(output_img_path)
                    output_img_path = osp.join(self.output_dir, output_img_path_without_path)

                # 绘制多边形
                for poly in polygon:
                    cls, p1, p2, p3, p4, angle = poly
                    points = np.array([p1, p2, p3, p4], dtype=np.int32).reshape((-1, 1, 2))
                    color = (255, 0, 0)  # 默认绿色
                    cv2.polylines(image_show, [points], isClosed=True, color=color, thickness=2)
                    for point in points:
                        x, y = point[0]  # 提取坐标
                        text = f"({int(x)}, {int(y)})"
                        font_scale = 0.4  # 字体大小
                        thickness = 1  # 字体粗细
                        text_color = (255, 0, 0)  # 白色字体
                        cv2.putText(image_show, text, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    text_color, thickness)
                # 保存标注后的图像
                image_show = cv2.cvtColor(image_show, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(output_img_path, image_show)
                print(f"Labeled image saved to: {output_img_path}")

                self.save_AI_obb_labels(label_file, polygon)
                self.loadFile(self.filename)
            elif task == "detect":
                polygon = []
                pred = np.squeeze(pred)
                pred = np.transpose(pred, (1, 0))
                pred_class = pred[..., 4:]
                pred_conf = np.max(pred_class, axis=-1)
                pred = np.insert(pred, 4, pred_conf, axis=-1)
                result = nms(pred, self.object_conf_thres, 0.45)
                for detection in result:
                    x_center, y_center, w, h, score, class_id = detection
                    detect = [int((x_center - w / 2) / ratio), int((y_center - h / 2) / ratio),
                              int((x_center + w / 2) / ratio), int((y_center + h / 2) / ratio)]

                    points0 = [detect[0], detect[1]]
                    points1 = [detect[2], detect[3]]
                    points0[0] = max(0, min(points0[0], image_width - 1))
                    points0[1] = max(0, min(points0[1], image_height - 1))
                    points1[0] = max(0, min(points1[0], image_width - 1))
                    points1[1] = max(0, min(points1[1], image_height - 1))
                    polygon.append([class_id, points0, points1])
                label_file = osp.splitext(self.imagePath)[0] + ".json"
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(self.output_dir, label_file_without_path)
                self.save_AI_Labels(label_file, polygon)
                self.loadFile(self.filename)

    def Run_Text2Label(self, _value=False):
        if self.check_all_infer.isChecked():
            start_index = self.imageList.index(self.filename)  # 获取当前文件的位置

            for i in range(start_index, len(self.imageList)):
                image = labelme.utils.img_qt_to_arr(self.image)
                if image.shape[-1] == 1:
                    image = np.squeeze(image, axis=-1)
                # image = Image.fromarray(image)
                boxes_filt, pred_phrases = self.text2label_model.detect(image, self.Text2Label_Text.text())

                size = image.shape
                pred_dict = {
                    "boxes": boxes_filt,
                    "size": [size[0], size[1]],  # H,W
                    "labels": pred_phrases,
                }
                H, W = pred_dict["size"]
                boxes = pred_dict["boxes"]
                labels = pred_dict["labels"]
                polygon = []
                for box, label in zip(boxes, labels):
                    # from 0..1 to 0..W, 0..H
                    box = box * np.array([W, H, W, H])
                    # from xywh to xyxy
                    box[:2] -= box[2:] / 2
                    box[2:] += box[:2]
                    x0, y0, x1, y1 = box
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    points0 = [x0, y0]
                    points1 = [x1, y1]
                    polygon.append([label, points0, points1])
                label_file = osp.splitext(self.imagePath)[0] + ".json"
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(self.output_dir, label_file_without_path)
                self.save_AI_Labels(label_file, polygon)
                self.loadFile(self.filename)
                self.openNextImg()
        else:
            image = labelme.utils.img_qt_to_arr(self.image)
            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)
            # image = Image.fromarray(image)
            if self.text2label_model:
                boxes_filt, pred_phrases = self.text2label_model.detect(image, self.Text2Label_Text.text())
            else:
                self.errorMessage(
                    "No onnx model",
                    "You must choose an onnx model.",
                )
                return

            size = image.shape
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[0], size[1]],  # H,W
                "labels": pred_phrases,
            }
            H, W = pred_dict["size"]
            boxes = pred_dict["boxes"]
            labels = pred_dict["labels"]
            polygon = []
            for box, label in zip(boxes, labels):
                # from 0..1 to 0..W, 0..H
                box = box * np.array([W, H, W, H])
                # from xywh to xyxy
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]
                x0, y0, x1, y1 = box
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                points0 = [x0, y0]
                points1 = [x1, y1]
                polygon.append([label, points0, points1])
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.save_AI_Labels(label_file, polygon)
            self.loadFile(self.filename)

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

    def Dataset(self, _value=False):
        dialog = DatasetDialog()
        dialog.exec_()

    def Yolovis(self, _value=False):
        dialog = Yolo_Vis_Dialog()
        dialog.exec_()

    def Video_slice(self, _value=False):
        dialog = Video_slice_Dialog()
        dialog.exec_()

    def Data_augmentation(self, _value=False):
        dialog = Data_augmentation_Dialog()
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

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
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

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
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
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
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
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        # msg = self.tr(
        #     "You are about to permanently delete this label file, " "proceed anyway?"
        # )
        # answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        # if answer != mb.Yes:
        #     return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)
            self.labelList.clear()
            # self.filename = None
            self.imagePath = None
            self.imageData = None
            self.labelFile = None
            self.otherData = None
            self.tmpimageData = None
            self.rotatevalue = None
            self.imagePass = None
            self.QCResult.setText(None)
            self.canvas.resetState()
            # self.resetState()

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
        self.remLabels(self.canvas.deleteSelected())
        self.setDirty()
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
        self.canvas.endMove(copy=False)
        self.setDirty()

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
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

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
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)
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
