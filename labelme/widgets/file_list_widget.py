from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt


class FileListItemDelegate(QtWidgets.QStyledItemDelegate):
    """Draw a file-list row with a check indicator and delete button."""

    deleteClicked = QtCore.Signal(object)
    _button_width = 24
    _margin = 4

    def _buttonRect(self, option):
        return QtCore.QRect(
            option.rect.right() - self._button_width - self._margin + 1,
            option.rect.top() + 2,
            self._button_width,
            max(0, option.rect.height() - 4),
        )

    def paint(self, painter, option, index):
        item_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        item_option.text = ""
        item_option.features &= ~QtWidgets.QStyleOptionViewItem.HasCheckIndicator
        style = option.widget.style() if option.widget else QtWidgets.QApplication.style()
        style.drawControl(
            QtWidgets.QStyle.CE_ItemViewItem,
            item_option,
            painter,
            option.widget,
        )

        check_option = QtWidgets.QStyleOptionButton()
        indicator = style.pixelMetric(
            QtWidgets.QStyle.PM_IndicatorWidth, None, option.widget
        )
        check_option.rect = QtCore.QRect(
            option.rect.left() + self._margin,
            option.rect.center().y() - indicator // 2,
            indicator,
            indicator,
        )
        check_option.state = QtWidgets.QStyle.State_Enabled
        check_option.state |= (
            QtWidgets.QStyle.State_On
            if index.data(Qt.CheckStateRole) == Qt.Checked
            else QtWidgets.QStyle.State_Off
        )
        style.drawPrimitive(
            QtWidgets.QStyle.PE_IndicatorCheckBox,
            check_option,
            painter,
            option.widget,
        )

        button_rect = self._buttonRect(option)
        text_rect = QtCore.QRect(
            check_option.rect.right() + self._margin,
            option.rect.top(),
            max(0, button_rect.left() - check_option.rect.right() - 2 * self._margin),
            option.rect.height(),
        )
        text = option.fontMetrics.elidedText(
            str(index.data(Qt.DisplayRole)), Qt.ElideMiddle, text_rect.width()
        )
        painter.save()
        painter.setPen(
            item_option.palette.color(
                QtGui.QPalette.HighlightedText
                if option.state & QtWidgets.QStyle.State_Selected
                else QtGui.QPalette.Text
            )
        )
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)
        painter.restore()

        button_option = QtWidgets.QStyleOptionButton()
        button_option.rect = button_rect
        button_option.text = "×"
        button_option.state = QtWidgets.QStyle.State_Enabled
        button_option.palette = item_option.palette
        button_option.palette.setColor(
            QtGui.QPalette.ButtonText, QtGui.QColor(220, 35, 35)
        )
        style.drawControl(
            QtWidgets.QStyle.CE_PushButton,
            button_option,
            painter,
            option.widget,
        )

    def sizeHint(self, option, index):
        size = super(FileListItemDelegate, self).sizeHint(option, index)
        size.setHeight(max(size.height(), 24))
        size.setWidth(1)
        return size

    def editorEvent(self, event, model, option, index):
        if (
            event.type() == QtCore.QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
            and self._buttonRect(option).contains(event.pos())
        ):
            widget = option.widget
            item = widget.item(index.row()) if widget is not None else None
            if item is not None:
                self.deleteClicked.emit(item)
            return True
        return super(FileListItemDelegate, self).editorEvent(
            event, model, option, index
        )


class FileListWidget(QtWidgets.QListWidget):
    """List dataset image paths and expose file-list-specific operations."""

    deleteClicked = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(FileListWidget, self).__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setTextElideMode(Qt.ElideMiddle)

        self._item_delegate = FileListItemDelegate(self)
        self._item_delegate.deleteClicked.connect(self.deleteClicked)
        self.setItemDelegate(self._item_delegate)

    @property
    def imageList(self):
        return [self.item(row).text() for row in range(self.count())]

    def addFile(self, filename, checked=False):
        item = QtWidgets.QListWidgetItem(filename)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.addItem(item)
        return item
