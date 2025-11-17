from __future__ import annotations
from typing import Optional
import pandas as pd

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex, Signal
from PySide6.QtGui import QUndoStack, QUndoCommand, QBrush, QColor


class _EditCellCommand(QUndoCommand):
    """Một lệnh chỉnh sửa 1 ô cho QUndoStack (Undo/Redo)."""
    def __init__(self, model: "PandasEditModel", row: int, col: int, old, new):
        super().__init__(f"Edit ({row},{col})")
        self.model = model
        self.row = row
        self.col = col
        self.old = old
        self.new = new

    def undo(self):
        self.model._set_value_silent(self.row, self.col, self.old)

    def redo(self):
        self.model._set_value_silent(self.row, self.col, self.new)


class PandasEditModel(QAbstractTableModel):
    """TableModel cho pandas.DataFrame hỗ trợ:
       - Edit tại chỗ (giống Excel)
       - Undo/Redo với QUndoStack
       - Highlight các dòng theo cột 'page'
    """
    # phát tín hiệu khi dataframe đổi (giúp UI khác nếu cần)
    frameChanged = Signal()

    def __init__(self, df: Optional[pd.DataFrame] = None, undo_stack: Optional[QUndoStack] = None, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame = df if df is not None else pd.DataFrame()
        self._undo: QUndoStack = undo_stack if undo_stack is not None else QUndoStack(self)
        # trang hiện tại cần highlight (1-based), None = không highlight
        self._highlight_page: Optional[int] = None

    # ---------- public helpers ----------
    def setUndoStack(self, stack: QUndoStack):
        self._undo = stack

    def undoStack(self) -> QUndoStack:
        return self._undo

    def setDataFrame(self, df: Optional[pd.DataFrame]):
        self.beginResetModel()
        self._df = df.copy() if df is not None else pd.DataFrame()
        self.endResetModel()
        # reset lại undo khi nạp khung mới
        if self._undo is not None:
            self._undo.clear()
        self.frameChanged.emit()
        # khi nạp dataframe mới, giữ nguyên trang highlight nếu có
        if self._highlight_page is not None:
            self._emit_background_changed()

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def dataframe_copy(self) -> pd.DataFrame:
        return self._df.copy()

    # ---------- highlight theo page ----------
    def setHighlightPage(self, page: Optional[int]):
        """
        Đặt trang cần highlight (1-based). None = tắt highlight.
        """
        self._highlight_page = page
        self._emit_background_changed()

    def _emit_background_changed(self):
        if self._df is None or self._df.empty:
            return
        if self.rowCount() <= 0 or self.columnCount() <= 0:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
        # báo cho view rằng màu nền có thể đã thay đổi
        self.dataChanged.emit(top_left, bottom_right, [Qt.BackgroundRole])

    # ---------- model basics ----------
    def rowCount(self, parent=QModelIndex()):
        return 0 if self._df is None else len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if self._df is None else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return None

        # Highlight background cho các dòng có cùng page
        if role == Qt.BackgroundRole and self._highlight_page is not None:
            try:
                if "page" in self._df.columns:
                    row = index.row()
                    val = self._df.iloc[row][self._df.columns.get_loc("page")]
                    # so sánh dạng int để tránh khác kiểu (str/float)
                    if pd.notna(val) and int(val) == int(self._highlight_page):
                        return QBrush(QColor(230, 230, 230))  # xám nhạt
            except Exception:
                pass

        if role in (Qt.DisplayRole, Qt.EditRole):
            val = self._df.iloc[index.row(), index.column()]
            # Hiển thị chuỗi rỗng cho None/NaN
            if val is None:
                return ""
            # Tránh hiển thị 'nan'
            try:
                import math
                if isinstance(val, float) and (math.isnan(val)):
                    return ""
            except Exception:
                pass
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if self._df is None or role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return ""
        else:
            try:
                return str(self._df.index[section])
            except Exception:
                return str(section)

    # ---------- editing ----------
    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        # Cho phép select + edit mọi ô
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole or self._df is None:
            return False

        r, c = index.row(), index.column()
        old = self._df.iloc[r, c]
        new = value

        # Không làm gì nếu không đổi
        if str(old) == str(new):
            return False

        # Đẩy vào undo stack (redo sẽ gọi _set_value_silent)
        if self._undo is not None:
            cmd = _EditCellCommand(self, r, c, old, new)
            self._undo.push(cmd)
            return True

        # fallback nếu không có undo stack
        self._set_value_silent(r, c, new)
        return True

    # phương thức nội bộ: set giá trị và phát dataChanged nhưng KHÔNG push vào stack
    def _set_value_silent(self, row: int, col: int, val):
        if self._df is None:
            return
        # Thử cast đơn giản: nếu cột đang là số và người dùng nhập số -> ép kiểu
        try:
            cur_col = self._df.columns[col]
            if pd.api.types.is_numeric_dtype(self._df[cur_col]):
                # cho phép chuỗi có dấu phẩy -> chấm
                if isinstance(val, str):
                    v = val.strip().replace(",", "")
                    self._df.iat[row, col] = float(v) if v != "" else None
                else:
                    self._df.iat[row, col] = float(val) if val is not None and val != "" else None
            else:
                self._df.iat[row, col] = val
        except Exception:
            # nếu cast lỗi thì lưu nguyên văn
            self._df.iat[row, col] = val

        ix = self.index(row, col)
        self.dataChanged.emit(ix, ix, [Qt.DisplayRole, Qt.EditRole])
        self.frameChanged.emit()
