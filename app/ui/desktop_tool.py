from __future__ import annotations
from pathlib import Path
import pandas as pd

from PySide6.QtCore import Qt, QItemSelectionModel
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QSplitter, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTabWidget,
    QTableView, QStatusBar, QLineEdit, QComboBox, QCheckBox
)

from .pdf_viewer import PdfViewer
from .qt_models import PandasEditModel
from .extract_preview import extract_preview
from .save_pipeline import persist_preview
from app.ddl_loader import apply_schema
from app.seed_line_items import run as seed_line_items


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Financial PDF Loader — Preview & Import")
        self.resize(1320, 820)

        # Left: PDF
        self.viewer = PdfViewer()

        # Right: controls + tabs
        self.btn_pick = QPushButton("Chọn PDF…")
        self.le_path = QLineEdit()
        self.le_path.setReadOnly(True)
        self.btn_preview = QPushButton("Xử lý (Preview)")
        self.btn_save = QPushButton("Lưu vào DB")
        self.btn_save.setEnabled(False)

        # OCR engine chọn Tesseract / PaddleOCR
        self.cmb_ocr = QComboBox()
        self.cmb_ocr.addItems(["Tesseract", "PaddleOCR"])
        self.cmb_ocr.setCurrentIndex(0)  # mặc định Tesseract
        self.cmb_ocr.setToolTip("Chọn engine OCR để trích dữ liệu từ PDF")

        # Bật/tắt sync trang PDF <-> bảng dữ liệu
        self.chk_sync = QCheckBox("Đồng bộ trang → bảng")
        self.chk_sync.setChecked(True)
        self.chk_sync.setToolTip("Nếu bật, khi chuyển trang PDF sẽ nhảy đến và tô xám các dòng có cùng số trang.")

        # Undo/Redo cho người dùng
        self.btn_undo_f = QPushButton("Hoàn tác (Facts)")
        self.btn_redo_f = QPushButton("Làm lại (Facts)")
        self.btn_undo_n = QPushButton("Hoàn tác (Notes)")
        self.btn_redo_n = QPushButton("Làm lại (Notes)")
        self.btn_undo_f.setEnabled(False)
        self.btn_redo_f.setEnabled(False)
        self.btn_undo_n.setEnabled(False)
        self.btn_redo_n.setEnabled(False)

        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_pick)
        ctrl.addWidget(self.le_path, 1)
        ctrl.addWidget(QLabel("OCR:"))
        ctrl.addWidget(self.cmb_ocr)
        ctrl.addWidget(self.chk_sync)
        ctrl.addWidget(self.btn_preview)
        ctrl.addWidget(self.btn_save)

        # Thanh Undo/Redo
        undo_bar = QHBoxLayout()
        undo_bar.addWidget(QLabel("Chỉnh sửa & Hoàn tác:"))
        undo_bar.addWidget(self.btn_undo_f)
        undo_bar.addWidget(self.btn_redo_f)
        undo_bar.addSpacing(12)
        undo_bar.addWidget(self.btn_undo_n)
        undo_bar.addWidget(self.btn_redo_n)
        undo_bar.addStretch(1)

        self.tabs = QTabWidget()

        # Tab 1: Facts (BS/IS/CF)
        self.tbl_facts = QTableView()
        self.model_facts = PandasEditModel(pd.DataFrame())
        self.tbl_facts.setModel(self.model_facts)
        self.tbl_facts.setSortingEnabled(True)
        # bật chế độ chỉnh sửa khi double-click/enter
        self.tbl_facts.setEditTriggers(QTableView.DoubleClicked | QTableView.SelectedClicked | QTableView.EditKeyPressed)

        # Tab 2: Notes (unified)
        self.tbl_notes = QTableView()
        self.model_notes = PandasEditModel(pd.DataFrame())
        self.tbl_notes.setModel(self.model_notes)
        self.tbl_notes.setSortingEnabled(True)
        self.tbl_notes.setEditTriggers(QTableView.DoubleClicked | QTableView.SelectedClicked | QTableView.EditKeyPressed)

        self.tabs.addTab(self.tbl_facts, "Bảng BS/IS/CF (Facts)")
        self.tabs.addTab(self.tbl_notes, "Thuyết minh (Notes)")

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.addLayout(ctrl)
        rlay.addLayout(undo_bar)
        rlay.addWidget(self.tabs, 1)

        splitter = QSplitter()
        splitter.addWidget(self.viewer)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar(self))

        # state
        self._pdf_path: Path | None = None

        # signals
        self.btn_pick.clicked.connect(self._pick_pdf)
        self.btn_preview.clicked.connect(self._do_preview)
        self.btn_save.clicked.connect(self._do_save)

        # sync toggle
        self.chk_sync.toggled.connect(self._on_sync_toggled)

        # lắng nghe khi PDF đổi trang
        self.viewer.pageChanged.connect(self._on_pdf_page_changed)

        # Undo/Redo buttons
        self.btn_undo_f.clicked.connect(lambda: self.model_facts.undoStack().undo())
        self.btn_redo_f.clicked.connect(lambda: self.model_facts.undoStack().redo())
        self.btn_undo_n.clicked.connect(lambda: self.model_notes.undoStack().undo())
        self.btn_redo_n.clicked.connect(lambda: self.model_notes.undoStack().redo())

        # enable buttons khi có thể undo/redo
        self.model_facts.undoStack().canUndoChanged.connect(lambda _: self._refresh_undo_buttons())
        self.model_facts.undoStack().canRedoChanged.connect(lambda _: self._refresh_undo_buttons())
        self.model_notes.undoStack().canUndoChanged.connect(lambda _: self._refresh_undo_buttons())
        self.model_notes.undoStack().canRedoChanged.connect(lambda _: self._refresh_undo_buttons())

        # shortcuts toàn cửa sổ: Ctrl+Z / Ctrl+Y
        sc_undo = QShortcut(QKeySequence.Undo, self)
        sc_redo = QShortcut(QKeySequence.Redo, self)
        sc_undo.activated.connect(self._global_undo)
        sc_redo.activated.connect(self._global_redo)

        # ensure schema is ready (idempotent, nhanh)
        try:
            apply_schema()
            seed_line_items()
        except Exception as e:
            QMessageBox.warning(self, "Schema", f"Lỗi khởi tạo schema: {e}")

    # ---------- helpers ----------
    def _refresh_undo_buttons(self):
        self.btn_undo_f.setEnabled(self.model_facts.undoStack().canUndo())
        self.btn_redo_f.setEnabled(self.model_facts.undoStack().canRedo())
        self.btn_undo_n.setEnabled(self.model_notes.undoStack().canUndo())
        self.btn_redo_n.setEnabled(self.model_notes.undoStack().canRedo())

    def _current_ocr_engine(self) -> str:
        """
        Lấy lựa chọn OCR hiện tại từ combo.
        Trả về 'tesseract' hoặc 'paddle' đúng với load.py / ocr_processor.
        """
        text = (self.cmb_ocr.currentText() or "").lower()
        if "paddle" in text:
            return "paddle"
        return "tesseract"

    def _global_undo(self):
        # Ưu tiên tab đang mở
        if self.tabs.currentIndex() == 0:
            self.model_facts.undoStack().undo()
        else:
            self.model_notes.undoStack().undo()

    def _global_redo(self):
        if self.tabs.currentIndex() == 0:
            self.model_facts.undoStack().redo()
        else:
            self.model_notes.undoStack().redo()

    # ---------- sync page <-> tables ----------
    def _on_sync_toggled(self, checked: bool):
        if not checked:
            # tắt highlight + selection
            self.model_facts.setHighlightPage(None)
            self.model_notes.setHighlightPage(None)
            if self.tbl_facts.selectionModel():
                self.tbl_facts.selectionModel().clearSelection()
            if self.tbl_notes.selectionModel():
                self.tbl_notes.selectionModel().clearSelection()
        else:
            # bật lại thì sync luôn với trang hiện tại của PDF (nếu có)
            page = self.viewer.current_page()
            if page > 0:
                self._highlight_page_in_tables(page)

    def _on_pdf_page_changed(self, page: int):
        """
        Được gọi mỗi khi người dùng chuyển trang PDF (page là 1-based).
        """
        if not self.chk_sync.isChecked():
            return
        self._highlight_page_in_tables(page)

    def _highlight_page_in_tables(self, page: int):
        # Facts
        self._apply_page_highlight_to_table(self.tbl_facts, self.model_facts, page)
        # Notes
        self._apply_page_highlight_to_table(self.tbl_notes, self.model_notes, page)

    def _apply_page_highlight_to_table(self, table: QTableView, model: PandasEditModel, page: int):
        df = model.dataframe()
        if df is None or df.empty or "page" not in df.columns:
            model.setHighlightPage(None)
            return

        # đặt highlight cho model
        model.setHighlightPage(page)

        # tìm các row có page == page
        try:
            pages = df["page"].tolist()
        except Exception:
            return

        matching_rows = []
        for i, v in enumerate(pages):
            try:
                if v is not None and int(v) == int(page):
                    matching_rows.append(i)
            except Exception:
                continue

        sel = table.selectionModel()
        if sel is None:
            return

        sel.clearSelection()

        if not matching_rows:
            return

        # scroll đến dòng đầu tiên
        first_row = matching_rows[0]
        idx_first = model.index(first_row, 0)
        table.scrollTo(idx_first, QTableView.PositionAtCenter)

        # chọn tất cả các dòng cùng page để user dễ thấy
        for r in matching_rows:
            idx = model.index(r, 0)
            sel.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)

    # ---------- slots ----------
    def _pick_pdf(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Chọn file báo cáo tài chính (PDF)", "", "PDF (*.pdf)")
        if not fn:
            return
        self._pdf_path = Path(fn)
        self.le_path.setText(str(self._pdf_path))
        try:
            self.viewer.load(self._pdf_path)
        except Exception as e:
            QMessageBox.warning(self, "PDF", f"Không mở được PDF: {e}")

    def _do_preview(self):
        if not self._pdf_path:
            QMessageBox.information(self, "Chưa chọn file", "Vui lòng chọn file PDF trước.")
            return
        self.statusBar().showMessage("Đang trích dữ liệu (preview)…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ocr_engine = self._current_ocr_engine()
            self.statusBar().showMessage(f"Đang trích dữ liệu (preview)… [OCR={ocr_engine}]")

            df_facts, df_notes = extract_preview(self._pdf_path, ocr_engine=ocr_engine)

            # đảm bảo các cột cốt lõi có mặt để hiển thị đẹp
            facts_cols = ["page","statement_hint","vas_code","src_label","context_key","context_label","amount","unit_hint","confidence"]
            if df_facts is None:
                df_facts = pd.DataFrame()
            if not df_facts.empty:
                for c in facts_cols:
                    if c not in df_facts.columns:
                        df_facts[c] = None
                df_facts = df_facts[facts_cols]

            # nạp vào model (undo stack sẽ được clear)
            self.model_facts.setDataFrame(df_facts)
            self.model_notes.setDataFrame(df_notes if df_notes is not None else pd.DataFrame())

            self.btn_save.setEnabled(not df_facts.empty or not (df_notes is None or df_notes.empty))
            n_f = 0 if df_facts is None else len(df_facts)
            n_n = 0 if df_notes is None else len(df_notes)
            self.statusBar().showMessage(f"Preview xong: {n_f} ô facts, {n_n} dòng notes")
            self._refresh_undo_buttons()

            # Sau khi có dữ liệu, nếu đang bật sync thì highlight theo trang hiện tại
            if self.chk_sync.isChecked():
                page = self.viewer.current_page()
                if page > 0:
                    self._highlight_page_in_tables(page)

        except Exception as e:
            QMessageBox.critical(self, "Xử lý lỗi", f"Lỗi preview: {e}")
            self.statusBar().clearMessage()
        finally:
            QApplication.restoreOverrideCursor()

    def _do_save(self):
        if self._pdf_path is None:
            return
        df_f_now = self.model_facts.dataframe()
        df_n_now = self.model_notes.dataframe()
        if (df_f_now is None or df_f_now.empty) and (df_n_now is None or df_n_now.empty):
            QMessageBox.information(self, "Không có dữ liệu", "Chưa có dữ liệu để lưu. Hãy chạy Preview trước.")
            return

        self.statusBar().showMessage("Đang lưu vào DB…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Lấy snapshot hiện tại (đã chỉnh sửa) để lưu
            df_f = self.model_facts.dataframe_copy()
            df_n = self.model_notes.dataframe_copy()

            rid = persist_preview(self._pdf_path, df_f, df_n)
            QMessageBox.information(self, "Thành công", f"Đã lưu vào DB cho report_id={rid}.")
            self.statusBar().showMessage(f"Lưu xong cho report_id={rid}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi lưu", f"Không lưu được dữ liệu: {e}")
            self.statusBar().clearMessage()
        finally:
            QApplication.restoreOverrideCursor()


def run_app():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
