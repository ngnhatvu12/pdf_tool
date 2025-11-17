from pathlib import Path
from typing import Optional
import os

from PySide6.QtCore import Qt, QSize, Slot, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider

USE_QTPDF = True
try:
    # Nếu muốn ép fallback: export PDF_VIEW_FORCE_FALLBACK=1
    if os.environ.get("PDF_VIEW_FORCE_FALLBACK", "0") == "1":
        raise ImportError("Force fallback")

    from PySide6.QtPdf import QPdfDocument, QPdfDocumentStatus
    from PySide6.QtPdfWidgets import QPdfView
    USE_QTPDF = True
except Exception:
    USE_QTPDF = False

import fitz  # PyMuPDF (fallback render ảnh)


class PdfViewer(QWidget):
    """
    Widget xem PDF:
      - Ưu tiên QtPdf (nếu sẵn & không bị ép fallback)
      - Fallback PyMuPDF: render ảnh theo trang
    """
    # phát ra số trang (1-based) mỗi khi trang hiện tại đổi
    pageChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path: Optional[Path] = None
        self._page_count = 0
        self._cur_page = 0  # 0-based index

        # ----- NAV UI -----
        self.btn_prev = QPushButton("←")
        self.btn_next = QPushButton("→")
        self.slider = QSlider(Qt.Horizontal)
        self.lbl_page = QLabel("0 / 0")

        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.lbl_page)

        # Container fallback (PyMuPDF)
        self.container = QLabel("Chưa mở PDF")
        self.container.setAlignment(Qt.AlignCenter)

        lay = QVBoxLayout(self)
        lay.addLayout(nav)
        lay.addWidget(self.container, 1)

        self.btn_prev.clicked.connect(self._prev)
        self.btn_next.clicked.connect(self._next)
        self.slider.valueChanged.connect(self._goto)

        # ----- QtPdf objects (nếu có) -----
        self._qtpdf_doc = None
        self._qtpdf_view = None
        self._have_qtpdf = USE_QTPDF

        if self._have_qtpdf:
            try:
                self._qtpdf_doc = QPdfDocument(self)
                self._qtpdf_view = QPdfView(self)
                self._qtpdf_view.setDocument(self._qtpdf_doc)

                # Thay container fallback bằng QPdfView
                lay.removeWidget(self.container)
                self.container.hide()
                self.container.setParent(None)
                lay.addWidget(self._qtpdf_view, 1)

                # Fit vừa khung, chế độ 1 trang
                try:
                    self._qtpdf_view.setZoomMode(QPdfView.FitInView)
                except Exception:
                    pass
                try:
                    self._qtpdf_view.setPageMode(QPdfView.SinglePage)
                except Exception:
                    pass

                # Quan trọng: đổi trang sau khi tài liệu đã load xong
                self._qtpdf_doc.documentStatusChanged.connect(self._on_qtpdf_status_changed)
            except Exception:
                # Nếu QtPdf lỗi → quay fallback
                self._qtpdf_doc = None
                self._qtpdf_view = None
                self._have_qtpdf = False
                # Khôi phục container fallback
                lay.addWidget(self.container, 1)

    # ========= Public API =========
    def load(self, path: Path):
        self._path = Path(path)
        if self._have_qtpdf and self._qtpdf_doc and self._qtpdf_view:
            self._qtpdf_doc.load(str(self._path))
            self._page_count = max(0, int(self._qtpdf_doc.pageCount()))
            self._cur_page = 0
            # KHÔNG đổi trang ngay ở đây; đợi status = Ready
            self._update_nav()
        else:
            # Fallback: tính số trang rồi render trang đầu
            with fitz.open(str(self._path)) as doc:
                self._page_count = len(doc)
            self._cur_page = 0
            self._render_current_with_pymupdf()
            self._update_nav()
            # phát signal trang hiện tại (1-based)
            self.pageChanged.emit(self._cur_page + 1)

    def current_page(self) -> int:
        """Trả về số trang hiện tại (1-based), hoặc 0 nếu chưa có PDF."""
        if self._page_count <= 0:
            return 0
        return int(self._cur_page) + 1

    # ========= QtPdf status handler =========
    @Slot()
    def _on_qtpdf_status_changed(self):
        if not (self._qtpdf_doc and self._qtpdf_view):
            return
        status = self._qtpdf_doc.status()
        # Chỉ khi Ready mới đảm bảo navigator hoạt động
        if status == QPdfDocumentStatus.Ready:
            self._qtpdf_go(self._cur_page)
            self.pageChanged.emit(self._cur_page + 1)

    # ========= NAV helpers =========
    def _update_nav(self):
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self._page_count - 1))
        self.slider.setValue(self._cur_page)
        self.slider.blockSignals(False)
        self.lbl_page.setText(f"{self._cur_page + 1} / {self._page_count}")

    def _qtpdf_go(self, page_index: int):
        """Điều hướng trang cho QtPdf theo API tương thích nhiều bản."""
        if not (self._qtpdf_view and self._qtpdf_doc):
            return
        # Ghim page_index vào [0, page_count-1]
        page_index = max(0, min(page_index, max(0, self._qtpdf_doc.pageCount() - 1)))

        # Một số bản PySide6 dùng pageNavigator(), số khác pageNavigation()
        nav = None
        if hasattr(self._qtpdf_view, "pageNavigator"):
            nav = self._qtpdf_view.pageNavigator()
        elif hasattr(self._qtpdf_view, "pageNavigation"):
            nav = self._qtpdf_view.pageNavigation()

        if nav is not None:
            # Ưu tiên setCurrentPage()
            if hasattr(nav, "setCurrentPage"):
                try:
                    nav.setCurrentPage(int(page_index))
                    return
                except Exception:
                    pass
            # Fallback: một số bản có goToPage()
            if hasattr(nav, "goToPage"):
                try:
                    nav.goToPage(int(page_index))
                    return
                except Exception:
                    pass

        # Cuối cùng: thử API setPage nếu bản đó có
        if hasattr(self._qtpdf_view, "setPage"):
            try:
                self._qtpdf_view.setPage(int(page_index))
                return
            except Exception:
                pass

    # ========= NAV slots =========
    def _prev(self):
        if self._page_count <= 0:
            return
        if self._cur_page > 0:
            self._cur_page -= 1
            if self._have_qtpdf and self._qtpdf_doc and self._qtpdf_view:
                self._qtpdf_go(self._cur_page)
            else:
                self._render_current_with_pymupdf()
            self._update_nav()
            self.pageChanged.emit(self._cur_page + 1)

    def _next(self):
        if self._page_count <= 0:
            return
        if self._cur_page < (self._page_count - 1):
            self._cur_page += 1
            if self._have_qtpdf and self._qtpdf_doc and self._qtpdf_view:
                self._qtpdf_go(self._cur_page)
            else:
                self._render_current_with_pymupdf()
            self._update_nav()
            self.pageChanged.emit(self._cur_page + 1)

    def _goto(self, v: int):
        if self._page_count <= 0:
            return
        self._cur_page = int(v)
        if self._have_qtpdf and self._qtpdf_doc and self._qtpdf_view:
            self._qtpdf_go(self._cur_page)
        else:
            self._render_current_with_pymupdf()
        self._update_nav()
        self.pageChanged.emit(self._cur_page + 1)

    # ========= Render fallback =========
    def _render_current_with_pymupdf(self, dpi=160):
        if not self._path:
            return
        with fitz.open(str(self._path)) as doc:
            pidx = max(0, min(self._cur_page, len(doc) - 1))
            p = doc.load_page(pidx)
            pm = p.get_pixmap(dpi=dpi, alpha=False)
            img = QImage(pm.samples, pm.width, pm.height, pm.stride, QImage.Format_RGB888)
            target_size = self.size() - QSize(24, 24)
            self.container.setPixmap(QPixmap.fromImage(img).scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    # Ảnh fallback tự scale khi đổi kích thước
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not (self._have_qtpdf and self._qtpdf_doc and self._qtpdf_view) and self._path:
            self._render_current_with_pymupdf()
