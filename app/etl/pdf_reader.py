from pathlib import Path
import fitz  # PyMuPDF
import camelot
from typing import List, Tuple
import pdfplumber


def read_pdf_tables_textbased(fp: Path, pages="1-end"):
    try:
        with pdfplumber.open(fp) as pdf:
            # thử trích bảng bằng pdfplumber trước
            return pdf
    except:
        return None
def read_pdf_tables(fp: Path, pages: str = "1-end"):
    """Trích bảng bằng Camelot (stream) để bám cột kẻ mảnh như biểu mẫu B01/B02/B03."""
    try:
        tables = camelot.read_pdf(str(fp), pages=pages, flavor="stream")
        return tables  # list of Table
    except Exception as e:
        print(f"[WARN] Camelot failed: {e}")
        return []
