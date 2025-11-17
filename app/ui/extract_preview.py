from pathlib import Path
from typing import Tuple
import pandas as pd

from app.etl.load import harvest_all_tables
from app.notes.pipeline import extract_notes_rows

def extract_preview(pdf_path: Path, ocr_engine: str = "tesseract") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trích dữ liệu ở chế độ 'xem trước', KHÔNG insert DB.
    Trả về:
      - df_facts_long: long-form (BS/IS/CF) như extract_tables_ocr tạo
      - df_notes: DataFrame từ rows của thuyết minh (notes_unified schema)

    ocr_engine: 'tesseract' hoặc 'paddle' – truyền xuống harvest_all_tables.
    """
    # long-form (context_key, amount, page, ...)
    df_facts = harvest_all_tables(pdf_path, ocr_engine=ocr_engine)
    if df_facts is None:
        df_facts = pd.DataFrame()

    # pipeline thuyết minh hiện tại không nhận ocr_engine,
    # nên vẫn dùng engine mặc định bên trong notes.pipeline (thường là Tesseract).
    rows = extract_notes_rows(pdf_path, search_from_page=None)
    df_notes = pd.DataFrame(rows) if rows else pd.DataFrame()

    return df_facts, df_notes
