# app/ui/save_pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sqlalchemy import text

from app.ddl_loader import apply_schema
from app.seed_line_items import run as seed_line_items
from app.etl.load import (
    get_line_item_candidates, insert_fact_rows,
    detect_consolidation, detect_report_type, detect_gaap, detect_period,
)
from app.etl.notes_unified_loader import insert_notes_unified_generic
from app.etl.extractors import guess_ticker_from_filename
from app.db import SessionLocal

def _ensure_schema_seed():
    # chạy safe-idempotent
    apply_schema()
    seed_line_items()

def _create_or_get_company(session, ticker: str) -> int:
    return session.execute(text("""
        INSERT INTO companies(ticker, company_name)
        VALUES (:t, :t)
        ON CONFLICT(ticker) DO UPDATE SET company_name=EXCLUDED.company_name
        RETURNING company_id
    """), {"t": ticker}).scalar_one()

def _create_or_get_report(session, company_id: int, pdf_path: Path) -> int:
    # metadata giống logic trong load.process_one_pdf
    meta_text = ""
    rtype = detect_report_type(meta_text, filename=pdf_path.name)
    cons  = detect_consolidation(meta_text)
    gaap  = detect_gaap(meta_text)
    per   = detect_period(meta_text, filename=pdf_path.name)
    rid = session.execute(text("""
        INSERT INTO reports(company_id, period_start, period_end, report_type, consolidation, gaap, source_file)
        VALUES (:cid, :ps, :pe, :rtype, :cons, :gaap, :src)
        ON CONFLICT (company_id, period_start, period_end, consolidation, gaap)
        DO UPDATE SET source_file=EXCLUDED.source_file
        RETURNING report_id
    """), dict(
        cid=company_id,
        ps=per["start"],
        pe=per["end"],
        rtype=rtype,
        cons=cons,
        gaap=gaap,
        src=str(pdf_path)
    )).scalar_one()
    return rid

def persist_preview(pdf_path: Path, df_facts: pd.DataFrame, df_notes: pd.DataFrame) -> int:
    """
    Lưu dữ liệu đã xem trước vào DB (chỉ khi người dùng bấm 'Lưu').
    - Tạo/khởi tạo schema nếu chưa có
    - Upsert company/report từ filename
    - Insert facts & notes_unified (idempotent theo UNIQUE/ON CONFLICT)
    """
    _ensure_schema_seed()
    ticker = guess_ticker_from_filename(pdf_path) or "UNKNOWN"

    with SessionLocal() as s:
        s.begin()
        try:
            cid = _create_or_get_company(s, ticker)
            rid = _create_or_get_report(s, cid, pdf_path)

            cands = get_line_item_candidates(s)
            if df_facts is not None and not df_facts.empty:
                insert_fact_rows(s, rid, df_facts, cands)

            rows = df_notes.to_dict("records") if df_notes is not None and not df_notes.empty else []
            if rows:
                insert_notes_unified_generic(rid, rows)

            s.commit()
            return rid
        except Exception:
            s.rollback()
            raise
