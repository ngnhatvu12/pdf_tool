from pathlib import Path
import pandas as pd
from .json_export import export_report_json
from sqlalchemy import text
from ..db import SessionLocal
from .extractors import guess_ticker_from_filename, detect_consolidation, detect_report_type, detect_gaap, detect_period, detect_auditor
from .normalize import choose_line_item
from app.ocr_processor import extract_tables_ocr
from .table_finder import locate_statement_pages
from app.notes.pipeline import extract_notes_rows
from app.etl.notes_unified_loader import insert_notes_unified_generic
from typing import Optional 

MAX_AMOUNT_NUMERIC = 10**18 - 1

def validate_amount(amount):
    if amount is None: return None
    try:
        v = float(amount)
        if abs(v) >= MAX_AMOUNT_NUMERIC: return None
        return amount
    except (ValueError, TypeError):
        return None

def get_line_item_candidates(session):
    rows = session.execute(text("SELECT std_code, native_examples, vas_code FROM line_items")).all()
    return [(r[0], r[1] or [], r[2]) for r in rows]

def insert_fact_rows(session, report_id: int, df: pd.DataFrame, candidates):
    inserted = skipped = 0
    unmatched_labels = set()
    is_long = {"context_key", "amount"} <= set(df.columns)

    for _, row in df.iterrows():
        label = str(row.get("src_label") or "")
        vas_code = row.get("vas_code")

        if not label.strip():
            skipped += 1
            continue

        code = choose_line_item(label, vas_code, candidates, score_cutoff=15)
        if not code:
            unmatched_labels.add(label)
            skipped += 1
            continue

        lid = session.execute(
            text("SELECT line_item_id FROM line_items WHERE std_code=:c"),
            {"c": code}
        ).scalar()
        if not lid:
            skipped += 1
            continue

        def _ins(ctx, val):
            nonlocal inserted, skipped
            v = validate_amount(val)
            if v is None: 
                return
            try:
                session.execute(text("""
                    INSERT INTO facts(report_id, line_item_id, context, amount, unit, page, src_label, confidence)
                    VALUES (:rid, :lid, :ctx, :amt, 'VND', :pg, :lbl, :conf)
                    ON CONFLICT (report_id, line_item_id, context, dims) DO UPDATE
                      SET amount=EXCLUDED.amount, src_label=EXCLUDED.src_label, page=EXCLUDED.page,
                          confidence=GREATEST(facts.confidence, EXCLUDED.confidence)
                """), dict(
                    rid=report_id, lid=lid, ctx=ctx, amt=v,
                    pg=int(row.get("page") or 0), lbl=label, conf=row.get("confidence", 0.7)
                ))
                inserted += 1
            except Exception as e:
                print(f"❌ Lỗi insert fact: {e}")
                skipped += 1

        if is_long:
            ctx = str(row.get("context_key") or "unknown")
            _ins(ctx, row.get("amount"))
        else:
            cur = validate_amount(row.get("amount_current"))
            pri = validate_amount(row.get("amount_prior"))
            if cur is not None: _ins("current_period", cur)
            if pri is not None: _ins("prior_period",  pri)

    if unmatched_labels:
        print(f"🔍 Có {len(unmatched_labels)} label không match được")
        short_labels = [l for l in unmatched_labels if len(l) < 10]
        long_labels = [l for l in unmatched_labels if len(l) >= 10]
        if short_labels:
            print(f"   📝 Label ngắn ({len(short_labels)}): {short_labels[:5]}...")
        if long_labels:
            print(f"   📖 Label dài ({len(long_labels)}), ví dụ:")
            for label in list(long_labels)[:5]:
                print(f"      - '{label}'")

    print(f"📊 Đã chèn {inserted} dòng, bỏ qua {skipped} dòng")

def harvest_all_tables(pdf_path: Path, ocr_engine: str = "tesseract") -> pd.DataFrame:
    print("🔎 Tự động nhận diện vùng trang chứa bảng (BS/IS/CF)...")
    # Để pages=None, extract_tables_ocr sẽ tự locate_statement_pages 1 lần.
    result = extract_tables_ocr(pdf_path, pages=None, ocr_engine=ocr_engine)
    if isinstance(result, tuple):
        df, _ = result
    else:
        df = result
    return df

def process_one_pdf(pdf_path: Path,ocr_engine: str = "tesseract",notes_search_from: Optional[int] = None):
    print(f"🚀 XỬ LÝ ĐẦY ĐỦ:")
    print(f"🔍 Xử lý file: {pdf_path.name}")
    ticker = guess_ticker_from_filename(pdf_path) or "UNKNOWN"

    meta = {
        "consolidation": detect_consolidation(""),
        "report_type":   detect_report_type("", filename=pdf_path.name),
        "gaap":          detect_gaap(""),
        "auditor":       detect_auditor(""),
    }
    per = detect_period("", filename=pdf_path.name)
    meta["period_start"], meta["period_end"] = per["start"], per["end"]
    print(f"📊 Meta extracted: {meta}")

    with SessionLocal() as s:
        s.begin()
        try:
            cid = s.execute(text("""
                INSERT INTO companies(ticker, company_name)
                VALUES (:t, :n)
                ON CONFLICT(ticker) DO UPDATE SET company_name=EXCLUDED.company_name
                RETURNING company_id
            """), {"t": ticker, "n": ticker}).scalar_one()

            rid = s.execute(text("""
              INSERT INTO reports(company_id, period_start, period_end, report_type, consolidation, gaap, auditor, source_file)
              VALUES (:cid, :ps, :pe, :rtype, :cons, :gaap, :aud, :src)
              ON CONFLICT (company_id, period_start, period_end, consolidation, gaap)
              DO UPDATE SET auditor=EXCLUDED.auditor, source_file=EXCLUDED.source_file
              RETURNING report_id
            """), dict(
                cid=cid, ps=meta["period_start"], pe=meta["period_end"],
                rtype=meta["report_type"], cons=meta["consolidation"],
                gaap=meta["gaap"], aud=meta.get("auditor"), src=str(pdf_path)
            )).scalar_one()

            # 1) BS/IS/CF → facts (như cũ)
            candidates = get_line_item_candidates(s)
            df = harvest_all_tables(pdf_path, ocr_engine=ocr_engine)
            print(f"📋 Dòng ứng viên từ OCR: {0 if df is None else len(df)}")
            if df is not None and not df.empty:
                insert_fact_rows(s, rid, df, candidates)
            else:
                print("❌ Không có dữ liệu nào được trích xuất")
            s.commit()

            # 2) NOTES (thuyết minh) – dùng pipeline mới
            rows = extract_notes_rows(pdf_path, search_from_page=notes_search_from)
            print(f"🧾 Notes rows extracted: {len(rows)}")
            inserted = insert_notes_unified_generic(rid, rows)
            print(f"✅ Notes unified inserted: {inserted} rows")

            # 3) JSON EXPORT
            json_path = export_report_json(rid)
            print(f"📦 Xuất JSON đối chiếu: {json_path}")

            return {"ticker": ticker, "report_meta": meta, "json_file": str(json_path)}
        except Exception as e:
            s.rollback()
            print(f"❌ Lỗi: {e}")
            raise