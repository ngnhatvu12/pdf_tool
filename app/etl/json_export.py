# app/etl/json_export.py
from __future__ import annotations
from typing import Dict, Any, List
from sqlalchemy import text
from ..db import SessionLocal
from pathlib import Path
import json
from datetime import datetime

def _as_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        return f
    except Exception:
        return None

def build_report_payload(report_id: int) -> Dict[str, Any]:
    """
    Gom toàn bộ dữ liệu đã insert cho report_id:
      - company, report meta
      - facts (join line_items để có std_code)
      - notes_unified (thuyết minh)
    Trả về dict thuần để ghi JSON.
    """
    with SessionLocal() as s:
        # company + report
        rp = s.execute(text("""
            SELECT r.report_id, r.company_id, r.period_start, r.period_end,
                   r.report_type, r.consolidation, r.gaap, r.auditor, r.source_file,
                   c.ticker, c.company_name
            FROM reports r
            JOIN companies c ON c.company_id = r.company_id
            WHERE r.report_id = :rid
        """), {"rid": report_id}).mappings().first()
        if not rp:
            raise RuntimeError(f"Report {report_id} not found")

        # facts
        facts = s.execute(text("""
            SELECT
              li.std_code,
              li.std_label,
              f.context,
              f.amount,
              f.unit,
              f.page,
              f.src_label,
              f.confidence
            FROM facts f
            JOIN line_items li ON li.line_item_id = f.line_item_id
            WHERE f.report_id = :rid
            ORDER BY li.std_code, f.context
        """), {"rid": report_id}).mappings().all()

        facts_out: List[Dict[str, Any]] = []
        for r in facts:
            facts_out.append(dict(
                std_code   = r["std_code"],
                std_label  = r["std_label"],
                context    = r["context"],
                amount     = _as_float(r["amount"]),
                unit       = r["unit"],
                page       = r["page"],
                src_label  = r["src_label"],
                confidence = float(r["confidence"] or 0.0),
            ))

        # notes_unified
        notes = s.execute(text("""
            SELECT
              note_group,
              item_key,
              item_label,
              context_key,
              amount,
              unit,
              dims,
              page,
              src_label,
              confidence
            FROM notes_unified
            WHERE report_id = :rid
            ORDER BY COALESCE(page,0), item_key, context_key
        """), {"rid": report_id}).mappings().all()

        notes_out: List[Dict[str, Any]] = []
        for r in notes:
            # dims đã là JSONB trong DB; lấy ra dạng str → parse về dict an toàn
            dims_val = r["dims"]
            if isinstance(dims_val, (dict, list)):
                dims_obj = dims_val
            else:
                try:
                    dims_obj = json.loads(dims_val) if dims_val else {}
                except Exception:
                    dims_obj = {}

            notes_out.append(dict(
                note_group = r["note_group"],
                item_key   = r["item_key"],
                item_label = r["item_label"],
                context_key= r["context_key"],
                amount     = _as_float(r["amount"]),
                unit       = r["unit"],
                dims       = dims_obj,
                page       = r["page"],
                src_label  = r["src_label"],
                confidence = float(r["confidence"] or 0.0),
            ))

        payload: Dict[str, Any] = dict(
            _generated_at = datetime.now().isoformat(timespec="seconds"),
            company = dict(
                company_id   = rp["company_id"],
                ticker       = rp["ticker"],
                company_name = rp["company_name"],
            ),
            report = dict(
                report_id     = rp["report_id"],
                period_start  = str(rp["period_start"]),
                period_end    = str(rp["period_end"]),
                report_type   = rp["report_type"],
                consolidation = rp["consolidation"],
                gaap          = rp["gaap"],
                auditor       = rp["auditor"],
                source_file   = rp["source_file"],
            ),
            facts  = facts_out,
            notes  = notes_out,
        )
        return payload

def export_report_json(report_id: int, out_dir: Path | None = None) -> Path:
    """
    Build payload và ghi ra file JSON trong thư mục 'json'.
    Trả về đường dẫn file vừa tạo.
    """
    payload = build_report_payload(report_id)

    # Mặc định: thư mục 'json' ở root repo (cùng level với app/)
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / "json"
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker = payload["company"]["ticker"] or "UNKNOWN"
    pe = payload["report"]["period_end"]
    rtype = payload["report"]["report_type"]
    cons = payload["report"]["consolidation"]
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    fname = f"{ticker}_{pe}_{rtype}_{cons}_{ts}.json"
    fpath = out_dir / fname

    with fpath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return fpath
