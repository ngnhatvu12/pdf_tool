# app/notes/notes_unified_loader.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from ..db import SessionLocal
import hashlib, json, math

def _clean_float(x):
    if x is None: 
        return None
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def _fp(report_id: int, note_group: str, item_key: Optional[str], context_key: Optional[str],
        dims: Optional[dict], amount: Optional[float], page: Optional[int]) -> str:
    """
    Fingerprint ổn định để upsert idempotent (không phụ thuộc thứ tự insert).
    """
    dims_norm = json.dumps(dims or {}, ensure_ascii=False, sort_keys=True)
    amt_norm = None if amount is None else round(float(amount), 4)
    raw = f"{report_id}|{note_group}|{item_key or ''}|{context_key or ''}|{dims_norm}|{amt_norm}|{page or 0}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _insert_many(rows: List[dict]) -> int:
    if not rows:
        return 0
    with SessionLocal() as s:
        s.begin()
        try:
            for r in rows:
                s.execute(text("""
                    INSERT INTO notes_unified(report_id, note_group, item_key, item_label, context_key,
                                              amount, unit, dims, page, src_label, confidence, fp)
                    VALUES (:rid, :grp, :ikey, :ilbl, :ctx, :amt, :unit, CAST(:dims AS JSONB), :pg, :src, :conf, :fp)
                    ON CONFLICT (report_id, fp) DO UPDATE
                    SET amount      = COALESCE(EXCLUDED.amount, notes_unified.amount),
                        unit        = COALESCE(EXCLUDED.unit, notes_unified.unit),
                        src_label   = COALESCE(EXCLUDED.src_label, notes_unified.src_label),
                        page        = COALESCE(EXCLUDED.page, notes_unified.page),
                        confidence  = GREATEST(notes_unified.confidence, EXCLUDED.confidence)
                """), r)
            s.commit()
            return len(rows)
        except Exception:
            s.rollback()
            raise

def insert_notes_unified_generic(report_id: int, rows: List[Dict[str, Any]]) -> int:
    """
    Nhận list row long-form (từ extract_notes_rows) và insert vào notes_unified.
    """
    payload: List[dict] = []
    for r in rows:
        amt = _clean_float(r.get("amount"))
        dims = r.get("dims") or {}
        fp = _fp(report_id, r.get("note_group","note_table"), r.get("item_key"),
                 r.get("context_key"), dims, amt, r.get("page"))
        payload.append(dict(
            rid=report_id,
            grp=r.get("note_group","note_table"),
            ikey=r.get("item_key"),
            ilbl=r.get("item_label"),
            ctx=r.get("context_key"),
            amt=amt,
            unit=r.get("unit") or "VND",
            dims=json.dumps(dims, ensure_ascii=False),
            pg=int(r.get("page") or 0),
            src=r.get("src_label"),
            conf=float(r.get("confidence") or 0.8),
            fp=fp
        ))
    return _insert_many(payload)
