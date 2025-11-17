from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

from .locator import locate_notes_range, slice_note_ranges
from .table_extractor import harvest_tables
from .parser import normalize_table, flatten_table

def _anchor_from_filename(pdf_path: Path) -> Optional[int]:
    """
    Cho phép chỉ định nhanh trang bắt đầu theo ticker (khi mục lục bị lỗi).
    """
    name = pdf_path.name.upper()
    if "GEX" in name: return 15
    if "CTD" in name: return 17
    if "HAC" in name: return 20
    return None

def extract_notes_rows(pdf_path: Path, search_from_page: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    1) Xác định vùng thuyết minh [a,b] (ưu tiên mục lục → anchor → tiêu đề)
    2) Chia theo từng note "10.", "11.",...
    3) Gộp bảng từ 3 nguồn (vector / camelot(stream) / OCR)
    4) Chuẩn hoá → flatten long-form rows
    """
    if search_from_page is None:
        search_from_page = _anchor_from_filename(pdf_path)

    a, b = locate_notes_range(pdf_path, default_start=search_from_page)
    print(f"🧭 Notes range: pages [{a}..{b}]")
    rows: List[Dict[str, Any]] = []

    note_ranges = slice_note_ranges(pdf_path, a, b)
    total_tables = 0

    if note_ranges:
        print(f"🔎 Detect note headers: {len(note_ranges)} notes -> {sorted(note_ranges.keys())[:12]}{' …' if len(note_ranges)>12 else ''}")
        for no, (s, e, title) in sorted(note_ranges.items(), key=lambda kv: kv[0]):
            print(f"— Note {no}: pages [{s}..{e}] · title='{(title or '')[:80]}'")
            tables = harvest_tables(pdf_path, s, e)
            for idx, (df, pg, mode) in enumerate(tables, 1):
                total_tables += 1
                debug_tag = f"p{pg} note={no} table={idx} mode={mode}"
                df_norm, cols, name_idx, unit_hint = normalize_table(df, mode, debug_tag=debug_tag)
                rows_out = flatten_table(df_norm, cols, name_idx, unit_hint, pg, no, title, idx)
                rows += rows_out
    else:
        print("⚠️  Không tìm thấy tiêu đề note rõ ràng. Quét full vùng thuyết minh.")
        tables = harvest_tables(pdf_path, a, b)
        for idx, (df, pg, mode) in enumerate(tables, 1):
            total_tables += 1
            debug_tag = f"p{pg} note=? table={idx} mode={mode}"
            df_norm, cols, name_idx, unit_hint = normalize_table(df, mode, debug_tag=debug_tag)
            rows_out = flatten_table(df_norm, cols, name_idx, unit_hint, pg, None, None, idx)
            rows += rows_out

    print(f"✅ Notes harvest summary: tables={total_tables}, rows_out={len(rows)}")
    return rows
