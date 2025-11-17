from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import fitz  # PyMuPDF

# "Thuyết minh" / "Notes to the financial statements"
NOTES_TITLE_PAT = re.compile(
    r"(thuy(?:ê|e)t\s*minh(?:\s*b(?:á|a)o\s*c(?:á|a)o\s*t(?:à|a)i\s*ch(?:í|i)nh)?)|"
    r"(notes?\s+to\s+the\s+financial\s+statements?)",
    re.I | re.U,
)

# Các cách viết trong mục lục: "Thuyết minh ... 10-49" hoặc "... trang 10–49"
TOC_RANGE_PAT = re.compile(
    r"thuy(?:ê|e)t\s*minh[^\n]*?(\d{1,3})\s*[\-–]\s*(\d{1,3})",
    re.I | re.U,
)

NOTE_HEAD_PAT = re.compile(r"^\s*(\d{1,3})\s*[\.\-:]\s+(.+)$", re.U)

def _read_pages_text(pdf_path: Path) -> List[str]:
    texts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for p in doc:
            texts.append(p.get_text("text") or "")
    return texts

def locate_notes_range(pdf_path: Path, default_start: Optional[int] = None) -> Tuple[int, int]:
    """
    Trả (start_page, end_page) 1-based cho vùng THUYẾT MINH.
    Ưu tiên:
      1) Mục lục có pattern "Thuyết minh ... X–Y"
      2) default_start được truyền từ filename anchor
      3) tìm trang đầu có tiêu đề "Thuyết minh", fallback từ trang 12 → cuối
    """
    texts = _read_pages_text(pdf_path)
    n = len(texts)

    # 1) Mục lục (thường xuất hiện đầu file)
    for i in range(min(8, n)):
        m = TOC_RANGE_PAT.search(texts[i])
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            a = max(1, min(a, n))
            b = max(a, min(b, n))
            return a, b

    # 2) default_start (anchor theo ticker/file)
    if default_start and 1 <= default_start <= n:
        return default_start, n

    # 3) tìm tiêu đề "Thuyết minh" đầu tiên
    start: Optional[int] = None
    for i in range(1, n + 1):
        if NOTES_TITLE_PAT.search(texts[i - 1]):
            start = i
            break
    if not start:
        start = min(12, n)
    return start, n

def detect_note_headers(pdf_path: Path, a: int, b: int) -> List[Tuple[int, int, str]]:
    """
    Trong [a,b] tìm các dòng dạng "10. Tiền và tương đương tiền".
    Trả: [(page, note_no, title)] theo thứ tự trang tăng dần, bỏ trùng theo số note.
    """
    texts = _read_pages_text(pdf_path)
    marks: List[Tuple[int, int, str]] = []
    for p in range(a, b + 1):
        for line in (texts[p - 1] or "").splitlines():
            m = NOTE_HEAD_PAT.search(line.strip())
            if not m:
                continue
            try:
                no = int(m.group(1))
            except Exception:
                continue
            title = m.group(2).strip()
            if title and len(title) >= 3:
                marks.append((p, no, title))

    seen = set()
    uniq: List[Tuple[int, int, str]] = []
    for p, k, t in sorted(marks, key=lambda z: (z[1], z[0])):
        if k in seen:
            continue
        seen.add(k)
        uniq.append((p, k, t))
    uniq.sort(key=lambda z: z[0])
    return uniq

def slice_note_ranges(pdf_path: Path, a: int, b: int) -> Dict[int, Tuple[int, int, str]]:
    """
    Map {note_no: (start, end, title)}. Nếu không phát hiện được tiêu đề → trả rỗng.
    """
    headers = detect_note_headers(pdf_path, a, b)
    if not headers:
        return {}

    ranges: Dict[int, Tuple[int, int, str]] = {}
    for i, (pg, no, title) in enumerate(headers):
        pend = (headers[i + 1][0] - 1) if (i + 1) < len(headers) else b
        ranges[no] = (pg, max(pg, pend), title)
    return ranges
