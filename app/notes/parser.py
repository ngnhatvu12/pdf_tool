from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re
import pandas as pd
from app.etl.normalize import normalize_with_unit

NUM_ANY  = re.compile(r"\(?[-+]?\s*(?:\d{1,3}(?:[.\s]\d{3})+|\d+)(?:,\d+|\.\d+)?\)?$")
DATE_RE  = re.compile(r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})')


# ---------- helpers ----------
def unit_from_blob(text_blob: str) -> str:
    t = (text_blob or "").lower()
    if "tỷ đồng" in t or " ty dong" in t or " tỷ " in t:
        return "tỷ đồng"
    if "triệu đồng" in t or " trieu dong" in t:
        return "triệu đồng"
    if "nghìn đồng" in t or "ngàn đồng" in t or " nghin dong" in t or " ngan dong" in t:
        return "nghìn đồng"
    return "VND"


def _clean_label(s: str) -> str:
    s = str(s or "")
    s = s.replace("\n", " ").strip()
    s = re.sub(r"^\s*[\-\u2022\.••▪●•·\*]+", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_garbage_label(s: str) -> bool:
    t = (s or "").strip()
    if t == "" or len(t) <= 1:
        return True
    if re.fullmatch(r"[^\w\u00C0-\u1EF9]+", t):
        return True   # toàn ký hiệu
    if re.fullmatch(r"\d{1,2}", t):
        return True   # STT 1-2 chữ số
    if t.lower() in {"col", "c", "đơn vị", "don vi"}:
        return True
    return False


def _is_numberish(x: Any) -> bool:
    """
    Nhận diện ô có "tính số liệu".
    Bộ lọc rất nới tay: chỉ cần có chữ số là True (kể cả 1-2 chữ số).
    (Lọc noise đã được xử lý ở bước nhận diện bảng của table_extractor.)
    """
    s = str(x or "").strip()
    if not s:
        return False
    if s in {"-", "—", "–"}:
        return True
    if NUM_ANY.fullmatch(s):
        return True
    return any(ch.isdigit() for ch in s)


def _detect_unit_table(df: pd.DataFrame) -> str:
    blob = " ".join(
        [str(v) for v in df.head(5).fillna("").values.flatten().tolist()]
    )
    return unit_from_blob(blob)


def _normalize_context_header(txt: str) -> str:
    t = (txt or "").lower()
    m = re.search(r'qu[ýy]?\s*(\d)\s*[\/\-]?\s*(20\d{2})', t)
    if m:
        return f"q{m.group(1)}_{m.group(2)}"
    if "kỳ kế toán" in t or "ky ke toan" in t or "lũy kế" in t or "luy ke" in t or "6 tháng" in t or "6 thang" in t:
        y = re.search(r'(20\d{2})', t)
        return f"ytd_{y.group(1) if y else 'unk'}"
    d = DATE_RE.findall(t)
    if d:
        last = d[-1].replace("/", "-")
        y = re.search(r"(20\d{2})", last)
        return f"asof_{y.group(1) if y else last}"
    if "đầu kỳ" in t or "opening" in t:
        return "opening"
    if "cuối kỳ" in t or "closing" in t:
        return "closing"
    t = re.sub(r"\s+", "_", t.strip())
    return t[:40] or "col"


# ---------- header collapse ----------
def _collapse_header_rows(df: pd.DataFrame, max_header_rows: int = 3) -> Tuple[List[str], int]:
    """
    Gộp 1–3 dòng đầu thành tên cột; dừng khi tỷ lệ numberish >= 40% (đã sang body).
    Trả: (cols, data_start_row_index)
    """
    if df.empty:
        cols = [f"h{i+1}" for i in range(max(1, df.shape[1]))]
        return cols, 0

    m = min(max_header_rows, max(1, df.shape[0]))
    data_start = 0
    for r in range(m):
        row = df.iloc[r].astype(str).tolist()
        num_ratio = sum(_is_numberish(x) for x in row) / max(1, len(row))
        if num_ratio >= 0.4 and r > 0:
            data_start = r
            break
        data_start = r + 1
    data_start = min(data_start, m)

    header_block = df.iloc[:data_start] if data_start > 0 else df.iloc[:0]
    if header_block.shape[0] == 0:
        cols = [f"h{i+1}" for i in range(df.shape[1])]
        return cols, 0

    pieces = []
    for r in range(header_block.shape[0]):
        pieces.append([str(v).replace("\n", " ").strip()
                       for v in header_block.iloc[r].tolist()])

    cols: List[str] = []
    ncol = df.shape[1]
    for j in range(ncol):
        parts = []
        for r in range(len(pieces)):
            cell = (pieces[r][j] if j < len(pieces[r]) else "").strip()
            if cell and cell not in {"-", "—"}:
                parts.append(cell)
        label = " ".join(parts).strip()
        label = re.sub(r"\s+", " ", label)
        cols.append(label if label else f"h{j+1}")
    return cols, data_start


def _pick_name_col(df: pd.DataFrame, cols: List[str]) -> int:
    """
    Chọn cột mô tả (label) cho bảng.
    Ưu tiên:
      - nhiều giá trị không phải số
      - đa dạng (nhiều giá trị khác nhau)
      - không chứa từ khoá kiểu đơn vị, số dư...
    """
    best_i, best_score = 0, -1.0
    n = df.shape[0]
    for i, c in enumerate(cols):
        series = df.iloc[:, i].astype(str).fillna("")
        nonnum_ratio = sum(not _is_numberish(x) for x in series) / max(1, n)
        uniq_ratio = series.nunique(dropna=True) / max(1, n)
        penalty = 0.0
        cl = c.lower()
        if any(k in cl for k in ["vnd", "vnđ", "số dư", "so du", "ngày", "date"]):
            penalty += 0.2
        short_ratio = sum(len(str(x).strip()) <= 2 for x in series) / max(1, n)
        score = nonnum_ratio * 0.65 + uniq_ratio * 0.45 - penalty - short_ratio * 0.35
        if score > best_score:
            best_score, best_i = score, i
    return best_i


# ---------- main normalize/flatten ----------
def _headers_from_vector(df: pd.DataFrame) -> Tuple[List[str], int, int]:
    """
    Bảng vector (pdfplumber / camelot):
      - Gộp vài dòng đầu làm header
      - Chọn cột name
    """
    cols_collapsed, data_start = _collapse_header_rows(df, max_header_rows=3)
    work = df.iloc[data_start:].reset_index(drop=True).copy()
    if work.empty:
        cols = [f"h{i+1}" for i in range(df.shape[1])]
        return cols, 0, data_start
    work.columns = cols_collapsed
    name_idx = _pick_name_col(work, cols_collapsed)
    return cols_collapsed, name_idx, data_start


def _headers_from_ocr(df: pd.DataFrame) -> Tuple[List[str], int]:
    """
    Bảng từ OCR:
      - Nếu đã có cột 'desc' (do table_extractor tạo) thì dùng luôn cột đó làm name,
        giữ nguyên các tên cột còn lại (đây chính là header OCR: 'Quý 2/2025', 'YTD', ...)
      - Nếu không có 'desc', fallback đặt tên h1, h2, ...
    """
    cols = list(df.columns)
    if "desc" in cols:
        name_idx = cols.index("desc")
        # Giữ nguyên tên cột để flatten_table dùng text header này làm context
        return cols, name_idx
    # fallback
    cols = [f"h{i+1}" for i in range(df.shape[1])]
    df.columns = cols
    return cols, 0


def normalize_table(df: pd.DataFrame, mode: str, debug_tag: str = "") -> Tuple[pd.DataFrame, List[str], int, str]:
    """
    Chuẩn hoá tiêu đề, xác định cột diễn giải, suy luận đơn vị.
    Trả: df_norm, cols, name_idx, unit_hint
    """
    df = df.copy().fillna("").replace("\n", " ", regex=True)
    if mode == "vector":
        cols, name_idx, data_start = _headers_from_vector(df)
        df = df.iloc[data_start:].reset_index(drop=True)
        df.columns = cols
    else:
        cols, name_idx = _headers_from_ocr(df)
        df.columns = cols
    unit_hint = _detect_unit_table(df)
    return df, cols, name_idx, unit_hint


def flatten_table(
    df: pd.DataFrame,
    cols: List[str],
    name_idx: int,
    unit_hint: str,
    page: int,
    note_no: Optional[int],
    note_title: Optional[str],
    table_idx: int,
) -> List[Dict[str, Any]]:

    # map cột -> context_key
    ctxs: List[str] = []
    for i, c in enumerate(cols):
        if i == name_idx:
            ctxs.append("desc")
            continue
        ctxs.append(_normalize_context_header(str(c)))

    out: List[Dict[str, Any]] = []
    body = df.copy()

    block_prefix = ""
    section_prefix = ""
    pending_suffix = ""

    value_cols_idx = [i for i in range(len(cols)) if i != name_idx]

    def row_has_number(row) -> bool:
        for i in value_cols_idx:
            s = str(row.iloc[i] or "").strip()
            if _is_numberish(s):
                return True
        return False

    def update_prefix(text: str):
        nonlocal block_prefix, section_prefix
        t = _clean_label(text).lower()
        keywords_block = [
            "ngắn hạn", "ngan han", "dài hạn", "dai han",
            "nguyên giá", "nguyen gia",
            "giá trị hao mòn lũy kế", "gia tri hao mon luy ke",
            "giá trị còn lại", "gia tri con lai",
        ]
        if any(k in t for k in keywords_block):
            block_prefix = _clean_label(text)
            section_prefix = ""
        else:
            section_prefix = _clean_label(text)

    def looks_like_suffix(text: str) -> bool:
        t = _clean_label(text).lower()
        if (t.startswith("(") and t.endswith(")")):
            return True
        if "thuyết minh" in t or "thuyet minh" in t:
            return True
        return False

    for _, row in body.iterrows():
        raw_label = _clean_label(row.iloc[name_idx])

        if not raw_label or _is_garbage_label(raw_label):
            # dòng tiêu đề / heading không có số
            if raw_label and not row_has_number(row):
                update_prefix(raw_label)
            continue

        if not row_has_number(row):
            # dòng thuần text trong bảng
            if looks_like_suffix(raw_label):
                pending_suffix = raw_label
            else:
                update_prefix(raw_label)
            continue

        parts: List[str] = []
        if block_prefix:
            parts.append(block_prefix)
        if section_prefix and section_prefix.lower() != raw_label.lower():
            parts.append(section_prefix)
        parts.append(raw_label)
        if pending_suffix:
            parts[-1] = f"{parts[-1]} {pending_suffix}"
            pending_suffix = ""

        final_label = " ".join(parts).strip()

        for cidx, ctx in enumerate(ctxs):
            if cidx == name_idx:
                continue
            raw_val = str(row.iloc[cidx] or "").strip()
            if not raw_val:
                continue
            if raw_val in {"-", "—", "–"}:
                v = 0.0
            else:
                clean_val = re.sub(r"[()]", "", raw_val)
                v, _ = normalize_with_unit(clean_val, unit_hint)
            if v is None:
                continue

            out.append(dict(
                note_group="note_table",
                item_key=final_label,
                item_label=final_label,
                context_key=ctx,
                amount=float(v),
                unit=unit_hint if unit_hint != "VND" else "VND",
                page=int(page),
                src_label=final_label,
                dims=dict(
                    note_no=note_no,
                    note_title=note_title,
                    table_idx=table_idx,
                    col_idx=cidx,
                    col_label=str(cols[cidx]) if cidx < len(cols) else str(ctx),
                ),
                confidence=0.92 if note_no is not None else 0.88,
            ))
    return out
