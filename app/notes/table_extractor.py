from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import io, re, os
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
from PIL import Image
import pytesseract
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- ENV (Windows optional) ---
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# --------- regex & flags ----------
NUM_ANY  = re.compile(r"\(?[-+]?\s*(?:\d{1,3}(?:[.\s]\d{3})+|\d+)(?:,\d+|\.\d+)?\)?$")
NUMISH   = re.compile(r'^[\d\.\,\(\)\-\s]+$')
DATE_RE  = re.compile(r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})')

# Bật / tắt log OCR raw lines (nếu nhiều log quá anh có thể đặt = False)
DEBUG_OCR_LINES = True


# ---------- OCR basic ----------
def _ocr_image(pdf_path: Path, pageno1: int, dpi=420) -> Image.Image:
    """
    Render 1 trang PDF -> ảnh RGB để dùng cho OCR.
    """
    with fitz.open(pdf_path) as doc:
        p = doc.load_page(pageno1 - 1)
        pm = p.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        return Image.open(io.BytesIO(pm.tobytes("png")))


def _ocr_words(img: Image.Image) -> pd.DataFrame:
    """
    Trả df(word,x,y,w,h,conf) từ Tesseract.
    - Pass 1: psm6 (mode block) với ngưỡng conf>=8
    - Nếu token quá ít -> Pass 2: psm4 (mode paragraph), không chặn conf.
    Bộ lọc rất nhẹ tay để không bỏ sót text.
    """
    cfg = r'--oem 3 --psm 6 -l vie+eng -c preserve_interword_spaces=1'
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg)
    rows = []
    for i, txt in enumerate(d["text"]):
        t = (txt or "").strip()
        if not t:
            continue
        conf = d["conf"][i]
        try:
            cf = float(conf)
        except Exception:
            cf = -1.0
        if cf < 8:
            continue
        rows.append({
            "text": t,
            "x": d["left"][i],
            "y": d["top"][i],
            "w": d["width"][i],
            "h": d["height"][i],
            "conf": cf
        })
    df = pd.DataFrame(rows)

    # Pass 2 nếu token quá ít => cố gắng OCR lại toàn trang
    if len(df) < 80:
        cfg2 = r'--oem 3 --psm 4 -l vie+eng -c preserve_interword_spaces=1'
        d2 = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg2)
        rows2 = []
        for i, txt in enumerate(d2["text"]):
            t = (txt or "").strip()
            if not t:
                continue
            conf = d2["conf"][i]
            try:
                cf = float(conf)
            except Exception:
                cf = -1.0
            rows2.append({
                "text": t,
                "x": d2["left"][i],
                "y": d2["top"][i],
                "w": d2["width"][i],
                "h": d2["height"][i],
                "conf": cf
            })
        df = pd.DataFrame(rows2)

    return df


def _group_lines(df: pd.DataFrame, y_tol=8) -> List[pd.DataFrame]:
    """
    Gom word theo dòng logic dựa vào toạ độ y.
    """
    if df.empty:
        return []
    df = df.sort_values(["y", "x"]).reset_index(drop=True)
    lines, cur = [], [df.iloc[0]]
    for i in range(1, len(df)):
        prev = cur[-1]
        row  = df.iloc[i]
        if abs(row["y"] - prev["y"]) <= y_tol:
            cur.append(row)
        else:
            lines.append(pd.DataFrame(cur))
            cur = [row]
    if cur:
        lines.append(pd.DataFrame(cur))
    return lines


def _merge_numeric_runs(texts, xs, gap_px=110):
    """
    Ghép các token số đứng cạnh nhau thành 1 số đầy đủ.
    Bỏ lọc mạnh, chỉ loại token không phải NUMISH.
    """
    items = sorted([(x, t) for t, x in zip(texts, xs)], key=lambda z: z[0])
    out, buf, bx, prev_x = [], "", None, None

    def flush():
        nonlocal buf, bx
        if not buf:
            return
        raw = re.sub(r'[^0-9\.\,\(\)\-\s]', '', buf)
        if re.search(r'\d', raw):
            out.append((bx, raw.strip()))
        buf, bx = "", None

    for x, t in items:
        t = (t or "").strip()
        if not t:
            continue
        if not NUMISH.match(t):
            flush()
            prev_x = x
            continue

        if buf == "":
            buf, bx = t, x
        else:
            near = prev_x is not None and (x - prev_x) <= gap_px
            thousand_glue = buf.rstrip().endswith((".", ",")) or re.search(r"[\.\,]\s*$", buf)
            only_2_3 = len(re.sub(r"\D", "", t)) in (2, 3)
            if near or (thousand_glue and only_2_3):
                buf += t
            else:
                flush()
                buf, bx = t, x
        prev_x = x

    flush()
    return out


def _debug_dump_ocr_lines(words: pd.DataFrame, pageno1: int):
    """
    Log toàn bộ dòng OCR đọc được (trước khi cắt header/footer), để anh check
    OCR & bộ lọc.
    """
    if not DEBUG_OCR_LINES or words.empty:
        return
    print(f"\n===== OCR RAW LINES – page {pageno1} =====")
    for ln in _group_lines(words, y_tol=8):
        y0 = int(ln["y"].min())
        text = " ".join(str(t) for t in ln["text"].tolist())
        print(f"[p{pageno1:03d} y~{y0:04d}] {text}")


# ---------- pdfplumber / camelot ----------
def _extract_plumber_tables(pdf_path: Path, pageno1: int) -> List[pd.DataFrame]:
    """
    Cố gắng lấy tối đa bảng vector từ pdfplumber.
    Bộ lọc nhẹ: chỉ bỏ bảng 1 cột hoặc toàn rỗng.
    """
    out: List[pd.DataFrame] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[pageno1 - 1]

            # 1) find_tables (khung mảnh)
            for tb in (page.find_tables() or []):
                df = pd.DataFrame(tb.extract()).fillna("").replace("\n", " ", regex=True)
                if df.shape[1] >= 2:
                    df = df.loc[:, ~(df.astype(str).apply(lambda s: (s.str.strip() == "").all()))]
                    if df.shape[1] >= 2:
                        out.append(df)

            # 2) lines-based
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 6,
                "snap_tolerance": 3,
                "join_tolerance": 3,
            }) or []

            # 3) text-based (không có khung)
            tables += page.extract_tables(table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "text_y_tolerance": 3,
                "text_x_tolerance": 2,
                "intersection_tolerance": 3,
                "snap_tolerance": 3,
                "join_tolerance": 3,
            }) or []

            for tb in tables:
                df = pd.DataFrame(tb).fillna("").replace("\n", " ", regex=True)
                if df.shape[1] >= 2:
                    df = df.loc[:, ~(df.astype(str).apply(lambda s: (s.str.strip() == "").all()))]
                    if df.shape[1] >= 2:
                        out.append(df)
    except Exception:
        pass
    return out


def _extract_camelot_tables(pdf_path: Path, pageno1: int) -> List[pd.DataFrame]:
    """
    Camelot stream-mode cho bảng text nhiều đường kẻ mảnh.
    Không filter gì ngoài số cột.
    """
    out: List[pd.DataFrame] = []
    try:
        tables = camelot.read_pdf(str(pdf_path), pages=str(pageno1), flavor="stream")
        for t in tables:
            df = t.df.replace("\n", " ", regex=True)
            if df.shape[1] >= 2:
                out.append(df)
    except Exception:
        pass
    return out


# ---------- multi-column OCR (header + numeric clusters) ----------
def _header_column_centers(words_df: pd.DataFrame) -> List[int]:
    """
    Ước lượng vị trí cột từ vùng header (text).
    Chỉ dùng để hỗ trợ, không quyết định chính.
    """
    if words_df.empty:
        return []
    y_min, y_max = words_df["y"].min(), words_df["y"].max()
    header = words_df[words_df["y"] <= (y_min + 0.30 * (y_max - y_min))].copy()
    if header.empty:
        return []
    xs = []
    for _, r in header.iterrows():
        t = (r["text"] or "").strip()
        tl = t.lower()
        if NUM_ANY.fullmatch(t):
            continue
        if len(t) < 3:
            continue
        if any(k in tl for k in ["đơn vị", "don vi", "vnd", "cột", "cot", "mục", "muc"]):
            continue
        xs.append(int(r["x"]))
    xs.sort()
    centers = []
    for x in xs:
        if not centers or abs(x - centers[-1]) > 55:
            centers.append(x)
        else:
            centers[-1] = int((centers[-1] + x) / 2)
    return centers


def _max_big_numbers_per_line(words_df: pd.DataFrame) -> int:
    """
    Đếm số lượng giá trị số "lớn" (>=5 chữ số) tối đa trên 1 dòng.
    Dùng để phân biệt trang text vs trang bảng.
    """
    if words_df.empty:
        return 0
    lines = _group_lines(words_df, y_tol=8)
    mx = 0
    for ln in lines:
        texts = ln["text"].tolist()
        xs    = ln["x"].tolist()
        merged = _merge_numeric_runs(texts, xs, gap_px=110)
        cnt = 0
        for _x, raw in merged:
            if len(re.sub(r"[^\d]", "", str(raw))) >= 5:
                cnt += 1
        mx = max(mx, cnt)
    return mx


def _extract_header_text_near(words_df: pd.DataFrame, x: int, tol: int = 200) -> str:
    if words_df.empty:
        return ""
    y_min, y_max = words_df["y"].min(), words_df["y"].max()
    y_cut = y_min + 0.20 * (y_max - y_min)
    header = words_df[words_df["y"] <= y_cut]
    col_words = header[(header["x"] >= x - tol) & (header["x"] <= x + tol)]
    return " ".join(col_words["text"].tolist()).strip().lower()


def _normalize_context_from_header(txt: str) -> Tuple[str, str]:
    """
    Chuyển header text -> context_key + context_label.
    Bộ nhận diện đơn giản, đủ dùng cho notes.
    """
    t = (txt or "").lower()

    m = re.search(r'qu[ýy]\s*(\d+)\s*[\/\-]?\s*(20\d{2})', t)
    if m:
        return (f"q{m.group(1)}_{m.group(2)}", f"Quý {m.group(1)}/{m.group(2)}")

    if any(k in t for k in ["kỳ kế toán", "ky ke toan", "lũy kế", "luy ke", "6 tháng", "6 thang"]):
        y = re.search(r'(20\d{2})', t)
        return (f"ytd_{y.group(1) if y else 'unk'}", "YTD")

    dm = DATE_RE.findall(t)
    if dm:
        y = re.search(r'(20\d{2})', dm[-1])
        return (f"asof_{y.group(1) if y else 'unk'}", f"As of {dm[-1]}")

    if "đầu kỳ" in t or "opening" in t:
        return ("opening", "Đầu kỳ")
    if "cuối kỳ" in t or "closing" in t:
        return ("closing", "Cuối kỳ")

    # fallback: giữ nguyên text để parser suy luận tiếp
    return ("col", txt or "")


def _cluster_numeric_columns(words_df: pd.DataFrame, max_k: int = 8) -> List[int]:
    """
    Gom cụm các toạ độ x của số liệu → suy ra vị trí cột.

    Sửa:
      - Dùng _max_big_numbers_per_line để ước lượng số cột tối thiểu (min_k),
        tránh case k quá nhỏ (ví dụ: chọn 2 cột trong khi 1 dòng có 6–7 số).
      - Vẫn lọc cột yếu: mỗi center phải có đủ số điểm (>=3) mới chấp nhận.
    """
    if words_df.empty:
        return []

    # gom toàn bộ số lớn theo line
    xs_points: List[int] = []
    for ln in _group_lines(words_df, y_tol=8):
        texts = ln["text"].tolist()
        xs    = ln["x"].tolist()
        merged = _merge_numeric_runs(texts, xs, gap_px=110)
        for x, raw in merged:
            digits = re.sub(r"[^\d]", "", str(raw))
            if len(digits) >= 5:  # chỉ coi là số liệu tiền / số lượng lớn
                xs_points.append(int(x))

    if not xs_points:
        return []

    X = np.array(xs_points).reshape(-1, 1)

    # ước lượng số cột tối thiểu từ dòng "giàu" số nhất
    max_big = _max_big_numbers_per_line(words_df)
    # ít nhất 2 cột, nhiều nhất max_k
    start_k = max(2, min(max_big, max_k))
    end_k   = min(max_k, len(X))

    best_centers, best_score = None, -1.0
    for k in range(start_k, end_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
            if len(set(km.labels_)) < 2:
                continue
            sc = silhouette_score(X, km.labels_)
            # ưu tiên k lớn hơn nếu score không tệ hơn quá 0.02
            prefer = (best_centers is not None and k > len(best_centers) and sc >= best_score - 0.02)
            if sc > best_score or prefer:
                best_score = sc
                best_centers = [int(c[0]) for c in km.cluster_centers_]
        except Exception:
            continue

    centers_num = sorted(best_centers) if best_centers else []

    # Lọc các center yếu: mỗi center phải có >=3 điểm số lớn
    strong_centers: List[int] = []
    for cx in centers_num:
        cnt = sum(1 for x in xs_points if abs(x - cx) <= 40)
        if cnt >= 3:
            strong_centers.append(int(cx))

    centers_hdr = _header_column_centers(words_df)

    # Gộp 2 nguồn (numeric + header)
    all_c = sorted(strong_centers + centers_hdr)
    merged: List[int] = []
    for x in all_c:
        if not merged or abs(x - merged[-1]) > 55:
            merged.append(int(x))
        else:
            merged[-1] = int((merged[-1] + x) // 2)

    if len(merged) < 2:
        return merged
    return merged[:max_k]

def _infer_multi_columns(words_df: pd.DataFrame) -> List[Dict]:
    """
    Trả list [{x, context_key, context_label}] từ trái sang phải.
    Không có cột nào → trả [].
    """
    centers = _cluster_numeric_columns(words_df, max_k=8)
    if not centers or len(centers) < 2:
        return []

    cols: List[Dict] = []
    used = set()
    for idx, x in enumerate(sorted(centers), 1):
        h = _extract_header_text_near(words_df, x)
        key, lbl = _normalize_context_from_header(h)
        if key == "col":
            key = f"col{idx}"
        if key in used:
            key = f"{key}_{idx}"
        used.add(key)
        cols.append({"x": int(x), "context_key": key, "context_label": lbl})
    return cols


def _parse_line_with_multi_columns(line_df: pd.DataFrame, cols: List[Dict], tol=260):
    """
    Từ 1 dòng OCR + danh sách cột (toạ độ x) → label + dict context_key -> raw_number.

    Sửa:
      - Vẫn dùng vùng bên phải để lấy số liệu (tránh số nằm trong STT, năm...).
      - Label được ghép từ TẤT CẢ token không phải số trên dòng (trừ VND/VNĐ),
        không giới hạn chỉ ở vùng bên trái → xử lý tốt case mô tả nằm bên phải
        như "Giá trị gia tăng", "Thu nhập doanh nghiệp", ...
    """
    if line_df.empty or not cols:
        return "", {}

    texts = line_df["text"].tolist()
    xs    = line_df["x"].tolist()
    ws    = line_df["w"].tolist() if "w" in line_df.columns else [20] * len(xs)

    left_most_col = min(c["x"] for c in cols)
    median_w = int(np.median([w for w in ws if w and w > 0])) if ws else 24
    margin   = max(30, int(1.5 * median_w))

    # Vùng label để LOẠI SỐ (số ở quá bên trái thường là STT, năm...)
    label_threshold_x = left_most_col - margin

    # --- 1) Gom số liệu ---
    merged = _merge_numeric_runs(texts, xs, gap_px=110)
    nums: List[Tuple[int, str]] = []
    for x, raw in merged:
        digits = re.sub(r"[^\d]", "", raw)
        # Bỏ toàn bộ số nằm trong vùng label (thường là số mục, số năm)
        if x < label_threshold_x:
            continue
        # Nhận tất cả số có >=3 chữ số (giữ EPS / số lượng nhỏ trong bảng)
        if len(digits) >= 3:
            nums.append((int(x), raw))

    if not nums:
        return "", {}

    # --- 2) Ghép label từ tất cả token không phải số ---
    label_tokens: List[str] = []
    for t in texts:
        t_clean = (t or "").strip()
        if not t_clean:
            continue
        tl = t_clean.lower()
        if tl in {"vnd", "vnđ"}:
            # bỏ đơn vị tiền ở cuối dòng
            continue
        if NUMISH.match(t_clean):
            # giống số → không đưa vào label
            continue
        label_tokens.append(t_clean)

    label = " ".join(label_tokens).strip()

    cols_sorted = sorted([(c["x"], c["context_key"]) for c in cols], key=lambda z: z[0])
    nums_sorted = sorted([(x, s) for x, s in nums], key=lambda z: z[0])

    def assign_once(limit: int):
        assigned, used_num = {}, set()
        pairs = []
        for ci, (cx, ck) in enumerate(cols_sorted):
            for ni, (nx, sv) in enumerate(nums_sorted):
                d = abs(nx - cx)
                if d <= limit:
                    pairs.append((d, ci, ni))
        for d, ci, ni in sorted(pairs, key=lambda z: z[0]):
            if ci in assigned or ni in used_num:
                continue
            assigned[ci] = ni
            used_num.add(ni)
        return assigned

    assigned = assign_once(tol)
    if len(assigned) < len(cols_sorted):
        assigned = assign_once(int(2.2 * tol))

    values: Dict[str, str] = {}

    # fallback: nếu không match toạ độ nhưng số lượng gần bằng số cột ⇒ gán tuần tự
    if len(assigned) == 0 and 1 <= len(nums_sorted) <= len(cols_sorted) + 1:
        for (cx, ck), (_nx, sv) in zip(cols_sorted, nums_sorted):
            values[ck] = sv
    else:
        for ci, (cx, ck) in enumerate(cols_sorted):
            if ci in assigned:
                ni = assigned[ci]
                values[ck] = nums_sorted[ni][1]

    return label, values


def _page_is_tabular(words: pd.DataFrame) -> bool:
    """
    Heuristic mạnh để phân biệt trang thuyết minh dạng text vs trang có bảng số liệu:
      - Phải có ít nhất 2 cột số lớn (>=5 chữ số)
      - Mỗi cột có >=3 điểm số.
    """
    if words.empty:
        return False
    if _max_big_numbers_per_line(words) < 2:
        # mỗi dòng tối đa <2 số lớn → nhiều khả năng chỉ là text, không phải bảng
        return False
    centers = _cluster_numeric_columns(words, max_k=8)
    return len(centers) >= 2


def _extract_ocr_table(pdf_path: Path, pageno1: int) -> Optional[pd.DataFrame]:
    """
    Dò bảng bằng OCR cho 1 trang:
      - Log toàn bộ dòng OCR đọc được (DEBUG_OCR_LINES)
      - Cắt header/footer
      - Suy đoán nhiều cột số từ số lớn (>=5 chữ số)
      - Parse từng dòng -> DataFrame(desc, col_1_label, col_2_label,...)

    Các trang thuyết minh thuần text (chính sách kế toán...) sẽ bị loại
    vì không đạt tiêu chí “trang bảng”.
    """
    img = _ocr_image(pdf_path, pageno1)
    words = _ocr_words(img)
    if words.empty:
        return None

    # Log raw OCR lines trước mọi filter
    _debug_dump_ocr_lines(words, pageno1)

    # Cắt phần giữa trang (bỏ header/footer mạnh chữ "năm yyyy")
    y_min, y_max = words["y"].min(), words["y"].max()
    height = max(1, y_max - y_min)
    work = words[(words["y"] >= y_min + 0.08 * height) &
                 (words["y"] <= y_max - 0.12 * height)].copy()
    if work.empty:
        work = words

    # Nếu trang không có cấu trúc bảng rõ ràng → bỏ
    if not _page_is_tabular(work):
        print(f"🔎 [OCR] page {pageno1}: looks like narrative text, skip as table.")
        return None

    cols = _infer_multi_columns(work)

    # Fallback: nếu vẫn không suy ra được multi-column thì thôi, không coi là bảng OCR
    if not cols:
        print(f"⚠️ [OCR] page {pageno1}: cannot infer numeric columns, skip.")
        return None

    # Tên cột hiển thị (header) – dùng context_label nếu có, ngược lại fallback "col1.."
    header_names: List[str] = ["desc"]
    for idx, c in enumerate(cols, 1):
        label = c["context_label"] or c["context_key"] or f"col{idx}"
        label = label.strip() or f"col{idx}"
        header_names.append(label)

    lines = _group_lines(work, y_tol=8)

    recs: List[dict] = []
    last_label = ""
    for ln in lines:
        label, vals = _parse_line_with_multi_columns(ln, cols)
        if not label:
            label = last_label
        if not vals:
            # dòng không có số → có thể là heading, bỏ qua ở đây
            continue
        if not label:
            label = "(no_label)"
        last_label = label

        row: Dict[str, Optional[str]] = {header_names[0]: label}
        for j, c in enumerate(cols, 1):
            ctx_key = c["context_key"]
            col_name = header_names[j]
            row[col_name] = vals.get(ctx_key)
        recs.append(row)

    if not recs:
        return None

    df = pd.DataFrame(recs)
    # đảm bảo thứ tự cột ổn định: desc, col1, col2, ...
    df = df.reindex(columns=header_names)
    df = df.drop_duplicates()
    return df


def harvest_tables(pdf_path: Path, a: int, b: int) -> List[Tuple[pd.DataFrame, int, str]]:
    """
    Quét [a,b] → [(df, page, mode)] với mode ∈ {"vector","camelot","ocr"}.
    - Luôn cố gắng lấy *hết* bảng (kể cả bảng không khung).
    - OCR luôn chạy bổ sung, không phụ thuộc đã có vector hay chưa.
    - Các trang thuyết minh text thuần túy sẽ bị _extract_ocr_table bỏ qua.
    """
    out: List[Tuple[pd.DataFrame, int, str]] = []
    for p in range(a, b + 1):
        # pdfplumber vector
        for df in _extract_plumber_tables(pdf_path, p):
            out.append((df, p, "vector"))
        # camelot stream
        for df in _extract_camelot_tables(pdf_path, p):
            out.append((df, p, "camelot"))
        # OCR multi-column
        mdf = _extract_ocr_table(pdf_path, p)
        if mdf is not None and not mdf.empty:
            out.append((mdf, p, "ocr"))
    return out
