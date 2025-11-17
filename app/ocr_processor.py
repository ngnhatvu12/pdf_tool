import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import os, re, traceback
from app.etl.normalize import normalize_with_unit
from app.etl.table_finder import detect_unit, locate_statement_pages
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 

# THÊM IMPORT PADDLEOCR
os.environ.setdefault("FLAGS_use_mkldnn", "0")   # tránh bug mkldnn + CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")    # hạn chế multi-thread inference
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

_paddle_ocr = None

def _get_paddle_ocr():
    """
    Khởi tạo PaddleOCR lazy (chỉ tạo 1 lần cho cả process).
    """
    global _paddle_ocr
    if _paddle_ocr is None:
        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR chưa được cài. Hãy chạy: pip install paddleocr"
            )
        # lang='vi' tốt cho báo cáo tiếng Việt, kèm tiếng Anh vẫn ổn
        _paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang="vi"
        )
    return _paddle_ocr


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH   = r"C:\poppler\poppler-25.07.0\Library\bin"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"✅ Đã cấu hình Tesseract: {TESSERACT_PATH}")

NUM_RE = re.compile(r"\(?[-+]?\s*(?:\d{1,3}(?:[.\s]\d{3})+|\d+)(?:,\d+|\.\d+)?\)?")
NUMISH_RE = re.compile(r'^[\d\.\,\(\)\-\s]+$')

_vie_path = Path(os.environ['TESSDATA_PREFIX']) / "vie.traineddata"
if not _vie_path.exists():
    print(f"⚠️ Không tìm thấy {_vie_path}. Hãy cài/chép 'vie.traineddata' vào thư mục tessdata.")
else:
    print(f"✅ Đã trỏ TESSDATA_PREFIX đến: {os.environ['TESSDATA_PREFIX']}")
    
def _merge_numeric_runs(texts, xs, gap_px: int = 60):
    """
    Ghép các token số đứng liền nhau thành 1 số đầy đủ.
    - Nếu 2 token gần nhau (<= gap_px) -> gộp
    - Nếu token trước kết thúc '.' hoặc ',' và token sau là 2–3 chữ số -> gộp (nhóm nghìn)
    - Hỗ trợ chuỗi như: '9.023', '989', '480', '000' (có/không có dấu)
    Trả về: [(x_left, clean_for_parse, raw_concat)] theo x tăng dần.
    """
    items = sorted([(x, t) for t, x in zip(texts, xs)], key=lambda z: z[0])
    out = []
    buf_raw, buf_x = "", None
    prev_x = None

    def _flush():
        nonlocal buf_raw, buf_x
        if not buf_raw:
            return
        # bỏ mọi ký tự lạ ngoại trừ . , ( ) - và số
        raw = re.sub(r'[^0-9\.\,\(\)\-\s]', '', buf_raw)
        # nếu toàn dấu thì bỏ
        if not re.search(r'\d', raw):
            buf_raw, buf_x = "", None
            return
        clean = raw
        out.append((buf_x, clean.strip(), raw.strip()))
        buf_raw, buf_x = "", None

    for x, t in items:
        t = t.strip()
        if not t:
            continue
        if not NUMISH_RE.match(t):
            _flush()
            prev_x = x
            continue

        if buf_raw == "":
            buf_raw = t
            buf_x = x
        else:
            near = prev_x is not None and (x - prev_x) <= gap_px
            # trường hợp dính nhóm nghìn
            thousand_glue = buf_raw.rstrip().endswith((".", ",")) or re.search(r"[\.\,]\s*$", buf_raw)
            only_2_3_digits = len(re.sub(r"\D", "", t)) in (2, 3)
            if near or (thousand_glue and only_2_3_digits):
                buf_raw += t
            else:
                _flush()
                buf_raw = t
                buf_x = x
        prev_x = x
    _flush()
    # loại bớt chuỗi quá ngắn (ít hơn 6 chữ số → hay là năm, số trang…)
    filtered = []
    for x, clean, raw in out:
        digits = re.sub(r"\D", "", clean)
        if len(digits) >= 6:  # bắt đầu từ hàng trăm nghìn trở lên
            filtered.append((x, clean, raw))
    return filtered

def pdf_to_images(pdf_path: Path, dpi=300) -> List:
    if os.path.exists(POPPLER_PATH):
        return convert_from_path(str(pdf_path), dpi=dpi, poppler_path=POPPLER_PATH)
    return convert_from_path(str(pdf_path), dpi=dpi)

def _ocr_words_tesseract(image) -> pd.DataFrame:
    cfg = r'--oem 3 --psm 6 -l vie+eng -c preserve_interword_spaces=1'
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=cfg)
    rows = []
    for i, txt in enumerate(d["text"]):
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1.0
        txt_clean = (txt or "").strip()
        if not txt_clean or len(txt_clean) < 1:
            continue
        if conf < 40:
            continue
        rows.append({
            "text": txt_clean,
            "x": d["left"][i],
            "y": d["top"][i],
            "w": d["width"][i],
            "h": d["height"][i],
            "conf": conf
        })
    return pd.DataFrame(rows)

def _ocr_words_paddle(image) -> pd.DataFrame:
    """
    OCR bằng PaddleOCR, trả về DataFrame cùng schema với Tesseract:
    columns: text, x, y, w, h, conf
    """
    ocr = _get_paddle_ocr()
    img = np.array(image)

    try:
        # với paddleocr 3.x mới, ocr() đã là pipeline chính
        result = ocr.ocr(img)   # KHÔNG truyền cls=...
    except Exception as e:
        print("🔥 PaddleOCR gặp lỗi không mong muốn:", repr(e))
        traceback.print_exc()
        # Không fallback sang tesseract – fail thẳng để bạn biết là Paddle đang lỗi
        raise RuntimeError("PaddleOCR inference failed") from e

    rows = []

    # result là list theo page; ảnh đơn thường là: [[(box, (txt, conf)), ...]]
    for line in result:
        for box, (txt, conf) in line:
            txt = (txt or "").strip()
            if not txt:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            w = x1 - x0
            h = y1 - y0
            rows.append({
                "text": txt,
                "x": int(x0),
                "y": int(y0),
                "w": int(w),
                "h": int(h),
                "conf": float(conf),
            })
    return pd.DataFrame(rows)


def ocr_words(image, engine: str = "tesseract") -> pd.DataFrame:
    engine = (engine or "tesseract").lower()
    if engine == "paddle":
        return _ocr_words_paddle(image)
    return _ocr_words_tesseract(image)

def _group_lines(df: pd.DataFrame, y_tol=8) -> List[pd.DataFrame]:
    """Gom dòng chính xác hơn"""
    if df.empty: return []
    df = df.sort_values(["y","x"]).reset_index(drop=True)
    lines, cur = [], [df.iloc[0]]
    for i in range(1, len(df)):
        prev = cur[-1]; row = df.iloc[i]
        if abs(row["y"] - prev["y"]) <= y_tol:
            cur.append(row)
        else:
            lines.append(pd.DataFrame(cur))
            cur = [row]
    if cur: lines.append(pd.DataFrame(cur))
    return lines

HEADER_CUR = re.compile(r"(năm\s*nay|số\s*cuối\s*kỳ|kỳ\s*này)", re.I|re.U)
HEADER_PRI = re.compile(r"(năm\s*trước|số\s*đầu\s*năm|kỳ\s*trước)", re.I|re.U)

def _peak_columns(nums_x: List[int]) -> List[int]:
    """Trả về ~tọa độ trung tâm 2 cột số (x) từ danh sách x của các tokens số trong trang."""
    if len(nums_x) < 6:
        return sorted(set(nums_x))[-2:] if len(set(nums_x)) >= 2 else sorted(set(nums_x))
    xs = np.array(sorted(nums_x))
    # K-means K=2 theo 1D x
    c1, c2 = xs.min(), xs.max()
    for _ in range(12):
        g1 = xs[np.abs(xs - c1) <= np.abs(xs - c2)]
        g2 = xs[np.abs(xs - c2) <  np.abs(xs - c1)]
        nc1 = float(g1.mean()) if len(g1) else c1
        nc2 = float(g2.mean()) if len(g2) else c2
        if abs(nc1-c1)+abs(nc2-c2) < 1.0: break
        c1, c2 = nc1, nc2
    return sorted([int(c1), int(c2)])

def _infer_col_roles(words_df: pd.DataFrame) -> Dict[str,int]:
    """Xác định cột với logic cải tiến cho bảng không khung"""
    if words_df.empty: 
        return {"current_x": None, "prior_x": None}
    
    # 1) Tìm tất cả các token số và lọc nhiễu
    num_df = words_df[words_df["text"].str.match(NUM_RE, na=False)].copy()
    if num_df.empty:
        return {"current_x": None, "prior_x": None}
    
    # Lọc nhiễu: loại bỏ số quá ngắn hoặc không hợp lệ
    def is_valid_number(txt):
        clean = re.sub(r'[^\d]', '', txt)
        return len(clean) >= 6  # Chỉ lấy số có ít nhất 6 chữ số
    
    num_df = num_df[num_df["text"].apply(is_valid_number)]
    if num_df.empty:
        return {"current_x": None, "prior_x": None}
    
    # 2) Gom cụm theo cột
    xs = np.array(num_df["x"].tolist())
    if len(xs) < 4:
        unique_xs = sorted(set(xs))
        if len(unique_xs) >= 2:
            left_x, right_x = unique_xs[0], unique_xs[-1]
        else:
            return {"current_x": None, "prior_x": None}
    else:
        try:
            # Sử dụng K-means với 2 cụm
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            kmeans.fit(xs.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            left_x, right_x = int(centers[0]), int(centers[1])
        except:
            # Fallback: lấy min và max
            left_x, right_x = int(xs.min()), int(xs.max())
    
    # 3) Xác định current/prior dựa trên header
    y_cut = words_df["y"].min() + 0.12 * (words_df["y"].max() - words_df["y"].min())
    header = words_df[words_df["y"] <= y_cut]
    
    def get_col_text(col_x, tol=120):
        col_words = header[
            (header["x"] >= col_x - tol) & 
            (header["x"] <= col_x + tol)
        ]
        return " ".join(col_words["text"].tolist()).lower()
    
    left_text = get_col_text(left_x)
    right_text = get_col_text(right_x)
    
    print(f"   Header left: '{left_text}'")
    print(f"   Header right: '{right_text}'")
    
    # 4) Quyết định mapping dựa trên ngày tháng rõ ràng
    left_has_cur = any(date_str in left_text for date_str in ["30/06/2025", "30/6/2025", "3062025"])
    right_has_cur = any(date_str in right_text for date_str in ["30/06/2025", "30/6/2025", "3062025"])
    left_has_pri = any(date_str in left_text for date_str in ["01/01/2025", "1/1/2025", "01012025"])
    right_has_pri = any(date_str in right_text for date_str in ["01/01/2025", "1/1/2025", "01012025"])
    
    # Ưu tiên các pattern rõ ràng
    if left_has_cur and right_has_pri:
        return {"current_x": left_x, "prior_x": right_x}
    elif right_has_cur and left_has_pri:
        return {"current_x": right_x, "prior_x": left_x}
    elif "30/06/2025" in left_text or "30/6/2025" in left_text:
        return {"current_x": left_x, "prior_x": right_x}
    elif "30/06/2025" in right_text or "30/6/2025" in right_text:
        return {"current_x": right_x, "prior_x": left_x}
    else:
        # Fallback: dựa vào vị trí (thường current bên trái trong VAS)
        return {"current_x": left_x, "prior_x": right_x}

def _parse_line_with_columns(line_df: pd.DataFrame, current_x: int, prior_x: int, tol=100):
    texts, xs = line_df["text"].tolist(), line_df["x"].tolist()

    # GHÉP SỐ bị tách
    merged = _merge_numeric_runs(texts, xs, gap_px=60)
    nums = []
    for x, clean_text, raw in merged:
        if clean_text and len(re.sub(r'[^\d]', '', clean_text)) >= 4:
            nums.append((x, clean_text, raw))
    if not nums:
        return "", None, None

    # label mềm hơn
    first_num_x = min(x for x, _n, _r in nums)
    threshold_x = min(first_num_x, current_x, prior_x) - 20
    label_tokens = [t for t, x in zip(texts, xs) if x < threshold_x and not NUMISH_RE.match(t)]
    if not label_tokens:
        label_tokens = [t for t, x in zip(texts, xs) if x < threshold_x]
    label = " ".join(label_tokens).strip()

    def find_best_match(target_x, numbers, tolerance=tol):
        if not numbers:
            return None
        distances = [(abs(x - target_x), num) for x, num, _r in numbers]
        distances.sort(key=lambda z: z[0])
        if distances and distances[0][0] <= tolerance:
            return distances[0][1]
        return None

    cur_raw = find_best_match(current_x, nums) or find_best_match(current_x, nums, tol * 1.6)
    pri_raw = find_best_match(prior_x, nums) or find_best_match(prior_x, nums, tol * 1.6)

    if label and (cur_raw or pri_raw):
        print(f"     Line: '{label[:50]}...' -> cur: {cur_raw}, prior: {pri_raw}")

    return label, pri_raw, cur_raw

def extract_tables_ocr(pdf_path: Path,pages: Optional[List[int]] = None,ocr_engine: str = "tesseract") -> pd.DataFrame:
    """
    Trả về DataFrame long-form:
        page, statement_hint, vas_code, src_label, context_key, context_label, amount, unit_hint, confidence
    Backward-compatible: nếu chỉ bắt được đúng 2 cột kiểu "current/prior" thì vẫn gán context_key tương ứng.
    """
    page_ranges = locate_statement_pages(pdf_path, max_pages_to_scan=20)
    section_by_page = {}
    for section, (start, end) in page_ranges.items():
        for page_num in range(start, end + 1):
            section_by_page[page_num] = section

    page_whitelist = set(pages or section_by_page.keys())
    images = pdf_to_images(pdf_path)

    all_rows = []
    print(f"📖 Tổng số trang PDF: {len(images)}")
    print(f"🔍 Quét các trang: {sorted(page_whitelist)}")

    for pageno1, img in enumerate(images, 1):
        if pageno1 not in page_whitelist:
            continue

        print(f"\n🔍 Đang xử lý trang {pageno1}...")
        words = ocr_words(img, engine=ocr_engine)
        if words.empty:
            print(f"   ❌ Không tìm thấy text trên trang {pageno1}")
            continue
        words = _strip_page_headers(words)
        if words.empty:
            continue
        unit_hint = detect_unit(" ".join(words["text"].tolist())) or "VND"
        section = section_by_page.get(pageno1, "UNKNOWN")

        # ==== NEW: phát hiện nhiều cột ====
        cols = _infer_multi_columns(words)

        # Fallback cũ: nếu không ra nhiều cột, thử logic 2 cột
        if not cols:
            two = _infer_col_roles(words)  # hàm cũ
            if not two["current_x"] or not two["prior_x"]:
                print(f"   ❌ Không xác định được vị trí cột trên trang {pageno1}")
                continue
            cols = [
                {"x": int(two["prior_x"]),   "context_key": "prior_period",   "context_label": "Prior"},
                {"x": int(two["current_x"]), "context_key": "current_period", "context_label": "Current"},
            ]
        cols = sorted(cols, key=lambda c: c["x"])

        lines = _group_lines(words)
        print(f"   📄 Tìm thấy {len(lines)} dòng | {len(cols)} cột")

        page_rows = 0
        for line in lines:
            label, values = _parse_line_with_multi_columns(line, cols)
            if not label:
                continue

            # VAS code tách từ đầu nhãn nếu có
            vas = None
            m = re.match(r"^\s*(\d{2,3}[A-Z]?)\s*[-.:]?\s*", label)
            if m:
                vas = m.group(1)
                label = re.sub(r"^\s*\d{2,3}[A-Z]?\s*[-.:]?\s*", "", label).strip()

            # loại nhiễu label toàn số
            digit_ratio = sum(c.isdigit() for c in label) / max(1, len(label))
            if (digit_ratio > 0.6 and
                not any(k in label.lower() for k in ["tổng", "cộng", "total", "100", "200", "300", "400"])
            ):
                continue

            # chuẩn hoá & ghi ra long-form
            for c in cols:
                raw = values.get(c["context_key"])
                if raw is None:
                    continue
                amt_vnd, _ = normalize_with_unit(raw, unit_hint)
                if amt_vnd is None:
                    continue

                all_rows.append(dict(
                    page=pageno1,
                    statement_hint=section,
                    vas_code=vas,
                    src_label=label,
                    context_key=c["context_key"],
                    context_label=c["context_label"],
                    amount=amt_vnd,
                    unit_hint=unit_hint,
                    confidence=0.9 if vas else 0.8
                ))
                page_rows += 1

        print(f"   ✅ Trích xuất được {page_rows} ô số liệu từ trang {pageno1}")

    df = pd.DataFrame(all_rows)
    print(f"\n📊 Tổng cộng trích xuất được {len(df)} ô số liệu ở dạng long-form")
    return df

def _cluster_numeric_columns(words_df: pd.DataFrame, max_k: int = 5) -> List[int]:
    """
    Gom cụm các toạ độ x của số liệu để suy ra số cột.
    Trả về danh sách center x (int) từ trái -> phải.
    """
    num_df = words_df[words_df["text"].str.match(NUM_RE, na=False)].copy()
    if num_df.empty:
        return []
    # lọc số "đủ dài" để tránh nhiễu
    def is_valid_number(txt):
        clean = re.sub(r'[^\d]', '', txt)
        return len(clean) >= 6
    xs = np.array([x for x, t in zip(num_df["x"].tolist(), num_df["text"].tolist()) if is_valid_number(t)])
    if len(xs) < 3:
        return sorted(list(set(map(int, xs))))
    xs = xs.reshape(-1, 1)

    best_k, best_score, best_centers = None, -1, None
    for k in range(2, max(2, min(max_k, len(xs)) + 1)):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(xs)
            if len(set(km.labels_)) < 2:  # silhouette cần >=2 cụm
                continue
            sc = silhouette_score(xs, km.labels_)
            if sc > best_score:
                best_score = sc
                best_k = k
                best_centers = sorted([int(c[0]) for c in km.cluster_centers_])
        except Exception:
            continue

    if best_centers:
        return best_centers
    # fallback
    uniq = sorted(list(set([int(v[0]) for v in xs])))
    return uniq[:max_k]

def _extract_header_text_near(words_df: pd.DataFrame, x: int, tol: int = 140) -> str:
    """Lấy text của vùng header gần tâm cột x."""
    if words_df.empty:
        return ""
    y_min, y_max = words_df["y"].min(), words_df["y"].max()
    y_cut = y_min + 0.20 * (y_max - y_min)  
    header = words_df[words_df["y"] <= y_cut]
    col_words = header[(header["x"] >= x - tol) & (header["x"] <= x + tol)]
    return " ".join(col_words["text"].tolist()).strip().lower()

DATE_RE = re.compile(r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})')

def _normalize_context_from_header(txt: str) -> Tuple[str, str]:
    """
    Quy đổi header text -> context_key (machine-friendly) và context_label (giữ nguyên để trace).
    Ưu tiên nhận diện:
        - Quý: 'q2/2025', 'quý 2/2025'
        - Lũy kế/kỳ kế toán: '01/01/2025 đến 30/06/2025' => 'ytd_2025'
    Nếu mơ hồ: trả 'col1', 'col2', ...
    """
    t = (txt or "").lower()
    # quarterly
    m = re.search(r'qu[ýy]\s*(\d+)\s*[\/\-]?\s*(20\d{2})', t)
    if m:
        q, y = m.group(1), m.group(2)
        return (f"q{q}_{y}", f"Quý {q}/{y}")
    m = re.search(r'(q|quy)\s*(\d+)\s*[\/\-]?\s*(20\d{2})', t)
    if m:
        q, y = m.group(2), m.group(3)
        return (f"q{q}_{y}", f"Quý {q}/{y}")
    # ytd / kỳ kế toán / lũy kế 6T
    if "kỳ kế toán" in t or "ky ke toan" in t or "lũy kế" in t or "luy ke" in t or "6 tháng" in t or "6 thang" in t:
        # cố gắng bắt năm: lấy năm xuất hiện trong chuỗi
        y = None
        ym = re.search(r'(20\d{2})', t)
        if ym: y = ym.group(1)
        return (f"ytd_{y or 'unk'}", f"YTD {y or ''}".strip())
    # nếu có ngày kết thúc kiểu dd/mm/yyyy -> dùng năm làm context
    dm = DATE_RE.findall(t)
    if dm:
        # lấy năm của ngày cuối
        y = re.search(r'(20\d{2})', dm[-1])
        yv = y.group(1) if y else "unk"
        return (f"asof_{yv}", f"As of {dm[-1]}")
    # default
    return ("col", txt or "")

def _infer_multi_columns(words_df: pd.DataFrame) -> List[Dict]:
    """
    Trả về danh sách [{x, context_key, context_label}], từ trái sang phải.
    """
    centers = _cluster_numeric_columns(words_df, max_k=6)
    cols = []
    used_keys = set()
    for idx, x in enumerate(centers, 1):
        htext = _extract_header_text_near(words_df, x)
        key, label = _normalize_context_from_header(htext)
        # tránh trùng key
        if key == "col":
            key = f"col{idx}"
        if key in used_keys:
            key = f"{key}_{idx}"
        used_keys.add(key)
        cols.append({"x": int(x), "context_key": key, "context_label": label})
    # lọc những trường hợp nhiễu có <2 cột
    if len(cols) >= 2:
        return cols
    return []

def _parse_line_with_multi_columns(line_df: pd.DataFrame, cols: List[Dict], tol=100):
    texts, xs = line_df["text"].tolist(), line_df["x"].tolist()
    # (1) GHÉP SỐ bị tách: thay vì duyệt token rời, dùng merge
    merged_nums = _merge_numeric_runs(texts, xs, gap_px=62)
    nums = []
    for x, clean_text, raw in merged_nums:
        # yêu cầu tối thiểu 5 chữ số để tránh nhiễu
        if clean_text and len(re.sub(r'[^\d]', '', clean_text)) >= 5:
            nums.append((x, clean_text, raw))
    if not nums:
        return "", {}

    # (2) LẤY NHÃN: mềm hơn – dựa trên vị trí số đầu tiên trong dòng
    first_num_x = min(x for x, _n, _r in nums)
    left_most_col = min(c["x"] for c in cols)
    threshold_x = min(first_num_x, left_most_col) - 20

    # ưu tiên token không phải số
    label_tokens = [t for t, x in zip(texts, xs) if x < threshold_x and not NUMISH_RE.match(t)]
    if not label_tokens:
        # fallback: lấy mọi token (kể cả số VAS code đầu dòng) nằm trước threshold
        label_tokens = [t for t, x in zip(texts, xs) if x < threshold_x]
    label = " ".join(label_tokens).strip()

    def find_best_match(target_x, numbers, tolerance=tol):
        if not numbers:
            return None
        distances = sorted([(abs(x - target_x), num) for x, num, _raw in numbers], key=lambda z: z[0])
        if distances and distances[0][0] <= tolerance:
            return distances[0][1]
        return None

    values = {}
    for c in cols:
        raw = find_best_match(c["x"], nums) or find_best_match(c["x"], nums, tol * 1.6)
        if raw is not None:
            values[c["context_key"]] = raw

    return label, values

def _strip_page_headers(words_df: pd.DataFrame) -> pd.DataFrame:
    if words_df.empty:
        return words_df
    y_min, y_max = words_df["y"].min(), words_df["y"].max()
    height = y_max - y_min if y_max > y_min else 1
    # cắt header/footer
    top_cut = y_min + 0.08 * height
    bot_cut = y_max - 0.12 * height
    df = words_df[(words_df["y"] >= top_cut) & (words_df["y"] <= bot_cut)].copy()
    if df.empty:
        return words_df

    # loại dòng “năm 2025”, “Hà Nội, ngày ... 2025”
    grouped = (
    df.sort_values(["y","x"])
      .groupby(pd.cut(df["y"], bins=max(1, int(height/20)), include_lowest=True), observed=False)
    )
    keep_rows = []
    for _bin, g in grouped:
        toks = g["text"].tolist()
        nums = [re.sub(r"\D","",t) for t in toks]
        years = [n for n in nums if len(n)==4 and 1900 <= int(n) <= 2100]
        # nếu hơn 60% token là năm/số 4 chữ số → bỏ (rất hay gặp ở header/footer)
        ratio_year = 0 if not toks else (len(years)/len(toks))
        if ratio_year > 0.6:
            continue
        keep_rows.append(g)
    if keep_rows:
        df = pd.concat(keep_rows, ignore_index=True)
    return df