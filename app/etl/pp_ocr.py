from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import os, re
from decimal import Decimal, ROUND_HALF_UP

try:
    from paddleocr import PPStructureV3
    PPSTRUCTURE_AVAILABLE = True
except ImportError:
    PPSTRUCTURE_AVAILABLE = False

from .table_finder import classify_title, detect_unit
from .normalize import normalize_with_unit

NUM_RE = re.compile(r"\(?[-+]?\s*(?:\d{1,3}(?:[.\s]\d{3})+|\d+)(?:,\d+|\.\d+)?\)?")
VAS_RE = re.compile(r"(^|\s)(\d{3})(\s|$)")

def _decimal2(v: Decimal) -> Decimal:
    return v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def extract_tables_pp(pdf_path: Path, max_pages: Optional[int] = None) -> pd.DataFrame:
    """
    Dùng PP-Structure V3 để tách bảng, đọc từng ô -> ráp theo cột.
    Chỉ trả các dòng khả tín: có 'vas_code' hoặc có nhãn + 1-2 số.
    """
    if not PPSTRUCTURE_AVAILABLE:
        print("⚠️ PPStructureV3 không khả dụng")
        return pd.DataFrame()
    
    # Khởi tạo PPStructureV3 không dùng tham số show_log
    try:
        engine = PPStructureV3()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo PPStructureV3: {e}")
        return pd.DataFrame()

    out_rows: List[Dict] = []

    # PP-Structure sẽ tự split ảnh theo trang
    try:
        result = engine(str(pdf_path))
    except Exception as e:
        print(f"❌ Lỗi PPStructure khi xử lý {pdf_path.name}: {e}")
        return pd.DataFrame()
        
    # result là list dicts gồm nhiều 'img_idx': mỗi idx ~ một trang
    # Ta gom theo img_idx (trang)
    pages = {}
    for block in result:
        idx = block.get("img_idx", 0) + 1
        if max_pages and idx > max_pages: 
            continue
        pages.setdefault(idx, []).append(block)

    for page_no, blocks in sorted(pages.items()):
        # ghép text của trang để bắt đơn vị + tiêu đề bảng (BS/IS/CF)
        page_text = []
        for b in blocks:
            if b["type"] in ("text", "title"):
                page_text.append(b.get("res", [{}])[0].get("text", ""))
        merged = "\n".join(page_text)
        unit_hint = detect_unit(merged) or "VND"
        stmt_hint  = classify_title(merged)  # 'BS'|'IS'|'CF'|None

        # duyệt các bảng
        for b in blocks:
            if b["type"] != "table":
                continue
            table = b.get("res", {})
            cells = table.get("cells", [])
            if not cells:
                continue

            # reconstruct matrix: hàng -> list ô (text + bbox)
            # PP-Structure cho sẵn 'bbox' và 'text' cho từng cell
            # Ta gom theo row_id
            by_row = {}
            for c in cells:
                r = c["row"]
                by_row.setdefault(r, []).append(c)
            # sắp xếp theo cột (x0)
            for r, items in by_row.items():
                items.sort(key=lambda c: c["bbox"][0])

            # tìm dòng header để biết cột mã/nhãn/2 cột số
            hdr_row = None
            for r, items in by_row.items():
                txt = " ".join([i["text"] for i in items if i.get("text")])
                tl = txt.lower()
                if any(k in tl for k in ["mã", "ma so", "mã số", "mã số"]) and \
                   any(k in tl for k in ["chỉ tiêu", "diễn giải", "chi tieu", "dien giai"]):
                    hdr_row = r
                    break
            if hdr_row is None:
                # thử bắt theo từ khóa cột kỳ: cuối kỳ/đầu năm/kỳ này/kỳ trước
                for r, items in by_row.items():
                    tl = " ".join([i["text"] for i in items if i.get("text")]).lower()
                    if any(k in tl for k in ["cuối kỳ","cuoi ky","đầu năm","dau nam","kỳ này","ky nay","kỳ trước","ky truoc"]):
                        hdr_row = r
                        break
            if hdr_row is None:
                # không có header => khó xác định cột -> bỏ
                continue

            # xác định index cột
            hdr_items = by_row[hdr_row]
            col_roles = ["code", "label", "current", "prior"]
            role_of_col = {}
            for j, it in enumerate(hdr_items):
                t = (it.get("text") or "").lower()
                if any(w in t for w in ["mã", "ma so", "mã số"]): role_of_col[j] = "code"
                elif any(w in t for w in ["chỉ tiêu", "diễn giải", "chi tieu", "dien giai"]): role_of_col[j] = "label"
                elif any(w in t for w in ["cuối kỳ","cuoi ky","kỳ này","ky nay"]): role_of_col[j] = "current"
                elif any(w in t for w in ["đầu năm","dau nam","kỳ trước","ky truoc"]): role_of_col[j] = "prior"

            # nếu chưa gán đủ cột số, ta gán 2 cột phải cùng (heurstic)
            # đảm bảo 'current' là cột phải cùng, 'prior' là kế bên trái
            if "current" not in role_of_col.values():
                # pick 2 cột phải cùng trong header như cột số
                last_two = list(range(len(hdr_items)))[-2:]
                if len(last_two) == 2:
                    role_of_col[last_two[1]] = role_of_col.get(last_two[1], "current")
                    role_of_col[last_two[0]] = role_of_col.get(last_two[0], "prior")

            # duyệt các dòng dữ liệu (sau header)
            for r, items in by_row.items():
                if r <= hdr_row: 
                    continue
                items = [i for i in items if (i.get("text") or "").strip()]
                if not items:
                    continue

                # map cell theo vai trò
                text_by_role = {"code":"", "label":"", "current":"", "prior":""}
                for j, it in enumerate(items):
                    role = role_of_col.get(j)
                    if role:
                        text_by_role[role] = (text_by_role[role] + " " + it.get("text","")).strip()

                raw_code   = text_by_role["code"] or ""
                raw_label  = text_by_role["label"] or ""
                raw_curr   = text_by_role["current"] or ""
                raw_prior  = text_by_role["prior"] or ""

                # Bắt mã VAS
                vas = None
                mvas = VAS_RE.search(raw_code) or VAS_RE.search(raw_label)
                if mvas:
                    vas = mvas.group(2)

                # Ít nhất phải có 1 số liệu ở current/prior
                nums = []
                if raw_curr: nums += NUM_RE.findall(raw_curr)
                if raw_prior: nums += NUM_RE.findall(raw_prior)
                if not nums:
                    continue

                # chuẩn hóa đơn vị -> VND
                val_curr, _ = normalize_with_unit(raw_curr, unit_hint) if raw_curr else (None, unit_hint)
                val_prior, _ = normalize_with_unit(raw_prior, unit_hint) if raw_prior else (None, unit_hint)

                # label hợp lệ
                label = raw_label.strip()
                if vas:
                    # nhiều báo cáo để phần label rỗng khi gộp ô -> fallback sang raw_code
                    if len(label) < 3:
                        label = (raw_code or "").strip()

                if not label or len(label) < 3:
                    continue

                # page + statement hint giúp lọc gắn đúng statement
                out_rows.append({
                    "page": page_no,
                    "statement_hint": stmt_hint,
                    "vas_code": vas,
                    "src_label": label,
                    "amount_current": float(_decimal2(val_curr)) if val_curr is not None else None,
                    "amount_prior":  float(_decimal2(val_prior)) if val_prior is not None else None,
                    "unit_hint": unit_hint,
                    "confidence": 0.88 if vas else 0.80
                })

    return pd.DataFrame(out_rows)