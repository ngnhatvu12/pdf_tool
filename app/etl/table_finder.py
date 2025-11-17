# app/etl/table_finder.py
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import fitz  # PyMuPDF

TITLE_PATTERNS = {
    "BS": [
        re.compile(r"b(ả|a)ng\s+c(â|a)n\s+đ(ô|o)́i\s+k(ế|e)\s+to(á|a)n", re.I|re.U),
        re.compile(r"c(â|a)n\s+đ(ô|o)́i\s+k(ế|e)\s+to(á|a)n", re.I|re.U),
        re.compile(r"b(ả|a)ng\s+c(â|a)n\s+đ(ô|o)́i", re.I|re.U),
        re.compile(r"t(à|a)i\s+s(ả|a)n", re.I|re.U),  # Thêm pattern cho tài sản
    ],
    "IS": [
        re.compile(r"b(á|a)o\s+c(á|a)o\s+k(ế|e)t\s+qu(ả|a)\s+ho(ạ|a)t\s+đ(ô|o)̣ng\s+kinh\s+doanh", re.I|re.U),
        re.compile(r"k(ế|e)t\s+qu(ả|a)\s+ho(ạ|a)t\s+đ(ô|o)̣ng\s+kinh\s+doanh", re.I|re.U),
        re.compile(r"b(á|a)o\s+c(á|a)o\s+doanh\s+thu", re.I|re.U),
        re.compile(r"l(ợ|o)i\s+nhu(ậ|a)n", re.I|re.U),  # Thêm pattern cho lợi nhuận
    ],
    "CF": [
        re.compile(r"b(á|a)o\s+c(á|a)o\s+l(ư|u)u\s+chuy(ê|e)n\s+ti(ê|e)n\s+t(ê|e)", re.I|re.U),
        re.compile(r"l(ư|u)u\s+chuy(ê|e)n\s+ti(ê|e)n\s+t(ê|e)", re.I|re.U),
        re.compile(r"b(á|a)o\s+c(á|a)o\s+ngu(ồ|o)n\s+ti(ề|e)n", re.I|re.U),
        re.compile(r"ti(ề|e)n\s+v(à|a)\s+t(ư|u)ơng\s+đ(ươ|u)ơng\s+ti(ề|e)n", re.I|re.U),  # Thêm pattern cho tiền
    ]
}

EXPLAIN_PAT = re.compile(r"thuy(ê|e)t\s+minh|notes?\s+to\s+the\s+financial", re.I|re.U)

def classify_title(text: str) -> Optional[str]:
    """Phân loại tiêu đề với nhiều pattern hơn"""
    text_lower = text.lower()
    
    # Kiểm tra từng loại báo cáo với nhiều pattern
    for key, patterns in TITLE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text_lower):
                return key
    return None

def detect_unit(page_text: str) -> Optional[str]:
    m = re.search(r"đ(ơ|o)n\s+v(ị|i)\s*[:：]\s*([^\n]+)", page_text, flags=re.I|re.U)
    if not m:
        return None
    unit_line = m.group(2).lower()
    if "triệu" in unit_line: return "triệu đồng"
    if "nghìn" in unit_line or "ngàn" in unit_line: return "nghìn đồng"
    if "tỷ" in unit_line or "ty" in unit_line: return "tỷ đồng"
    return "VND"

def _read_page_texts(pdf_path: Path) -> List[str]:
    out = []
    with fitz.open(pdf_path) as doc:
        for p in doc:
            out.append(p.get_text("text") or "")
    return out

def locate_statement_pages(pdf_path: Path, max_pages_to_scan=20) -> Dict[str, Tuple[int,int]]:
    """
    Trả về khoảng (1-based, inclusive) cho {'BS':(s,e), 'IS':(...), 'CF':(...)}.
    - Quét tự động max_pages_to_scan trang đầu để tìm báo cáo
    - Tự động xác định phạm vi trang cho từng loại báo cáo
    """
    pages = _read_page_texts(pdf_path)
    n = len(pages)
    
    # Giới hạn số trang quét
    scan_limit = min(max_pages_to_scan, n)
    
    print(f"🔍 Quét tự động {scan_limit} trang đầu để tìm báo cáo...")
    
    marks: List[Tuple[int,str]] = []
    found_sections = set()
    
    # Quét các trang để tìm tiêu đề báo cáo
    for i, tx in enumerate(pages[:scan_limit], 1):
        t = classify_title(tx)
        if t and t not in found_sections:
            marks.append((i, t))
            found_sections.add(t)
            print(f"   ✅ Tìm thấy {t} tại trang {i}")

    ranges: Dict[str, Tuple[int,int]] = {}
    
    if marks:
        # Sắp xếp theo thứ tự trang
        marks.sort(key=lambda x: x[0])
        
        # Xác định phạm vi cho từng báo cáo
        for idx, (pg, kind) in enumerate(marks):
            # Tìm trang kế tiếp (báo cáo tiếp theo hoặc thuyết minh)
            next_pg = n + 1  # mặc định là cuối file
            
            # Tìm báo cáo tiếp theo
            if idx + 1 < len(marks):
                next_pg = marks[idx+1][0]
            else:
                # Tìm phần thuyết minh
                for j in range(pg, min(pg + 15, n)):  # tìm trong 15 trang tiếp
                    if EXPLAIN_PAT.search(pages[j-1]):
                        next_pg = j
                        break
            
            end = next_pg - 1
            
            # Đảm bảo không vượt quá scan_limit
            end = min(end, pg + 12)  # tối đa 12 trang cho mỗi báo cáo
            end = min(end, scan_limit)
            
            ranges[kind] = (pg, end)
            print(f"   📄 {kind}: trang {pg} - {end}")
    
    # Fallback: nếu không tìm thấy tự động, quét 20 trang đầu
    if not ranges:
        print("   ⚠️ Không tìm thấy báo cáo tự động, quét toàn bộ 20 trang đầu")
        ranges = {
            "BS": (1, min(20, scan_limit)),
            "IS": (1, min(20, scan_limit)), 
            "CF": (1, min(20, scan_limit))
        }
    
    return ranges

def locate_notes_pages(pdf_path: Path, default_start: int = 15) -> Tuple[int, int]:
    pages = _read_page_texts(pdf_path)
    n = len(pages)
    start = None
    for i, tx in enumerate(pages, 1):
        if EXPLAIN_PAT.search(tx):
            start = i
            break
    if not start:
        start = min(default_start, n)
    return start, n