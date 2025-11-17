# app/etl/extractors.py
import re
from pathlib import Path
from typing import Optional, Dict
from datetime import date, datetime

def guess_ticker_from_filename(fp: Path) -> Optional[str]:
    m = re.search(r"\b([A-Z]{2,5})_", fp.name)
    return m.group(1) if m else None

def detect_consolidation(text: str) -> str:
    if re.search(r"\bhợp nhất\b", text, flags=re.I|re.U): return "consolidated"
    if re.search(r"\brieng\b|công ty mẹ\b", text, flags=re.I|re.U): return "separate"
    return "consolidated"

def detect_gaap(text: str) -> str:
    return "IFRS" if re.search(r"\bIFRS\b", text) else "VAS"

def detect_period(text: str, filename: str = "") -> Dict[str, Optional[date]]:
    f = (filename or "").lower()
    t = (text or "").lower()
    
    # Ưu tiên tìm trong text trước
    # Pattern cho quý 2/2025
    m = re.search(r'q\s*2\s*[/\-]?\s*2025', f) or re.search(r'qu[ýy]?\s*2\s*[/\-]?\s*2025', t)
    if m:
        return {"start": date(2025, 4, 1), "end": date(2025, 6, 30)}
    
    # Pattern cho 6 tháng đầu năm 2025
    m = re.search(r'6\s*[tth]\s*2025', f) or re.search(r'6\s*th[aá]ng\s*2025', t)
    if m:
        return {"start": date(2025, 1, 1), "end": date(2025, 6, 30)}
    
    # Pattern cho năm 2025
    m = re.search(r'2025', f) or re.search(r'2025', t)
    if m:
        # Nếu có Q2 thì là quý 2, ngược lại mặc định là 6 tháng
        if 'q2' in f or 'quý 2' in t:
            return {"start": date(2025, 4, 1), "end": date(2025, 6, 30)}
        else:
            return {"start": date(2025, 1, 1), "end": date(2025, 6, 30)}
    
    # Fallback an toàn
    return {"start": date(2025, 1, 1), "end": date(2025, 6, 30)}

def detect_report_type(text: str, filename: str = "") -> str:
    f = (filename or "").lower()
    t = (text or "").lower()
    
    if re.search(r'q\s*2|qu[ýy]\s*2', f + " " + t):
        return "quarter"
    if re.search(r'6\s*[tth]|6\s*th[aá]ng', f + " " + t):
        return "semi_annual"
    if re.search(r'n[aă]m|annual|nien', f + " " + t):
        return "annual"
    
    # Dựa vào filename
    if 'q2' in f:
        return "quarter"
    if '6t' in f:
        return "semi_annual"
    
    return "quarter"  # Mặc định

def detect_auditor(text: str) -> Optional[str]:
    for k in ["Ernst & Young","EY","BDO","KPMG","PwC","PricewaterhouseCoopers"]:
        if k.lower() in (text or "").lower():
            return "Ernst & Young Vietnam" if "ey" in k.lower() else ("PwC Vietnam" if "pwc" in k.lower() else k)
    return None
