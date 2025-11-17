# app/etl/normalize.py
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional, Tuple
from rapidfuzz import fuzz, process

VN_NUM_RE = re.compile(r"\(?[-+]?\s*(?:\d{1,3}(?:[.\s]\d{3})+|\d+)(?:,\d+|\.\d+)?\)?")
MAX_ABS_VND = Decimal("1e15")

def _to_decimal(txt: str) -> Optional[Decimal]:
    s = txt.strip()
    neg = s.startswith("(") and s.endswith(")")
    s1 = s.replace(" ", "").replace(".", "").replace(",", ".")
    try:
        v = Decimal(s1)
        return -v if neg else v
    except InvalidOperation:
        try:
            v = Decimal(s.replace(",", ""))
            return -v if neg else v
        except InvalidOperation:
            return None

def parse_vn_number(s: str) -> Optional[Decimal]:
    matches = list(VN_NUM_RE.finditer(s))
    if not matches:
        return None
    # Lấy match có nhiều chữ số nhất (số dài nhất)
    best = None
    best_len = -1
    for m in matches:
        raw = m.group(0)
        digits = re.sub(r"[^\d]", "", raw)
        if len(digits) > best_len:
            best_len = len(digits)
            best = raw
    if best is None or best_len >= 18:
        return None
    return _to_decimal(best)


def normalize_with_unit(raw: str, unit_hint: Optional[str]) -> Tuple[Optional[Decimal], str]:
    amt = parse_vn_number(raw)
    if amt is None: return None, unit_hint or "VND"

    unit = (unit_hint or "VND").lower()
    mul = Decimal(1)
    if "ngh" in unit or "ngan" in unit or "ngàn" in unit: mul = Decimal(1_000)
    elif "tri" in unit: mul = Decimal(1_000_000)
    elif "tỷ" in unit or "ty" in unit: mul = Decimal(1_000_000_000)

    vnd = (amt * mul)
    if vnd.copy_abs() > MAX_ABS_VND: return None, unit_hint or "VND"
    vnd = vnd.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return vnd, unit_hint or "VND"

def choose_line_item(label: str, vas_code: Optional[str], candidates, score_cutoff=50) -> Optional[str]:
    """Cải tiến matching với mapping cứng mở rộng và logic thông minh hơn"""
    if not label or len(label.strip()) < 2:
        return None
        
    label_clean = label.lower().strip()
    
    # Ưu tiên cao: map theo mã VAS
    if vas_code:
        vas_clean = vas_code.strip()
        for code, _nats, vas in candidates:
            if vas and vas.strip() == vas_clean:
                return code
        # Thử partial VAS matching
        for code, _nats, vas in candidates:
            if vas and vas_clean.startswith(vas.strip()):
                return code
    
    # MAPPING CỨNG MỞ RỘNG - ĐÃ CẬP NHẬT THEO SEED.PY
    hard_mapping = {
        # === BẢNG CÂN ĐỐI KẾ TOÁN ===
        # TÀI SẢN NGẮN HẠN
        "ngan taisan han": "BS_CURRENT_ASSETS",
        "tiền và các khoản tương đương tiền": "BS_CASH",
        "tiền": "BS_CASH", 
        "các khoản tương đương tiền": "BS_CASH_EQUIV",
        "tương đương tiền": "BS_CASH",
        "đầu tư tài chính ngắn hạn": "BS_ST_INVEST",
        "chứng khoán kinh doanh": "BS_TRADING_SECURITIES",
        "các khoản phải thu ngắn hạn": "BS_ST_RECEIVABLES",
        "phải thu ngắn hạn": "BS_ST_RECEIVABLES",
        "phải thu ngắn hạn của khách hàng": "BS_TRADE_AR_ST",
        "phải thu khách hàng": "BS_TRADE_AR_ST",
        "trả trước cho người bán": "BS_PREPAY_SUPPLIER_ST",
        "hàng tồn kho": "BS_INVENTORIES",
        "tài sản ngắn hạn khác": "BS_OTHER_CURRENT_ASSETS",
        "chi phí trả trước ngắn hạn": "BS_PREPAID_EXP_ST",
        "thuế gtgt được khấu trừ": "BS_VAT_DEDUCTIBLE",
        
        # TÀI SẢN DÀI HẠN
        "taisan dai han": "BS_NONCURRENT_ASSETS",
        "các khoản phải thu dài hạn": "BS_LONG_TERM_RECEIVABLES",
        "phải thu dài hạn": "BS_LONG_TERM_RECEIVABLES",
        "phải thu về cho vay dài hạn": "BS_LT_LOANS_RECEIVABLE",
        "tài sản cố định": "BS_PPE",
        "tài sản cố định hữu hình": "BS_PPE_TANGIBLE",
        "tài sản cố định thuê tài chính": "BS_PPE_FINANCE_LEASED",
        "tài sản cố định vô hình": "BS_INTANGIBLE_ASSETS",
        "bất động sản đầu tư": "BS_INVESTMENT_PROPERTY",
        "bat dau déng sin": "BS_INVESTMENT_PROPERTY",
        "tài sản dở dang dài hạn": "BS_CONSTRUCTION_IN_PROGRESS_LT",
        "tai san do dang dai han": "BS_CONSTRUCTION_IN_PROGRESS_LT",
        "chi phí xây dựng cơ bản dở dang": "BS_CONSTRUCTION_IN_PROGRESS_LT",
        "đầu tư tài chính dài hạn": "BS_LONG_TERM_INVESTMENTS",
        "dau tai chinh dai han": "BS_LONG_TERM_INVESTMENTS",
        "đầu tư vào công ty liên doanh": "BS_INVESTMENTS_IN_ASSOC",
        "đầu tư góp vốn vào đơn vị khác": "BS_LONG_TERM_INVESTMENTS",
        "lợi thế thương mại": "BS_GOODWILL",
        "loi the thuong mai": "BS_GOODWILL",
        "tài sản dài hạn khác": "BS_OTHER_NONCURRENT_ASSETS",
        "tai san dai han khac": "BS_OTHER_NONCURRENT_ASSETS",
        "chi phí trả trước dài hạn": "BS_OTHER_NONCURRENT_ASSETS",
        "tài sản thuế thu nhập hoãn lại": "BS_OTHER_NONCURRENT_ASSETS",
        
        # TỔNG CỘNG TÀI SẢN
        "tổng cộng tài sản": "BS_TOTAL_ASSETS",
        "tong cong tai san": "BS_TOTAL_ASSETS",
        "san tong tai cong": "BS_TOTAL_ASSETS",
        "tổng tài sản": "BS_TOTAL_ASSETS",
        
        # NỢ PHẢI TRẢ
        "nợ phải trả": "BS_TOTAL_LIABILITIES",
        "no phai tra": "BS_TOTAL_LIABILITIES",
        "nợ ngắn hạn": "BS_CURRENT_LIAB",
        "no ngan han": "BS_CURRENT_LIAB",
        "phải trả người bán ngắn hạn": "BS_ST_TRADE_PAYABLES",
        "người mua trả tiền trước": "BS_ST_TRADE_PAYABLES",
        "thuế và các khoản phải nộp nhà nước": "BS_TAX_PAYABLE_ST",
        "phải trả người lao động": "BS_PAYROLL_PAYABLE",
        "chi phí phải trả ngắn hạn": "BS_ST_EXPENSES_PAYABLE",
        "vay và nợ thuê tài chính ngắn hạn": "BS_ST_BORROWINGS",
        "nợ dài hạn": "BS_NONCURRENT_LIAB",
        "no dai han": "BS_NONCURRENT_LIAB",
        "vay và nợ thuê tài chính dài hạn": "BS_LT_BORROWINGS",
        "thuế thu nhập hoãn lại phải trả": "BS_NONCURRENT_LIAB",
        
        # VỐN CHỦ SỞ HỮU
        "vốn chủ sở hữu": "BS_EQUITY",
        "von chu so huu": "BS_EQUITY",
        "vốn góp của chủ sở hữu": "BS_PAID_IN_CAPITAL",
        "thặng dư vốn cổ phần": "BS_SHARE_PREMIUM",
        "lợi nhuận sau thuế chưa phân phối": "BS_RETAINED_EARNINGS",
        "loi nhuan sau thue chua phan phoi": "BS_RETAINED_EARNINGS",
        "lợi ích cổ đông không kiểm soát": "BS_NONCONTROLLING_INTEREST",
        "loi ich co dong khong kiem soat": "BS_NONCONTROLLING_INTEREST",
        
        # TỔNG CỘNG NGUỒN VỐN
        "tổng cộng nguồn vốn": "BS_TOTAL_LIAB_EQUITY",
        "tong cong nguon von": "BS_TOTAL_LIAB_EQUITY",
        "von nguon tong cong": "BS_TOTAL_LIAB_EQUITY",
        
        # === KẾT QUẢ KINH DOANH ===
        "doanh thu thuần bán hàng và cung cấp dịch vụ": "IS_REVENUE",
        "doanh thu bán hàng và cung cấp dịch vụ": "IS_REVENUE",
        "giá vốn hàng bán": "IS_COGS",
        "gia von hang ban": "IS_COGS",
        "lợi nhuận gộp về bán hàng và cung cấp dịch vụ": "IS_GROSS_PROFIT",
        "doanh thu hoạt động tài chính": "IS_FINANCE_INCOME",
        "chi phí tài chính": "IS_FINANCE_EXPENSE",
        "chi phi tai chinh": "IS_FINANCE_EXPENSE",
        "chi phí bán hàng": "IS_SELLING_EXP",
        "chi phí quản lý doanh nghiệp": "IS_ADMIN_EXP",
        "lợi nhuận thuần từ hoạt động kinh doanh": "IS_ACCOUNTING_PROFIT_B4_TAX",
        "loi nhuan thuan tu hoat dong kinh doanh": "IS_ACCOUNTING_PROFIT_B4_TAX",
        "thu nhập khác": "IS_OTHER_INCOME",
        "chi phí khác": "IS_OTHER_EXPENSES",
        "lợi nhuận khác": "IS_OTHER_INCOME",
        "tổng lợi nhuận kế toán trước thuế": "IS_ACCOUNTING_PROFIT_B4_TAX",
        "tong loi nhuan ke toan truoc thue": "IS_ACCOUNTING_PROFIT_B4_TAX",
        "lợi nhuận trước thuế": "IS_ACCOUNTING_PROFIT_B4_TAX",
        "chi phí thuế thu nhập doanh nghiệp": "IS_CIT_EXPENSE",
        "lợi nhuận sau thuế thu nhập doanh nghiệp": "IS_PROFIT_AFTER_TAX",
        "loi nhuan sau thue thu nhap doanh nghiep": "IS_PROFIT_AFTER_TAX",
        "lợi nhuận sau thuế": "IS_PROFIT_AFTER_TAX",
        
        # === LƯU CHUYỂN TIỀN TỆ ===
        "lưu chuyển tiền thuần từ hoạt động kinh doanh": "CF_NET_CFO",
        "luu chuyen tien thuan tu hoat dong kinh doanh": "CF_NET_CFO",
        "lưu chuyển tiền thuần từ hoạt động đầu tư": "CF_NET_CFI",
        "lưu chuyển tiền thuần từ hoạt động tài chính": "CF_NET_CFF",
        "khấu hao tài sản cố định": "CF_DEPRECIATION",
        "khao tai san co dinh": "CF_DEPRECIATION",
        "lợi nhuận trước thuế": "CF_CASH_FROM_OPERATIONS_B4_WC",
        "tiền chi để mua sắm tài sản cố định": "CF_PURCHASE_PPE",
        "tiền thu từ thanh lý tài sản cố định": "CF_PROCEEDS_DISPOSAL_PPE",
        "tiền thu từ phát hành cổ phiếu": "CF_NET_CFF",
        "tiền thu từ đi vay": "CF_BORROWINGS_OBTAINED",
        "tiền trả nợ gốc vay": "CF_BORROWINGS_REPAID",
        "cổ tức, lợi nhuận đã trả": "CF_DIVIDENDS_PAID",

        # === MÃ CTCK ĐẶC BIỆT ===
        "tài sản tài chính ghi nhận thông qua lãi/lỗ": "BS_FVTPL",
        "chứng khoán kinh doanh (fvptl)": "BS_FVTPL",
        "chứng khoán sẵn sàng để bán": "BS_AFS",
        "chứng khoán nắm giữ đến ngày đáo hạn": "BS_HTM",
        "cho vay ký quỹ": "BS_MARGIN_LOANS",
        "phải thu với vsd": "BS_RECEIVABLES_VSD",
    }
    
    # Tìm trong mapping cứng với partial matching
    for keyword, code in hard_mapping.items():
        if keyword in label_clean:
            return code
    
    # Thử matching với từ khóa ngắn hơn
    short_mapping = {
        "tiền": "BS_CASH",
        "phải thu": "BS_ST_RECEIVABLES", 
        "hàng tồn kho": "BS_INVENTORIES",
        "tài sản cố định": "BS_PPE",
        "bất động sản": "BS_INVESTMENT_PROPERTY",
        "đầu tư tài chính": "BS_LONG_TERM_INVESTMENTS",
        "lợi thế": "BS_GOODWILL",
        "vốn chủ": "BS_EQUITY",
        "vốn góp": "BS_PAID_IN_CAPITAL",
        "lợi nhuận chưa phân phối": "BS_RETAINED_EARNINGS",
        "doanh thu": "IS_REVENUE",
        "giá vốn": "IS_COGS",
        "lợi nhuận gộp": "IS_GROSS_PROFIT",
        "lợi nhuận trước thuế": "IS_ACCOUNTING_PROFIT_B4_TAX",
        "lợi nhuận sau thuế": "IS_PROFIT_AFTER_TAX",
        "lưu chuyển tiền": "CF_NET_CFO",
        "ngan han": "BS_CURRENT_LIAB",
        "dai han": "BS_NONCURRENT_LIAB",
        "tong cong": "BS_TOTAL_ASSETS",
        "loi nhuan": "IS_PROFIT_AFTER_TAX",
        "chi phi": "IS_ADMIN_EXP",
        "thue": "IS_CIT_EXPENSE",
    }
    
    for keyword, code in short_mapping.items():
        if keyword in label_clean:
            return code
    
    # Thử matching theo số VAS trong label (nếu có số nhưng không có mã VAS rõ)
    vas_match = re.search(r'\b(\d{2,3})\b', label)
    if vas_match:
        vas_num = vas_match.group(1)
        for code, _nats, vas in candidates:
            if vas and vas == vas_num:
                return code
    
    # Fuzzy matching với native examples (giảm cutoff xuống 50)
    flat, exs = [], []
    for code, natives, _vas in candidates:
        for ex in (natives or []):
            flat.append((code, ex))
            exs.append(ex.lower())
    
    if exs:
        best = process.extractOne(label_clean, exs, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
        if best and best[1] >= score_cutoff:
            return flat[exs.index(best[0])][0]
    
    # Fallback: thử tìm theo từ khóa chính trong label
    main_keywords = ["tiền", "thu", "kho", "cố định", "đầu tư", "vốn", "nợ", "lợi nhuận", "doanh thu", "chi phí"]
    for keyword in main_keywords:
        if keyword in label_clean:
            # Trả về code phù hợp nhất với keyword
            if keyword in ["tiền", "thu"]: return "BS_CASH"
            elif keyword in ["kho"]: return "BS_INVENTORIES" 
            elif keyword in ["cố định"]: return "BS_PPE"
            elif keyword in ["đầu tư"]: return "BS_LONG_TERM_INVESTMENTS"
            elif keyword in ["vốn"]: return "BS_EQUITY"
            elif keyword in ["nợ"]: return "BS_TOTAL_LIABILITIES"
            elif keyword in ["lợi nhuận"]: return "IS_PROFIT_AFTER_TAX"
            elif keyword in ["doanh thu"]: return "IS_REVENUE"
            elif keyword in ["chi phí"]: return "IS_ADMIN_EXP"
    
    return None