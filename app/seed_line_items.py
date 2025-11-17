from sqlalchemy import text
from app.db import engine
from app.models import Statement

INITIAL_ITEMS = [
    # ===========================
    # BẢNG CÂN ĐỐI KẾ TOÁN (BS)
    # ===========================

    # --- Nhóm tài sản ngắn hạn (100) ---
    dict(statement=Statement.BS.value, std_code="BS_CURRENT_ASSETS", std_label="Current assets",
         native_examples=["Tài sản ngắn hạn"], vas_code="100", sign=+1),

    dict(statement=Statement.BS.value, std_code="BS_CASH", std_label="Cash and cash equivalents",
         native_examples=["Tiền và tương đương tiền", "Tiền", "Tương đương tiền"], vas_code="110", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_CASH_CASH", std_label="Cash",
         native_examples=["Tiền"], vas_code="111", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_CASH_EQUIV", std_label="Cash equivalents",
         native_examples=["Tương đương tiền"], vas_code="112", sign=+1),

    dict(statement=Statement.BS.value, std_code="BS_ST_INVEST", std_label="Short-term financial investments",
         native_examples=["Đầu tư tài chính ngắn hạn"], vas_code="120", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_TRADING_SECURITIES", std_label="Trading securities",
         native_examples=["Chứng khoán kinh doanh"], vas_code="121", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ALLOW_TRADING_SECURITIES", std_label="Allowance for trading securities",
         native_examples=["Dự phòng giảm giá chứng khoán kinh doanh"], vas_code="122", sign=-1),
    dict(statement=Statement.BS.value, std_code="BS_HELD_TO_MATURITY_ST", std_label="Held-to-maturity investments (short-term)",
         native_examples=["Đầu tư nắm giữ đến ngày đáo hạn (ngắn hạn)", "Đầu tư nắm giữ đến ngày đáo hạn"], vas_code="123", sign=+1),

    dict(statement=Statement.BS.value, std_code="BS_ST_RECEIVABLES", std_label="Short-term receivables",
         native_examples=["Các khoản phải thu ngắn hạn"], vas_code="130", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_TRADE_AR_ST", std_label="Short-term trade receivables",
         native_examples=["Phải thu ngắn hạn của khách hàng", "Phải thu khách hàng"], vas_code="131", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_PREPAY_SUPPLIER_ST", std_label="Short-term advances to suppliers",
         native_examples=["Trả trước cho người bán ngắn hạn", "Trả trước cho người bán"], vas_code="132", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_VAT_DEDUCTIBLE", std_label="VAT deductible",
         native_examples=["Thuế và các khoản phải thu Nhà nước", "Thuế GTGT được khấu trừ"], vas_code="133", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_INTERCO_AR_ST", std_label="Short-term intercompany receivables",
         native_examples=["Phải thu nội bộ ngắn hạn"], vas_code="134", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_CONSTRUCTION_PROGRESS_AR", std_label="Receivables on construction progress",
         native_examples=["Phải thu theo tiến độ kế hoạch hợp đồng xây dựng"], vas_code="135", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_OTHER_AR_ST", std_label="Other short-term receivables",
         native_examples=["Phải thu ngắn hạn khác"], vas_code="136", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ALLOW_AR_ST", std_label="Allowance for short-term receivables",
         native_examples=["Dự phòng phải thu ngắn hạn khó đòi"], vas_code="137", sign=-1),

    dict(statement=Statement.BS.value, std_code="BS_INVENTORIES", std_label="Inventories",
         native_examples=["Hàng tồn kho"], vas_code="140", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ALLOW_INVENTORIES", std_label="Allowance for devaluation of inventories",
        native_examples=["Dự phòng giảm giá hàng tồn kho"], vas_code="149", sign=-1),

    dict(statement=Statement.BS.value, std_code="BS_OTHER_CURRENT_ASSETS", std_label="Other current assets",
         native_examples=["Tài sản ngắn hạn khác", "Chi phí trả trước ngắn hạn"], vas_code="150", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_PREPAID_EXP_ST", std_label="Short-term prepaid expenses",
         native_examples=["Chi phí trả trước ngắn hạn"], vas_code="151", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_TAX_RECLAIMABLE_ST", std_label="Short-term tax assets",
         native_examples=["Thuế và các khoản phải thu Nhà nước (ngắn hạn)"], vas_code="152", sign=+1),

    # --- Nhóm tài sản dài hạn (200) ---
    dict(statement=Statement.BS.value, std_code="BS_NONCURRENT_ASSETS", std_label="Non-current assets",
         native_examples=["Tài sản dài hạn"], vas_code="200", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_LONG_TERM_RECEIVABLES", std_label="Long-term receivables",
         native_examples=["Các khoản phải thu dài hạn", "Phải thu dài hạn"], vas_code="210", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_LT_TRADE_AR", std_label="Trade receivables - long term",
         native_examples=["Phải thu dài hạn của khách hàng"], vas_code="211", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_LT_ADVANCES", std_label="Long-term advances to suppliers",
         native_examples=["Trả trước cho người bán dài hạn"], vas_code="212", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_LT_INTERCO_AR", std_label="Long-term intercompany receivables",
         native_examples=["Phải thu nội bộ dài hạn"], vas_code="213", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_LT_LOANS_RECEIVABLE", std_label="Loans receivable - long term",
         native_examples=["Phải thu về cho vay dài hạn"], vas_code="215", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_OTHER_AR_LT", std_label="Other long-term receivables",
         native_examples=["Phải thu dài hạn khác"], vas_code="216", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ALLOW_AR_LT", std_label="Allowance for long-term receivables",
         native_examples=["Dự phòng phải thu dài hạn khó đòi"], vas_code="219", sign=-1),

    dict(statement=Statement.BS.value, std_code="BS_PPE", std_label="Property, plant and equipment",
         native_examples=["Tài sản cố định", "Tài sản cố định hữu hình"], vas_code="220", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_PPE_TANGIBLE", std_label="Tangible fixed assets",
         native_examples=["Tài sản cố định hữu hình"], vas_code="221", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_PPE_FINANCE_LEASED", std_label="Finance leased assets",
         native_examples=["Tài sản cố định thuê tài chính"], vas_code="222", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ACCUM_DEPR_PPE", std_label="Accumulated depreciation of PPE",
         native_examples=["Giá trị hao mòn lũy kế"], vas_code="223", sign=-1),

    dict(statement=Statement.BS.value, std_code="BS_INVESTMENT_PROPERTY", std_label="Investment property",
         native_examples=["Bất động sản đầu tư"], vas_code="230", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ORIGINAL_COST_IP", std_label="Investment property - historical cost",
         native_examples=["Nguyên giá BĐS đầu tư"], vas_code="231", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ACCUM_DEPR_IP", std_label="Investment property - accumulated depreciation",
         native_examples=["Giá trị hao mòn lũy kế BĐS đầu tư"], vas_code="232", sign=-1),

    dict(statement=Statement.BS.value, std_code="BS_CONSTRUCTION_IN_PROGRESS_LT", std_label="Construction in progress (long term)",
         native_examples=["Tài sản dở dang dài hạn", "Chi phí xây dựng cơ bản dở dang"], vas_code="240", sign=+1),

    dict(statement=Statement.BS.value, std_code="BS_LONG_TERM_INVESTMENTS", std_label="Long-term financial investments",
         native_examples=["Đầu tư tài chính dài hạn"], vas_code="250", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_INVESTMENTS_IN_SUBS", std_label="Investments in subsidiaries",
         native_examples=["Đầu tư vào công ty con"], vas_code="251", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_INVESTMENTS_IN_ASSOC", std_label="Investments in associates and joint ventures",
         native_examples=["Đầu tư vào công ty liên doanh, liên kết"], vas_code="252", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_HELD_TO_MATURITY_LT", std_label="Held-to-maturity investments (long-term)",
         native_examples=["Đầu tư nắm giữ đến ngày đáo hạn (dài hạn)"], vas_code="253", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_OTHER_LT_INVESTMENTS", std_label="Other long-term investments",
         native_examples=["Đầu tư dài hạn khác"], vas_code="258", sign=+1),

    dict(statement=Statement.BS.value, std_code="BS_INTANGIBLE_ASSETS", std_label="Intangible assets",
         native_examples=["Tài sản cố định vô hình"], vas_code="260", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_GOODWILL", std_label="Goodwill",
         native_examples=["Lợi thế thương mại"], vas_code="261", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_OTHER_NONCURRENT_ASSETS", std_label="Other non-current assets",
         native_examples=["Tài sản dài hạn khác"], vas_code="268", sign=+1),

    dict(statement=Statement.BS.value, std_code="BS_TOTAL_ASSETS", std_label="Total assets",
         native_examples=["Tổng cộng tài sản", "TỔNG CỘNG TÀI SẢN", "Tổng tài sản"], vas_code="270", sign=+1),

    # --- Nợ phải trả (300) ---
    dict(statement=Statement.BS.value, std_code="BS_LIABILITIES", std_label="Liabilities",
         native_examples=["Nợ phải trả"], vas_code="300", sign=+1),

    # Nợ ngắn hạn (310)
    dict(statement=Statement.BS.value, std_code="BS_CURRENT_LIAB", std_label="Current liabilities",
         native_examples=["Nợ ngắn hạn"], vas_code="310", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ST_TRADE_PAYABLES", std_label="Short-term trade payables",
         native_examples=["Phải trả người bán ngắn hạn"], vas_code="311", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_TAX_PAYABLE_ST", std_label="Taxes and statutory obligations",
         native_examples=["Thuế và các khoản phải nộp Nhà nước"], vas_code="313", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_PAYROLL_PAYABLE", std_label="Payables to employees",
         native_examples=["Phải trả người lao động"], vas_code="314", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ST_EXPENSES_PAYABLE", std_label="Short-term accrued expenses",
         native_examples=["Chi phí phải trả ngắn hạn"], vas_code="315", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_UNEARNED_REVENUE_ST", std_label="Short-term unearned revenue",
         native_examples=["Doanh thu chưa thực hiện ngắn hạn"], vas_code="318", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_ST_BORROWINGS", std_label="Short-term borrowings",
         native_examples=["Vay và nợ thuê tài chính ngắn hạn"], vas_code="319", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_DIVIDENDS_PAYABLE", std_label="Dividends payable",
         native_examples=["Cổ tức, lợi nhuận phải trả"], vas_code="317", sign=+1),

    # Nợ dài hạn (330)
    dict(statement=Statement.BS.value, std_code="BS_NONCURRENT_LIAB", std_label="Non-current liabilities",
         native_examples=["Nợ dài hạn"], vas_code="330", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_LT_TRADE_PAYABLES", std_label="Long-term trade payables",
         native_examples=["Phải trả người bán dài hạn"], vas_code="331", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_LT_ACCRUED_EXP", std_label="Long-term accrued expenses",
         native_examples=["Chi phí phải trả dài hạn"], vas_code="335", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_BONDS_PAYABLE", std_label="Bonds payable",
         native_examples=["Trái phiếu phải trả"], vas_code="336", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_LT_BORROWINGS", std_label="Long-term borrowings",
         native_examples=["Vay và nợ thuê tài chính dài hạn", "Vay dài hạn"], vas_code="338", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_PROVISIONS_LT", std_label="Long-term provisions",
         native_examples=["Dự phòng dài hạn"], vas_code="342", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_REVENUE_RECEIVED_AHEAD_LT", std_label="Long-term unearned revenue",
         native_examples=["Doanh thu chưa thực hiện dài hạn"], vas_code="337", sign=+1),

    dict(statement=Statement.BS.value, std_code="BS_TOTAL_LIABILITIES", std_label="Total liabilities",
         native_examples=["Tổng cộng nợ phải trả", "Nợ phải trả"], vas_code="300", sign=+1),

    # --- Vốn chủ sở hữu (400) ---
    dict(statement=Statement.BS.value, std_code="BS_EQUITY", std_label="Owner's equity",
         native_examples=["Vốn chủ sở hữu"], vas_code="400", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_PAID_IN_CAPITAL", std_label="Share capital",
         native_examples=["Vốn góp của chủ sở hữu", "Vốn cổ phần"], vas_code="411", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_SHARE_PREMIUM", std_label="Share premium",
         native_examples=["Thặng dư vốn cổ phần"], vas_code="412", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_OTHER_OWNER_FUNDS", std_label="Other owners' funds",
         native_examples=["Vốn khác của chủ sở hữu"], vas_code="418", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_TREASURY_SHARES", std_label="Treasury shares",
         native_examples=["Cổ phiếu quỹ"], vas_code="419", sign=-1),
    dict(statement=Statement.BS.value, std_code="BS_RETAINED_EARNINGS", std_label="Retained earnings",
         native_examples=["Lợi nhuận sau thuế chưa phân phối"], vas_code="421", sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_NONCONTROLLING_INTEREST", std_label="Non-controlling interests",
         native_examples=["Lợi ích cổ đông không kiểm soát"], vas_code="429", sign=+1),

    dict(statement=Statement.BS.value, std_code="BS_TOTAL_LIAB_EQUITY", std_label="Total liabilities and equity",
         native_examples=["Tổng cộng nguồn vốn", "TỔNG CỘNG NGUỒN VỐN"], vas_code="440", sign=+1),

    # --- Mục CTCK (securities company) bổ sung ---
    dict(statement=Statement.BS.value, std_code="BS_FVTPL", std_label="Financial assets at FVTPL",
         native_examples=["Tài sản tài chính ghi nhận thông qua lãi/lỗ (FVTPL)", "Chứng khoán kinh doanh (FVTPL)"], vas_code=None, sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_AFS", std_label="Available-for-sale securities",
         native_examples=["Chứng khoán sẵn sàng để bán (AFS)"], vas_code=None, sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_HTM", std_label="Held-to-maturity securities",
         native_examples=["Chứng khoán nắm giữ đến ngày đáo hạn (HTM)"], vas_code=None, sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_MARGIN_LOANS", std_label="Margin loans to customers",
         native_examples=["Cho vay ký quỹ", "Dư nợ cho vay giao dịch ký quỹ"], vas_code=None, sign=+1),
    dict(statement=Statement.BS.value, std_code="BS_RECEIVABLES_VSD", std_label="Receivables with VSD/clearing",
         native_examples=["Phải thu với VSD", "Phải thu dịch vụ lưu ký"], vas_code=None, sign=+1),

    # =============================
    # BÁO CÁO KẾT QUẢ KINH DOANH (IS)
    # =============================
    dict(statement=Statement.IS.value, std_code="IS_REVENUE", std_label="Revenue",
         native_examples=["Doanh thu thuần", "Doanh thu bán hàng và cung cấp dịch vụ"], vas_code="10", sign=+1),
    dict(statement=Statement.IS.value, std_code="IS_COGS", std_label="Cost of goods sold",
         native_examples=["Giá vốn hàng bán", "Giá vốn"], vas_code="11", sign=-1),
    dict(statement=Statement.IS.value, std_code="IS_GROSS_PROFIT", std_label="Gross profit",
         native_examples=["Lợi nhuận gộp về bán hàng và cung cấp dịch vụ"], vas_code="20", sign=+1),

    dict(statement=Statement.IS.value, std_code="IS_FINANCE_INCOME", std_label="Finance income",
         native_examples=["Doanh thu hoạt động tài chính"], vas_code="21", sign=+1),
    dict(statement=Statement.IS.value, std_code="IS_FINANCE_EXPENSE", std_label="Finance costs",
         native_examples=["Chi phí tài chính"], vas_code="22", sign=-1),
    dict(statement=Statement.IS.value, std_code="IS_SELLING_EXP", std_label="Selling expenses",
         native_examples=["Chi phí bán hàng"], vas_code="24", sign=-1),
    dict(statement=Statement.IS.value, std_code="IS_ADMIN_EXP", std_label="General and administrative expenses",
         native_examples=["Chi phí quản lý doanh nghiệp"], vas_code="25", sign=-1),

    dict(statement=Statement.IS.value, std_code="IS_OTHER_INCOME", std_label="Other income",
         native_examples=["Thu nhập khác"], vas_code="30", sign=+1),
    dict(statement=Statement.IS.value, std_code="IS_OTHER_EXPENSES", std_label="Other expenses",
         native_examples=["Chi phí khác"], vas_code="31", sign=-1),

    dict(statement=Statement.IS.value, std_code="IS_ACCOUNTING_PROFIT_B4_TAX", std_label="Profit before tax",
         native_examples=["Tổng lợi nhuận kế toán trước thuế", "Lợi nhuận trước thuế"], vas_code="50", sign=+1),
    dict(statement=Statement.IS.value, std_code="IS_CIT_EXPENSE", std_label="Corporate income tax expense",
         native_examples=["Chi phí thuế TNDN hiện hành", "Chi phí thuế TNDN hoãn lại"], vas_code="51", sign=-1),
    dict(statement=Statement.IS.value, std_code="IS_PROFIT_AFTER_TAX", std_label="Profit after tax",
         native_examples=["Lợi nhuận sau thuế TNDN", "Lợi nhuận sau thuế"], vas_code="60", sign=+1),
    dict(statement=Statement.IS.value, std_code="IS_EPS_BASIC", std_label="Basic earnings per share",
         native_examples=["Lãi cơ bản trên cổ phiếu"], vas_code="70", sign=+1),
    dict(statement=Statement.IS.value, std_code="IS_EPS_DILUTED", std_label="Diluted earnings per share",
         native_examples=["Lãi suy giảm trên cổ phiếu"], vas_code="71", sign=+1),

    # ======================================
    # BÁO CÁO LƯU CHUYỂN TIỀN TỆ (CASH FLOW)
    # ======================================
    dict(statement=Statement.CF.value, std_code="CF_CASH_FROM_OPERATIONS_B4_WC", std_label="Profit before tax & changes in working capital",
         native_examples=["Lợi nhuận trước thuế", "Lợi nhuận trước thay đổi vốn lưu động"], vas_code="08", sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_DEPRECIATION", std_label="Depreciation and amortization",
         native_examples=["Khấu hao tài sản cố định", "Khấu hao"], vas_code=None, sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_PROVISION_EXP", std_label="Provision expenses",
         native_examples=["Các khoản dự phòng"], vas_code=None, sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_INTEREST_EXPENSE", std_label="Interest expense",
         native_examples=["Chi phí lãi vay"], vas_code=None, sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_INTEREST_PAID", std_label="Interest paid",
         native_examples=["Tiền lãi vay đã trả"], vas_code="36", sign=-1),
    dict(statement=Statement.CF.value, std_code="CF_CIT_PAID", std_label="Corporate income tax paid",
         native_examples=["Thuế thu nhập doanh nghiệp đã nộp"], vas_code="34", sign=-1),

    # Lưu chuyển tiền thuần từ HĐKD
    dict(statement=Statement.CF.value, std_code="CF_NET_CFO", std_label="Net cash from operating activities",
         native_examples=["Lưu chuyển tiền thuần từ hoạt động kinh doanh"], vas_code="20", sign=+1),

    # Lưu chuyển tiền từ HĐĐT
    dict(statement=Statement.CF.value, std_code="CF_PURCHASE_PPE", std_label="Purchase of PPE",
         native_examples=["Tiền chi để mua sắm, xây dựng TSCĐ và các TS dài hạn khác"], vas_code="21", sign=-1),
    dict(statement=Statement.CF.value, std_code="CF_PROCEEDS_DISPOSAL_PPE", std_label="Proceeds from disposal of PPE",
         native_examples=["Tiền thu từ thanh lý, nhượng bán TSCĐ và các TS dài hạn khác"], vas_code="22", sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_ST_INVESTMENT", std_label="Payments for investments",
         native_examples=["Tiền chi đầu tư góp vốn vào đơn vị khác"], vas_code="25", sign=-1),
    dict(statement=Statement.CF.value, std_code="CF_INTEREST_DIVIDEND_RECEIVED", std_label="Interest and dividends received",
         native_examples=["Tiền lãi vay, cổ tức và lợi nhuận được chia đã nhận"], vas_code="27", sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_NET_CFI", std_label="Net cash from investing activities",
         native_examples=["Lưu chuyển tiền thuần từ hoạt động đầu tư"], vas_code="30", sign=+1),

    # Lưu chuyển tiền từ HĐTC
    dict(statement=Statement.CF.value, std_code="CF_BORROWINGS_OBTAINED", std_label="Proceeds from borrowings",
         native_examples=["Tiền thu từ đi vay"], vas_code="33", sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_BORROWINGS_REPAID", std_label="Repayments of borrowings",
         native_examples=["Tiền chi trả nợ gốc vay"], vas_code="34", sign=-1),
    dict(statement=Statement.CF.value, std_code="CF_DIVIDENDS_PAID", std_label="Dividends paid",
         native_examples=["Cổ tức, lợi nhuận đã trả cho chủ sở hữu"], vas_code="35", sign=-1),
    dict(statement=Statement.CF.value, std_code="CF_NET_CFF", std_label="Net cash from financing activities",
         native_examples=["Lưu chuyển tiền thuần từ hoạt động tài chính"], vas_code="40", sign=+1),

    # Tăng/giảm tiền & số dư đầu/cuối kỳ
    dict(statement=Statement.CF.value, std_code="CF_NET_INCREASE_CASH", std_label="Net increase/(decrease) in cash",
         native_examples=["Lưu chuyển tiền thuần trong kỳ"], vas_code="50", sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_CASH_BEGINNING", std_label="Cash at beginning of period",
         native_examples=["Tiền và tương đương tiền đầu kỳ"], vas_code="60", sign=+1),
    dict(statement=Statement.CF.value, std_code="CF_CASH_ENDING", std_label="Cash at end of period",
         native_examples=["Tiền và tương đương tiền cuối kỳ"], vas_code="70", sign=+1),
]

def run():
    with engine.begin() as conn:
        for item in INITIAL_ITEMS:
            conn.execute(
                text("""
                INSERT INTO line_items(statement, std_code, std_label, native_examples, vas_code, sign)
                VALUES (:statement, :std_code, :std_label, :native_examples, :vas_code, :sign)
                ON CONFLICT (statement, std_code) DO NOTHING
                """),
                item
            )
    print("\u2714 Seeded line_items (comprehensive)")

if __name__ == "__main__":
    run()
