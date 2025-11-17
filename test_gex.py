# test_gex.py
from pathlib import Path
from app.etl.load import process_one_pdf
from app.ocr_processor import extract_tables_ocr
import pandas as pd

def test_gex():
    gex_path = Path("data/raw/GEX_Baocaotaichinh_Q2_2025_Hopnhat_28072025092612.pdf")
    
    print("🧪 TEST TRÍCH XUẤT GEX")
    print("=" * 50)
    
    # Trích xuất thô
    df = extract_tables_ocr(gex_path, pages=[9, 10, 11, 12, 13, 14])
    
    print(f"\n📊 KẾT QUẢ TRÍCH XUẤT:")
    print(f"Tổng số dòng: {len(df)}")
    
    if not df.empty:
        # Hiển thị tất cả các dòng
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        print(df[['page', 'vas_code', 'src_label', 'amount_current', 'amount_prior']].to_string())
    
    # Xử lý đầy đủ
    print(f"\n🚀 XỬ LÝ ĐẦY ĐỦ:")
    result = process_one_pdf(gex_path)
    print(f"Kết quả: {result}")

if __name__ == "__main__":
    test_gex()