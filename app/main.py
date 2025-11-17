from pathlib import Path
import argparse

from app.ddl_loader import apply_schema
from app.seed_line_items import run as seed_line_items
from app.etl.load import process_one_pdf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ocr",
        choices=["tesseract", "paddle"],
        default="tesseract",
        help="Chọn engine OCR: tesseract hoặc paddle",
    )
    args = parser.parse_args()

    print(f"🔧 Sử dụng OCR engine: {args.ocr}")

    apply_schema()
    seed_line_items()

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    for fp in data_dir.glob("*.pdf"):
        try:
            info = process_one_pdf(fp, ocr_engine=args.ocr)
            print(f"✔ Loaded {fp.name}: {info}")
        except Exception as e:
            print(f"[ERROR] {fp.name}: {e}")

if __name__ == "__main__":
    main()
