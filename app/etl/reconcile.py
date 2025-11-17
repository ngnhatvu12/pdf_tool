from pathlib import Path
import pandas as pd
from sqlalchemy import text
from ..db import SessionLocal
from app.ocr_processor import extract_tables_ocr
from .normalize import choose_line_item

def reconcile_report(report_id: int, pdf_path: Path):
    with SessionLocal() as s:
        s.begin()
        try:
            df = extract_tables_ocr(pdf_path, pages=None)

            cand = [(r[0], r[1] or [], r[2]) for r in s.execute(
                text("SELECT std_code, native_examples, vas_code FROM line_items")
            ).all()]

            # Long-form: map std_code ngay tại đây
            if {"context_key","amount"} <= set(df.columns):
                df["std_code"] = df.apply(
                    lambda r: choose_line_item(r["src_label"], r.get("vas_code"), cand),
                    axis=1
                )
                # đọc facts hiện có
                facts = s.execute(text("""
                    SELECT f.fact_id, f.line_item_id, li.std_code, f.context, f.amount, f.page, f.src_label
                    FROM facts f JOIN line_items li ON li.line_item_id=f.line_item_id
                    WHERE f.report_id=:rid
                """), {"rid": report_id}).mappings().all()

                def key(std, ctx): return f"{std}|{ctx}"
                newmap = {}
                for _, r in df.dropna(subset=["std_code"]).iterrows():
                    ctx = str(r["context_key"])
                    newmap[key(r["std_code"], ctx)] = (r["amount"], r["page"], r["src_label"])

                changed = 0
                for row in facts:
                    k = key(row["std_code"], row["context"])
                    if k in newmap:
                        new_amt, new_pg, new_lbl = newmap[k]
                        if row["amount"] is None or abs(float(row["amount"]) - float(new_amt)) > max(1.0, abs(float(new_amt))*0.01):
                            s.execute(text("""
                              UPDATE facts SET amount=:a, page=:p, src_label=:l WHERE fact_id=:id
                            """), {"a": new_amt, "p": new_pg, "l": new_lbl, "id": row["fact_id"]})
                            changed += 1
                        newmap.pop(k)

                for k, (val, pg, lbl) in newmap.items():
                    std, ctx = k.split("|", 1)
                    lid = s.execute(text(
                        "SELECT line_item_id FROM line_items WHERE std_code=:c"), {"c": std}
                    ).scalar_one()
                    s.execute(text("""
                      INSERT INTO facts(report_id, line_item_id, context, amount, unit, page, src_label, confidence)
                      VALUES (:rid, :lid, :ctx, :amt, 'VND', :pg, :lbl, 0.85)
                      ON CONFLICT DO NOTHING
                    """), {"rid": report_id, "lid": lid, "ctx": ctx, "amt": val, "pg": pg, "lbl": lbl})

                s.commit()
                print(f"✔ Reconciled report {report_id}. Updated {changed} rows, inserted {len(newmap)} rows.")
                return

            # Fallback: wide-form cũ (nếu vì lý do nào đó OCR chỉ bắt 2 cột)
            # ... (khối cũ của bạn có thể giữ nguyên) ...

        except Exception as e:
            s.rollback()
            print(f"❌ Reconcile error: {e}")
            raise