# app/etl/notes_loader.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from ..db import SessionLocal
import re

PCT = re.compile(r"(\d{1,3}(?:[.,]\d+)?)[%]")

def _upsert_lender(session, name: str) -> Optional[int]:
    if not name: return None
    return session.execute(text("""
        INSERT INTO lenders(name) VALUES (:n)
        ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name
        RETURNING lender_id
    """), {"n": name}).scalar()

def _valid_invest_enum(session) -> List[str]:
    try:
        rows = session.execute(text("SELECT unnest(enum_range(NULL::invest_class))::text")).all()
        return [r[0] for r in rows]
    except Exception:
        return ["bond","equity","subsidiary","associate","term_deposit"]

def _coerce_rate(val: Optional[str]) -> Optional[float]:
    if not val: return None
    m = PCT.search(str(val))
    if not m: return None
    return float(m.group(1).replace(",", "."))

def insert_notes_bundle(report_id: int, bundle: Dict[str, List[Dict[str, Any]]]):
    """
    Idempotent-ish insert (không xóa cũ). Logic:
    - Bảng nào không có dữ liệu -> bỏ qua yên lặng
    - Validate enum cho 'investments'
    - Lưu page để trace
    """
    with SessionLocal() as s:
        s.begin()
        try:
            # 1) borrowings
            for r in bundle.get("borrowings", []):
                lender_id = _upsert_lender(s, r.get("lender"))
                s.execute(text("""
                    INSERT INTO borrowings(report_id, debt_type, lender_id, short_long, currency, principal, interest_rate, maturity_date, collateral, purpose, page)
                    VALUES (:rid, :dt, :lid, :sl, :cur, :pri, :rate, :mat, :col, :pur, :pg)
                """), dict(
                    rid=report_id, dt=r.get("debt_type"), lid=lender_id, sl=r.get("short_long"),
                    cur=r.get("currency") or "VND", pri=r.get("principal"),
                    rate=_coerce_rate(r.get("interest_rate")) if isinstance(r.get("interest_rate"), str) else r.get("interest_rate"),
                    mat=r.get("maturity"), col=r.get("collateral"), pur=r.get("purpose"), pg=r.get("page")
                ))

            # 2) inventories
            for r in bundle.get("inventories", []):
                s.execute(text("""
                    INSERT INTO inventories(report_id, category, gross_amount, provision, net_amount, page)
                    VALUES (:rid, :cat, :gross, :prov, :net, :pg)
                """), dict(
                    rid=report_id, cat=r.get("category"),
                    gross=r.get("gross_amount"), prov=r.get("provision"),
                    net=r.get("net_amount"), pg=r.get("page")
                ))

            # 3) receivables
            for r in bundle.get("receivables", []):
                s.execute(text("""
                    INSERT INTO receivables(report_id, receivable_type, counterparty, short_long, gross_amount, allowance, net_amount, aging_bucket, page)
                    VALUES (:rid, :rt, :cp, :sl, :gross, :alw, :net, :bucket, :pg)
                """), dict(
                    rid=report_id, rt=r.get("receivable_type"), cp=r.get("counterparty"),
                    sl=r.get("short_long"), gross=r.get("gross_amount"), alw=r.get("allowance"),
                    net=r.get("net_amount"), bucket=r.get("aging_bucket"), pg=r.get("page")
                ))

            # 4) ppe roll-forward
            for r in bundle.get("ppe_rollforward", []):
                s.execute(text("""
                    INSERT INTO ppe_rollforward(report_id, asset_class, opening_cost, additions, disposals, transfers, closing_cost,
                                                opening_accdep, depreciation, disposals_accdep, closing_accdep, net_book_value, page)
                    VALUES (:rid, :cls, :op, :add, :dis, :tr, :cl, :opd, :dep, :disd, :cld, :nbv, :pg)
                """), dict(
                    rid=report_id, cls=r.get("asset_class"), op=r.get("opening_cost"), add=r.get("additions"),
                    dis=r.get("disposals"), tr=r.get("transfers"), cl=r.get("closing_cost"),
                    opd=r.get("opening_accdep"), dep=r.get("depreciation"),
                    disd=r.get("disposals_accdep"), cld=r.get("closing_accdep"),
                    nbv=r.get("net_book_value"), pg=r.get("page")
                ))

            # 5) investments (validate enum)
            valid_enums = set(_valid_invest_enum(s))
            for r in bundle.get("investments", []):
                cls = r.get("invest_class")
                if cls not in valid_enums:
                    print(f"[investments] Skip enum không hợp lệ: {cls} -> {r.get('instrument')}")
                    continue
                s.execute(text("""
                    INSERT INTO investments(report_id, invest_class, instrument, original_cost, carrying_amount, fair_value, reserve, ownership_pct, page)
                    VALUES (:rid, :cls, :ins, :oc, :car, :fv, :res, :own, :pg)
                """), dict(
                    rid=report_id, cls=cls, ins=r.get("instrument"),
                    oc=r.get("original_cost"), car=r.get("carrying_amount"),
                    fv=r.get("fair_value"), res=r.get("reserve"),
                    own=r.get("ownership_pct"), pg=r.get("page")
                ))

            # 6) income taxes
            for r in bundle.get("income_taxes", []):
                s.execute(text("""
                    INSERT INTO income_taxes(report_id, current_tax_expense, deferred_tax_expense, tax_payable, deferred_tax_assets, deferred_tax_liabilities, effective_tax_rate, page)
                    VALUES (:rid, :cur, :deff, :pay, :dta, :dtl, :etr, :pg)
                """), dict(
                    rid=report_id, cur=r.get("current_tax_expense"), deff=r.get("deferred_tax_expense"),
                    pay=r.get("tax_payable"), dta=r.get("deferred_tax_assets"),
                    dtl=r.get("deferred_tax_liabilities"), etr=r.get("effective_tax_rate"),
                    pg=r.get("page")
                ))

            # 7) equity changes
            for r in bundle.get("equity_changes", []):
                s.execute(text("""
                    INSERT INTO equity_changes(report_id, component, opening, increase, decrease, closing, page)
                    VALUES (:rid, :comp, :op, :inc, :dec, :clo, :pg)
                """), dict(
                    rid=report_id, comp=r.get("component"), op=r.get("opening"),
                    inc=r.get("increase"), dec=r.get("decrease"), clo=r.get("closing"),
                    pg=r.get("page")
                ))

            # 8) per share data
            for r in bundle.get("per_share_data", []):
                s.execute(text("""
                    INSERT INTO per_share_data(report_id, basic_eps, diluted_eps, shares_outstanding, dividends_per_share, page)
                    VALUES (:rid, :b, :d, :s, :dp, :pg)
                """), dict(
                    rid=report_id, b=r.get("basic_eps"), d=r.get("diluted_eps"),
                    s=r.get("shares_outstanding"), dp=r.get("dividends_per_share"),
                    pg=r.get("page")
                ))

            # 9) related parties
            party_cache = {}
            for r in bundle.get("related_parties", []):
                nm = r.get("name")
                if not nm: continue
                if nm not in party_cache:
                    pid = s.execute(text("""
                        INSERT INTO related_parties(name) VALUES (:n)
                        ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name
                        RETURNING rp_id
                    """), {"n": nm}).scalar()
                    party_cache[nm] = pid
                s.execute(text("""
                    INSERT INTO related_party_transactions(report_id, rp_id, nature, amount, balance, page)
                    VALUES (:rid, :pid, :nat, :amt, :bal, :pg)
                """), dict(
                    rid=report_id, pid=party_cache[nm], nat=r.get("nature"), amt=r.get("amount"),
                    bal=r.get("balance"), pg=r.get("page")
                ))

            # 10) cashflows
            for r in bundle.get("cashflows", []):
                s.execute(text("""
                    INSERT INTO cashflows(report_id, section, item, amount, page)
                    VALUES (:rid, :sec, :it, :amt, :pg)
                """), dict(
                    rid=report_id, sec=r.get("section"), it=r.get("item"),
                    amt=r.get("amount"), pg=r.get("page")
                ))

            s.commit()
            print("✅ Notes inserted/updated.")
        except Exception as e:
            s.rollback()
            print(f"❌ Notes load error: {e}")
            raise
