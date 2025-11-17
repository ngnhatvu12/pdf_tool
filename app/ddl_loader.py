from sqlalchemy import text, inspect
from .db import engine
from pathlib import Path

def apply_schema():
    ddl = Path(__file__).resolve().parents[1] / "csdl" / "csdl.txt"
    sql = ddl.read_text(encoding="utf-8")
    
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    for stmt in filter(None, [s.strip() for s in sql.split(";\n")]):
        # Kiểm tra nếu là CREATE TABLE
        if stmt.upper().startswith("CREATE TABLE"):
            table_name = extract_table_name(stmt)
            if table_name and table_name in existing_tables:
                print(f"⚠️ Bảng {table_name} đã tồn tại, bỏ qua...")
                continue
        
        # Kiểm tra nếu là CREATE TYPE
        if stmt.upper().startswith("CREATE TYPE"):
            type_name = extract_type_name(stmt)
            if type_name and type_exists(engine, type_name):
                print(f"⚠️ Type {type_name} đã tồn tại, bỏ qua...")
                continue
        
        # Kiểm tra nếu là CREATE INDEX
        if stmt.upper().startswith("CREATE INDEX"):
            index_name = extract_index_name(stmt)
            if index_name and index_exists(engine, index_name):
                print(f"⚠️ Index {index_name} đã tồn tại, bỏ qua...")
                continue
        
        # Thực thi mỗi câu lệnh trong transaction riêng
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
            object_name = stmt.split()[2] if stmt.upper().startswith('CREATE') else 'object'
            print(f"✅ Đã tạo: {object_name}")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"⚠️ Đã tồn tại: {stmt.split()[2] if stmt.upper().startswith('CREATE') else 'object'}")
            else:
                print(f"❌ Lỗi khi thực thi: {e}")

def extract_index_name(create_index_stmt):
    """Extract index name từ câu lệnh CREATE INDEX"""
    try:
        parts = create_index_stmt.upper().split()
        index_index = parts.index("INDEX") + 1
        index_name = parts[index_index].split("ON")[0].strip()
        return index_name.lower()
    except:
        return None

def index_exists(engine, index_name):
    """Kiểm tra xem index đã tồn tại chưa"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 1 FROM pg_indexes WHERE indexname = :index_name
            """), {"index_name": index_name})
            return result.scalar() is not None
    except:
        return False

def extract_table_name(create_table_stmt):
    """Extract table name từ câu lệnh CREATE TABLE"""
    try:
        parts = create_table_stmt.upper().split()
        table_index = parts.index("TABLE") + 1
        table_name = parts[table_index].split("(")[0].strip()
        return table_name.lower()
    except:
        return None

def extract_type_name(create_type_stmt):
    """Extract type name từ câu lệnh CREATE TYPE"""
    try:
        parts = create_type_stmt.upper().split()
        type_index = parts.index("TYPE") + 1
        type_name = parts[type_index].split("AS")[0].strip()
        return type_name.lower()
    except:
        return None

def type_exists(engine, type_name):
    """Kiểm tra xem type đã tồn tại chưa"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 1 FROM pg_type WHERE typname = :type_name
            """), {"type_name": type_name})
            return result.scalar() is not None
    except:
        return False

if __name__ == "__main__":
    apply_schema()
    print("✔ Schema applied from csdl.txt")