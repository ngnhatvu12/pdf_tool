📦 1. Tải về (Download Release)

👉 Vào trang Releases:
https://github.com/**YOUR_GITHUB_REPO**/releases

Tải file:

BaoCaoPDF.zip

Giải nén ra thư mục bất kỳ:

BaoCaoPDF/
 ├── BaoCaoPDF.exe
 ├── tesseract/
 ├── tessdata/
 ├── poppler_bin/
 ├── csdl/
 ├── .env.example
 └── ...

Sau khi giải nén bạn có thể chạy luôn BaoCaoPDF.exe

⚙️ 2. Cấu hình kết nối database (nếu cần)

Ứng dụng hỗ trợ 2 chế độ:

🟢 A) Chạy không cần database (khuyến nghị cho người dùng bình thường)

Chỉ cần sửa file .env như sau:

1️⃣ Copy file mẫu:

Đổi tên .env.example → .env

2️⃣ Bật chế độ offline:
APP_ENV=client


→ Chế độ này tắt hoàn toàn PostgreSQL, bạn có thể chạy ứng dụng mà không cần cài DB hoặc server.

🔵 B) Chạy với PostgreSQL (nếu bạn có server database riêng)

Sửa file .env:

APP_ENV=prod
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?connect_timeout=5


Ví dụ:

DATABASE_URL=postgresql+psycopg://baocao:123456@10.10.10.5:5432/aimdb?connect_timeout=5


📌 Lưu ý quan trọng

USER, PASSWORD, HOST phải đúng với server PostgreSQL của bạn

Nếu sai thông tin đăng nhập sẽ xuất hiện lỗi:
password authentication failed for user ...

Nếu không muốn dùng database → đặt APP_ENV=client

▶️ 3. Chạy ứng dụng

Chạy file:

BaoCaoPDF.exe


Nếu lần đầu chạy Windows SmartScreen cảnh báo, nhấn:

More info → Run anyway

📋 4. Hướng dẫn sử dụng

1️⃣ Chọn file PDF báo cáo tài chính

Ứng dụng hỗ trợ:

PDF dạng scan (OCR)

PDF dạng text

Báo cáo VAS, BCTC quý, bán niên, năm

2️⃣ Nhấn “Trích xuất”

Ứng dụng sẽ:

OCR tiếng Việt + tiếng Anh (Tesseract)

Nhận diện nhiều cột tài chính

Ghép số bị tách (ví dụ: 9.0, 23 → 9.023)

Xác định loại giá trị: Current, Prior, As-of, YTD, Quý

Chuẩn hoá về VND

