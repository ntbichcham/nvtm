# Brand Health Demo — Hướng dẫn website

## 1. Trạng thái

Đã hoàn thành phần nền của website:

- Kết nối MySQL bằng PDO, prepared statement.
- Đăng nhập / đăng xuất bằng PHP session.
- Mật khẩu kiểm tra bằng `password_verify`, lưu bằng bcrypt cost 12.
- CSRF token cho mọi form POST.
- Layout sidebar + topbar, giao diện Bootstrap 5, tiếng Việt.
- Trang tổng quan, danh sách đánh giá (tìm kiếm, lọc, phân trang), thêm đánh giá,
  xem chi tiết, xóa, xem 10 khía cạnh, đổi mật khẩu.
- Chặn truy cập khi chưa đăng nhập, chặn CSRF, escape toàn bộ output.
- Các file nhạy cảm (`app/`, `database/`, `python_api/`) nằm ngoài thư mục public.

Giai đoạn 5 đã nối Python API:

- Nút **Phân tích bằng PhoBERT** trên trang chi tiết gọi `POST /predict`.
- Lưu kết quả vào `analysis_results`, bỏ qua nhãn `None`.
- Cập nhật `analysis_status`, `model_name`, `model_version`, `inference_ms`, `analyzed_at`.
- Phân tích lại sẽ thay thế kết quả cũ trong một transaction.
- Nút **Phân tích đang chờ** xử lý tối đa 20 đánh giá mỗi lần.
- Lỗi API/timeout được ghi log và hiển thị thông báo an toàn, không lộ chi tiết kỹ thuật.

Giai đoạn 6 đã bổ sung BHI và biểu đồ:

- Tính điểm từng khía cạnh theo Tích cực = 1,0; Trung tính = 0,5; Tiêu cực = 0.
- Tính BHI có trọng số, chuẩn hóa lại trên các khía cạnh có dữ liệu.
- Hiển thị độ phủ `x/10`, N/A cho khía cạnh thiếu dữ liệu.
- Biểu đồ Chart.js: điểm khía cạnh, phân bố cảm xúc, xu hướng BHI theo tháng.
- Bộ lọc thời gian (7/30/90 ngày hoặc khoảng ngày tùy chọn).

Còn lại: Giai đoạn 7 — hoàn thiện tài liệu, kiểm thử tổng thể và kịch bản demo.

---

## 2. Cấu trúc thư mục

```text
brand_health_demo/
├── app/
│   ├── bootstrap.php          # cấu hình, session, header bảo mật, nạp PDO
│   ├── config.php             # DB/API config
│   ├── db.php                 # PDO singleton
│   ├── helpers.php            # escape, URL, flash, CSRF, format
│   ├── auth.php               # đăng nhập, phân quyền, logout
│   ├── api_client.php         # client gọi Python API, kiểm tra phản hồi
│   ├── analysis.php           # điều phối phân tích + transaction
│   ├── bhi.php                # tính BHI, độ phủ, xu hướng tháng
│   └── views/
│       ├── layout_header.php
│       └── layout_footer.php
├── public/                    # thư mục DUY NHẤT được Apache phục vụ
│   ├── index.php              # tổng quan
│   ├── login.php
│   ├── logout.php
│   ├── reviews.php
│   ├── review_create.php
│   ├── review_view.php
│   ├── review_delete.php
│   ├── review_analyze.php     # phân tích một đánh giá
│   ├── review_analyze_pending.php # phân tích theo lô
│   ├── aspects.php
│   ├── change_password.php
│   ├── .htaccess
│   └── assets/
│       ├── css/app.css
│       ├── js/app.js
│       ├── js/dashboard.js    # biểu đồ Chart.js
│       └── vendor/            # Bootstrap 5.3.3, Chart.js 4.4.7 (bản local)
├── storage/logs/              # log PHP, không public
├── database/                  # schema + README
├── python_api/                # Giai đoạn 3
├── run_web.ps1                # khởi động MySQL + Apache
└── run_api.ps1                # khởi động Python API
```

---

## 3. Cấu hình Apache

Website dùng Alias để chỉ phục vụ thư mục `public/`, không copy mã nguồn vào `htdocs`.

Đã tạo:

```text
D:\xamp\apache\conf\extra\brandhealth.conf
```

Nội dung:

```apache
Alias /brand_health_demo "D:/Ct255-NVTM(BI)/nhom2/brand_health_demo/public"

<Directory "D:/Ct255-NVTM(BI)/nhom2/brand_health_demo/public">
    Options -Indexes +FollowSymLinks
    AllowOverride All
    Require all granted
</Directory>
```

Và thêm dòng sau vào cuối `D:\xamp\apache\conf\httpd.conf`:

```apache
Include conf/extra/brandhealth.conf
```

Kiểm tra cấu hình:

```powershell
& "D:\xamp\apache\bin\httpd.exe" -t
```

Kết quả mong đợi: `Syntax OK`.

> Nếu chuyển sang máy khác hoặc đổi đường dẫn dự án, sửa lại `Alias` và `<Directory>`
> trong `brandhealth.conf` rồi khởi động lại Apache.

---

## 4. Chạy website

Cách nhanh (không vướng Execution Policy):

```powershell
cd "D:\Ct255-NVTM(BI)\nhom2\brand_health_demo"
.\run_web.bat
```

Hoặc gọi trực tiếp PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_web.ps1
```

> Nếu PowerShell báo "running scripts is disabled on this system", hãy dùng file `.bat`
> hoặc thêm `-ExecutionPolicy Bypass` như trên.

Script sẽ:

1. Khởi động MySQL nếu cổng 3306 chưa mở.
2. Khởi động Apache nếu cổng 80 chưa mở.
3. Kiểm tra database `brand_health_db`.
4. In URL và tài khoản đăng nhập.

Sau đó mở:

```text
http://localhost/brand_health_demo/
```

Để phân tích bằng PhoBERT, mở thêm một cửa sổ và chạy Python API:

```powershell
cd "D:\Ct255-NVTM(BI)\nhom2\brand_health_demo"
.\run_api.bat
```

Chờ đến khi log báo `PhoBERT ABSA ready` (lần đầu khoảng 30–90 giây), rồi quay lại website.
Nếu API chưa chạy, website vẫn hoạt động nhưng nút phân tích sẽ báo lỗi an toàn.

Có thể dùng XAMPP Control Panel thay cho script, chỉ cần bật **MySQL** và **Apache**.

---

## 5. Đăng nhập

| Trường | Giá trị |
|---|---|
| URL | `http://localhost/brand_health_demo/login.php` |
| Username | `admin` |
| Password | `Admin@123` |

Đổi mật khẩu tại menu **Đổi mật khẩu**. Mật khẩu mới được băm bcrypt cost 12.

---

## 6. Các trang đã có

| Trang | URL | Chức năng |
|---|---|---|
| Tổng quan | `/index.php` | BHI, độ phủ, 3 biểu đồ, bộ lọc thời gian, chi tiết khía cạnh, đánh giá gần đây |
| Đánh giá | `/reviews.php` | Tìm theo nội dung, lọc trạng thái, phân trang 10 dòng |
| Thêm đánh giá | `/review_create.php` | Nhập nội dung, số sao, ngày; lưu trạng thái `pending` |
| Chi tiết | `/review_view.php?id=...` | Nội dung, metadata, nút **Phân tích bằng PhoBERT**, kết quả ABSA |
| Khía cạnh | `/aspects.php` | 10 khía cạnh, trọng số cố định, lượt nhắc, điểm, N/A |
| Đổi mật khẩu | `/change_password.php` | Kiểm tra mật khẩu hiện tại, cập nhật bcrypt |

---

## 7. Kiểm chứng đã chạy thực tế

Môi trường: XAMPP tại `D:\xamp` (PHP 8.0.30, MariaDB 10.4.32, Apache 2.4.58).

| Kiểm tra | Kết quả |
|---|---|
| Import `brand_health_db.sql` | 5 bảng, 10 khía cạnh, 4 nhãn, 1 user |
| `password_verify('Admin@123', hash)` | `true` |
| Truy cập khi chưa đăng nhập | `302 → login.php` |
| Đăng nhập đúng | `302 → index.php` |
| Đăng nhập sai | `200`, báo "Tên đăng nhập hoặc mật khẩu không đúng." |
| Thêm đánh giá | `302 → review_view.php?id=1`, dữ liệu tiếng Việt lưu đúng |
| Tìm kiếm + lọc trạng thái | Đúng số bản ghi, không lẫn dòng khác |
| Xóa đánh giá | `302 → reviews.php`, bản ghi bị xóa |
| CSRF sai | `403`, hiển thị thông báo hết phiên |
| Đổi mật khẩu rồi đổi lại | Đăng nhập được bằng cả mật khẩu mới và mật khẩu gốc sau khi phục hồi |
| Đăng xuất | `302 → login.php`, session mất hiệu lực |
| Trang chi tiết ID không tồn tại | `404` |
| Gọi `/predict` thật cho đánh giá #5 | `completed`, 210 ms, 2 khía cạnh |
| Kết quả #5 | CAMERA Negative 0.93519, PERFORMANCE Positive 0.78479 |
| Phân tích theo lô 2 đánh giá | Cả hai `completed`, lần lượt 3 và 2 khía cạnh |
| Phân tích khi API tắt | `failed`, thông báo an toàn, không lộ lỗi cURL |
| PDO transaction khi lưu kết quả | Xóa kết quả cũ + ghi mới + cập nhật review trong một transaction |
| BHI với dữ liệu hiện tại (#5) | `50,0/100`, phủ `2/10`, 2 lượt khía cạnh |
| Kiểm tra công thức với 3 tháng dữ liệu thử | 09/2026 = 100; 10/2026 = 0; 11/2026 = 50; BHI tổng = `62,5` |
| Bộ lọc `from=2026-10-01` | BHI `25,0` (chỉ Camera âm và Hiệu năng) |
| Bộ lọc sau `2026-11-15` | `N/A`, độ phủ `0/10`, ẩn toàn bộ biểu đồ |
| Ngày sai / khoảng ngược | Hiện cảnh báo và bỏ qua bộ lọc |
| Cú pháp `dashboard.js` | `node --check` thành công |
| Dữ liệu biểu đồ JSON | Parse thành công, đúng nhãn và giá trị |

Dữ liệu thử cho kiểm tra 3 tháng đã được xóa; database hiện có 1 đánh giá của người dùng
(id #5) đã phân tích thành công.

---

## 8. Bảo mật đã áp dụng

- `PDO::ATTR_EMULATE_PREPARES = false`, mọi truy vấn có tham số dùng prepared statement.
- Escape output bằng `htmlspecialchars(..., ENT_QUOTES, 'UTF-8')`.
- CSRF token 32 byte cho toàn bộ POST, so sánh bằng `hash_equals`.
- `session_regenerate_id(true)` sau đăng nhập và sau đổi mật khẩu.
- Cookie session `HttpOnly`, `SameSite=Lax`.
- Giới hạn 5 lần đăng nhập sai trong 5 phút.
- Không hiển thị lỗi PDO/stack trace ra trình duyệt; ghi vào `storage/logs/php-error.log`.
- `public/.htaccess` chặn liệt kê thư mục và file ẩn.
- Không đưa checkpoint, mã Python, SQL vào thư mục public.

---

## 9. Lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| `404 Not Found` khi mở `/brand_health_demo/` | Chưa thêm `Include conf/extra/brandhealth.conf` hoặc chưa restart Apache |
| `Không kết nối được cơ sở dữ liệu` | MySQL chưa chạy hoặc chưa import `brand_health_db.sql` |
| Trang trắng | Xem `storage/logs/php-error.log` và `D:\xamp\apache\logs\error.log` |
| `403` khi gửi form | CSRF hết hạn do session bị xóa; tải lại trang rồi gửi lại |
| `#1045 Access denied` | `app/config.php` sai user/mật khẩu MySQL |
| CSS/JS không tải | Thiếu thư mục `public/assets/vendor/`; tải lại Bootstrap/Chart.js |
| Sai giờ / ngày | PHP dùng `Asia/Ho_Chi_Minh`, MySQL `SET time_zone='+07:00'` |
| `running scripts is disabled` | Dùng `run_web.bat` hoặc `powershell -ExecutionPolicy Bypass -File .\run_web.ps1` |

---

## 10. Giai đoạn 5 — Phân tích bằng PhoBERT

### Luồng xử lý

1. Người dùng bấm **Phân tích bằng PhoBERT** → `POST review_analyze.php` kèm `id` và CSRF.
2. `analyze_review()` đánh dấu review `processing`.
3. `api_predict()` gọi `POST http://127.0.0.1:8000/predict` với `text` và `review_id`.
4. API trả 10 khía cạnh (chỉ dùng các khía cạnh khác `None`).
5. PHP kiểm tra cấu trúc phản hồi, ánh xạ `aspect.code → aspects.id` và
   `sentiment → sentiment_labels.id`.
6. Trong một transaction:
   - Xóa kết quả cũ của review.
   - Chèn các dòng mới vào `analysis_results`.
   - Cập nhật `analysis_status = 'completed'`, `analyzed_at`, `model_name`,
     `model_version`, `inference_ms`.
7. Chuyển về trang chi tiết kèm thông báo số khía cạnh.

Phân tích lại sẽ thay thế kết quả cũ, không tạo bản ghi trùng.

### File mới

| File | Chức năng |
|---|---|
| `app/api_client.php` | Gọi API bằng cURL, timeout, API key, kiểm tra JSON và mã khía cạnh/cảm xúc |
| `app/analysis.php` | Điều phối trạng thái, transaction, ánh xạ code → id |
| `public/review_analyze.php` | Endpoint POST phân tích một đánh giá |
| `public/review_analyze_pending.php` | Endpoint POST phân tích tối đa 20 đánh giá đang chờ |

### Xử lý lỗi

| Tình huống | Trạng thái review | Thông báo cho người dùng |
|---|---|---|
| API không chạy | `failed` (nếu chưa có kết quả) | "Không kết nối được dịch vụ phân tích PhoBERT..." |
| API quá thời gian | `failed` | "Dịch vụ phân tích phản hồi quá lâu..." |
| Model chưa nạp (`503`) | `failed` | "Mô hình PhoBERT chưa sẵn sàng..." |
| Phản hồi sai cấu trúc | `failed` | "Phản hồi của dịch vụ phân tích không hợp lệ." |
| Lỗi khi ghi database | `failed` | "Không lưu được kết quả phân tích." |
| Phân tích lại thất bại nhưng đã có kết quả cũ | Giữ `completed` | Cảnh báo lỗi, không xóa kết quả cũ |

Chi tiết kỹ thuật (HTTP status, lỗi cURL, mã khía cạnh lạ) chỉ ghi vào
`storage/logs/php-error.log`, không hiển thị ra trình duyệt.

### Cấu hình

Trong `app/config.php`:

```php
'api' => [
    'base_url' => 'http://127.0.0.1:8000',
    'key'      => '',   // khớp API_KEY trong python_api/.env
    'timeout'  => 30,
],
```

- `base_url` phải trỏ đúng địa chỉ Python API.
- Nếu đặt `API_KEY` trong `python_api/.env`, phải điền cùng giá trị vào `key`.
- Không đưa API key hay đường dẫn checkpoint vào database.

### Cách thử

1. Chạy `.\run_api.bat`, chờ log `PhoBERT ABSA ready`.
2. Mở một đánh giá đang `pending`.
3. Bấm **Phân tích bằng PhoBERT**.
4. Kiểm tra bảng kết quả và trang **Khía cạnh** (cột Lượt nhắc tăng).
5. Muốn xử lý nhiều đánh giá: ở trang **Đánh giá**, bấm **Phân tích đang chờ**.

### Ghi chú về trọng số

Trọng số BHI vẫn cố định `0,10` cho mỗi khía cạnh và **không thay đổi** khi thêm
hoặc phân tích đánh giá. Đây là trọng số cấu hình dùng cho công thức BHI ở Giai đoạn 6.
Cột **Lượt nhắc** mới là số liệu thay đổi theo kết quả phân tích.

---

## 11. Giai đoạn 6 — BHI và biểu đồ

### Công thức

| Thành phần | Công thức |
|---|---|
| Điểm cảm xúc | Tích cực = `1,0`; Trung tính = `0,5`; Tiêu cực = `0,0` |
| Điểm khía cạnh | Trung bình điểm cảm xúc của tất cả lượt nhắc × 100 |
| BHI | `Σ(w × điểm khía cạnh) / Σ(w)` trên các khía cạnh **có dữ liệu** |
| Độ phủ | Số khía cạnh có dữ liệu / 10 |
| Trọng số dùng | Tổng `w` của các khía cạnh có dữ liệu |

Ví dụ với đánh giá #5:

- `CAMERA` = Tiêu cực → điểm `0`.
- `PERFORMANCE` = Tích cực → điểm `100`.
- BHI = `(0,1 × 0 + 0,1 × 100) / (0,1 + 0,1)` = **50,0**.
- Độ phủ = **2/10** (20%), trọng số dùng = **0,20**.

### Xử lý khía cạnh thiếu dữ liệu

- Khía cạnh không có lượt nhắc nào → điểm **N/A**.
- Không đưa khía cạnh N/A vào tử số và mẫu số.
- Trọng số được chuẩn hóa lại trên các khía cạnh còn lại, nên BHI vẫn nằm trong 0–100.
- Nếu cả 10 khía cạnh đều N/A → BHI = **N/A**, giao diện hiển thị "Chưa đủ dữ liệu".
- Độ phủ giúp người xem biết BHI đang dựa trên bao nhiêu phần của danh mục.

### Bộ lọc thời gian

- Lọc theo `reviews.review_date`, áp dụng cho BHI, độ phủ, biểu đồ và danh sách gần đây.
- Preset: **7 ngày**, **30 ngày**, **90 ngày**, **Tất cả**.
- Có thể nhập khoảng ngày tùy chọn.
- Ngày sai định dạng hoặc `from > to` sẽ hiện cảnh báo và bỏ qua bộ lọc.

### Biểu đồ

| Biểu đồ | Nội dung |
|---|---|
| Cột | Điểm từng khía cạnh có dữ liệu, thang 0–100 |
| Tròn (doughnut) | Số lượt Tích cực / Trung tính / Tiêu cực |
| Đường | BHI theo tháng của `review_date` |

Màu điểm: xanh `≥ 70`, vàng `50–69,9`, đỏ `< 50`. Chart.js được tải cục bộ,
chỉ nạp ở trang Tổng quan.

### File liên quan

| File | Chức năng |
|---|---|
| `app/bhi.php` | Tổng hợp, tính điểm khía cạnh, BHI, độ phủ, xu hướng tháng |
| `public/index.php` | Dashboard BHI, bộ lọc, bảng chi tiết, dữ liệu biểu đồ |
| `public/aspects.php` | Thêm cột điểm và N/A cho từng khía cạnh |
| `public/assets/js/dashboard.js` | Khởi tạo 3 biểu đồ Chart.js |

### Lưu ý

- BHI chỉ dùng kết quả ABSA, **không dùng số sao** hay rating.
- Điểm khía cạnh là trung bình theo **lượt nhắc**, không phải trung bình theo đánh giá.
- Phân tích lại một đánh giá sẽ thay đổi điểm khía cạnh và BHI tương ứng.
- Xu hướng tháng cần ít nhất 2 tháng dữ liệu để thấy rõ; giao diện có ghi chú khi chỉ có 1 tháng.

---

## 12. Giai đoạn 7 — Kiểm thử, báo cáo và bàn giao

### Báo cáo riêng

Báo cáo Giai đoạn 7 được ghi thành **file riêng**, không chỉnh sửa báo cáo chính:

| File | Nội dung |
|---|---|
| `BAO_CAO_GIAI_DOAN_7_WEBSITE_BHI.docx` | Báo cáo đầy đủ 12 trang (12 mục + 3 phụ lục) |
| `BAO_CAO_GIAI_DOAN_7_WEBSITE_BHI.pdf` | Bản PDF để nộp/in |
| `brand_health_demo/docs/BAO_CAO_GIAI_DOAN_7.md` | Bản Markdown tóm tắt |
| `brand_health_demo/docs/generate_phase7_report.py` | Script tạo lại file docx |

Báo cáo chính `baocao_NVTM_CT255.docx` không bị mở để ghi hay sửa; SHA-256 tại thời điểm
lập báo cáo là `905B6B43EA607574DC283E7C518DBF5CDD69571679BE2AC3C425112E4D5ED192`.

### Kiểm thử tự động

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  D:\Ct255-NVTM(BI)\nhom2\brand_health_demo\docs\phase7_e2e_test.ps1
```

Script tạo 02 đánh giá tạm, phân tích bằng PhoBERT thật, kiểm tra dashboard, validation,
XSS, SQL injection, file riêng tư, xóa và đăng xuất; cuối cùng tự dọn bản ghi tạm.
Kết quả lần chạy ngày 14/11/2026: **20/20 kiểm tra đạt**, CSDL trở về 02 đánh giá.

Các nhóm đã kiểm chứng:

| Nhóm | Kết quả |
|---|---|
| End-to-end (đăng nhập → tạo → phân tích → BHI → xóa → đăng xuất) | Đạt |
| Validation (nội dung rỗng, ngày tương lai, ngày sai, sao sai) | Đạt |
| Bảo mật (CSRF 403, XSS escape, SQLi 200 không lỗi, file riêng tư 404) | Đạt |
| Sao lưu và phục hồi CSDL | Khớp 100% số dòng |

### Sao lưu và phục hồi

```bat
set BACKUP_DIR=D:\Ct255-NVTM(BI)\nhom2\brand_health_demo\database\backups
D:\xamp\mysql\bin\mysqldump.exe -u root --default-character-set=utf8mb4 ^
  --single-transaction --skip-lock-tables --databases brand_health_db ^
  --result-file="%BACKUP_DIR%\brand_health_db_YYYYMMDD_HHMMSS.sql"
```

Phục hồi:

```bat
D:\xamp\mysql\bin\mysql.exe -u root --default-character-set=utf8mb4 ^
  < "%BACKUP_DIR%\brand_health_db_20261114_175235.sql"
```

> Dùng `--result-file` (hoặc chuyển hướng trong `cmd.exe`). Toán tử `>` của PowerShell 5.1
> ghi file UTF-16 và `mysql` sẽ báo lỗi `ASCII '\0' appeared in the statement`.

File backup chứa hash bcrypt và nội dung đánh giá nên **không commit**; thư mục
`database/backups/` đã được thêm vào `.gitignore`.

### Bàn giao

- Kịch bản demo 10 phút: mục 8 của báo cáo.
- Checklist chuyển máy 12 hạng mục: mục 7.3 của báo cáo.
- Danh mục ảnh chụp màn hình cần chèn: mục 9 của báo cáo.
- Hạn chế và hướng phát triển: mục 10–11 của báo cáo.