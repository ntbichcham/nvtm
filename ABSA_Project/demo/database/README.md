# Giai đoạn 2 — Thiết kế cơ sở dữ liệu MySQL

## 1. File cần dùng

```text
brand_health_demo/database/brand_health_db.sql
```

File này tạo database `brand_health_db`, 5 bảng chính, khóa ngoại, chỉ mục và dữ liệu danh mục ban đầu.

---

## 2. Sơ đồ quan hệ

```text
users (1) ─────< (n) reviews
                         │
                         │ 1
                         ▼
                    (n) analysis_results (n) >───── (1) aspects
                              │
                              │ n
                              ▼
                         (1) sentiment_labels
```

Quan hệ chính:

- Một `user` có thể tạo nhiều `review`.
- Một `review` có nhiều `analysis_results`.
- Mỗi `analysis_result` thuộc một `aspect`.
- Mỗi `analysis_result` có một `sentiment_label`.
- Xóa `review` sẽ tự động xóa các `analysis_results` của review đó.
- Không thể xóa `aspect` hoặc `sentiment_label` đang được sử dụng.

---

## 3. Giải thích các bảng

### 3.1. `users`

Lưu tài khoản đăng nhập quản trị.

| Cột | Ý nghĩa |
|---|---|
| `username` | Tên đăng nhập, duy nhất |
| `password_hash` | Mật khẩu đã băm bằng `password_hash()` của PHP |
| `full_name` | Tên hiển thị |
| `role` | `admin` hoặc `analyst` |
| `is_active` | Khóa/mở tài khoản |
| `last_login_at` | Lần đăng nhập gần nhất |

Không lưu mật khẩu dạng văn bản thuần.

### 3.2. `aspects`

Danh mục 10 khía cạnh của UIT-ViSFD.

| Cột | Ý nghĩa |
|---|---|
| `code` | Mã khía cạnh, ví dụ `BATTERY` |
| `name_vi` | Tên tiếng Việt để hiển thị |
| `model_index` | Vị trí đầu ra của mô hình, từ `0` đến `9` |
| `weight` | Trọng số `w(a)` dùng cho BHI, ban đầu `0.1` |
| `display_order` | Thứ tự hiển thị trên giao diện |

`model_index` phải giữ đúng thứ tự:

```text
0 BATTERY
1 CAMERA
2 DESIGN
3 FEATURES
4 GENERAL
5 PERFORMANCE
6 PRICE
7 SCREEN
8 SER&ACC
9 STORAGE
```

### 3.3. `sentiment_labels`

Danh mục 4 nhãn cảm xúc.

| `model_index` | `code` | Ý nghĩa |
|---:|---|---|
| 0 | `None` | Không đề cập khía cạnh |
| 1 | `Negative` | Tiêu cực |
| 2 | `Neutral` | Trung tính |
| 3 | `Positive` | Tích cực |

`is_mention = 0` chỉ dành cho nhãn `None`.

### 3.4. `reviews`

Lưu nội dung đánh giá gốc và trạng thái phân tích.

| Cột | Ý nghĩa |
|---|---|
| `content` | Nội dung đánh giá tiếng Việt |
| `rating` | Số sao 1–5, có thể `NULL` |
| `review_date` | Ngày đánh giá, dùng để lọc thời gian |
| `source` | Nguồn dữ liệu: `manual`, `csv`, `uit_visfd`... |
| `external_id` | ID gốc khi import CSV; tránh import trùng |
| `analysis_status` | `pending`, `processing`, `completed`, `failed` |
| `analysis_error` | Thông báo lỗi an toàn cho người dùng |
| `analyzed_at` | Thời điểm phân tích xong |
| `model_name` | Tên mô hình, không phải đường dẫn checkpoint |
| `model_version` | Phiên bản/logic suy luận |
| `inference_ms` | Thời gian suy luận |

Ràng buộc:

```text
UNIQUE (source, external_id)
```

Cho phép nhiều review nhập tay có `external_id = NULL`, nhưng không cho import trùng cùng một ID từ cùng một nguồn.

### 3.5. `analysis_results`

Lưu kết quả thật của mô hình PhoBERT.

Mỗi dòng là một cặp:

```text
(review, aspect) → sentiment + confidence
```

Ví dụ:

| review_id | aspect_id | sentiment_label_id | confidence |
|---:|---:|---:|---:|
| 1 | 1 (`BATTERY`) | 4 (`Positive`) | 0.72270 |
| 1 | 2 (`CAMERA`) | 2 (`Negative`) | 0.71120 |
| 1 | 3 (`DESIGN`) | 4 (`Positive`) | 0.89490 |

Ràng buộc quan trọng:

```text
UNIQUE (review_id, aspect_id)
```

Mỗi đánh giá chỉ có tối đa một kết quả cho mỗi khía cạnh.

**Chỉ lưu các khía cạnh được mô hình dự đoán khác `None`.**  
Không lưu 10 dòng `None` cho mỗi review vì không cần thiết và làm phình dữ liệu.

---

## 4. Các quyết định thiết kế quan trọng

### 4.1. Charset

Toàn bộ database dùng:

```text
utf8mb4
utf8mb4_unicode_ci
```

Để lưu tiếng Việt và ký tự đặc biệt như `&` trong `SER&ACC`.

### 4.2. Không lưu đường dẫn checkpoint

Database chỉ lưu:

```text
model_name
model_version
```

Không lưu đường dẫn tuyệt đối của checkpoint để tránh lộ thông tin hệ thống nếu database bị xem trộm.

### 4.3. Không có trường thương hiệu

UIT-ViSFD không cung cấp định danh thương hiệu, nên schema không có:

```text
brand_id
brand_name
competitor
```

Không so sánh nhiều thương hiệu.

### 4.4. Không có cột “cụm ý kiến”

Checkpoint hiện tại là mô hình phân loại đa đầu ở mức câu, không có đầu ra BIO và không trích xuất cụm từ.

Vì vậy `analysis_results` **không có** cột `opinion_span`.  
Giao diện sẽ hiển thị:

```text
Cụm ý kiến: Không hỗ trợ bởi checkpoint hiện tại
```

Không tự sinh cụm ý kiến rồi gán cho PhoBERT.

### 4.5. Thời gian

- `review_date`: dùng cho biểu đồ và bộ lọc thời gian.
- `created_at`: thời điểm đưa dữ liệu vào hệ thống.
- `analyzed_at`: thời điểm phân tích xong.

Không dùng `created_at` thay cho `review_date` khi tính chỉ số theo thời gian, vì ngày nhập dữ liệu khác ngày đánh giá thực tế.

### 4.6. Trọng số BHI

`aspects.weight` ban đầu bằng `0.1` cho cả 10 khía cạnh.

Khi đủ dữ liệu cho cả 10 khía cạnh, công thức BHI giữ nguyên:

```text
BHI(t) = Σ w(a) × [NSS(a,t) + 100] / 2
```

Khi một khía cạnh không có mẫu, NSS không xác định.  
Giai đoạn 6 sẽ xử lý theo hướng:

- hiển thị `N/A` cho NSS của khía cạnh không có mẫu;
- tính BHI trên các khía cạnh có dữ liệu và chuẩn hóa lại trọng số;
- hiển thị số khía cạnh có dữ liệu, ví dụ `7/10`;
- hiển thị số mẫu để tránh diễn giải sai khi dữ liệu ít.

Đây là cách xử lý missing data, không thay đổi công thức khi có đủ 10 khía cạnh.

---

## 5. Import database bằng phpMyAdmin

### Bước 1: Khởi động XAMPP

Mở XAMPP Control Panel, bấm **Start** cho:

- Apache
- MySQL

### Bước 2: Mở phpMyAdmin

Truy cập:

```text
http://localhost/phpmyadmin
```

### Bước 3: Import file SQL

1. Chọn tab **Import**.
2. Chọn file:

```text
D:\Ct255-NVTM(BI)\nhom2\brand_health_demo\database\brand_health_db.sql
```

3. Chọn charset `utf-8`.
4. Bấm **Go / Import**.

File SQL tự tạo database `brand_health_db`, nên không cần tạo database trước.

### Bước 4: Kiểm tra

Chạy các lệnh:

```sql
USE brand_health_db;

SHOW TABLES;

SELECT COUNT(*) AS so_khia_canh FROM aspects;
SELECT COUNT(*) AS so_nhan_cam_xuc FROM sentiment_labels;
SELECT id, username, role, is_active FROM users;
```

Kết quả mong đợi:

```text
Số bảng: 5
so_khia_canh: 10
so_nhan_cam_xuc: 4
admin | admin | 1
```

---

## 6. Tài khoản đăng nhập mặc định

```text
Username: admin
Password: Admin@123
```

Mật khẩu đã được băm bằng bcrypt cost 12, không lưu plaintext.

**Hãy đổi mật khẩu ngay sau lần đăng nhập đầu tiên.**

Nếu cần tạo hash mới bằng PHP:

```php
<?php
echo password_hash('MatKhauMoi@123', PASSWORD_DEFAULT);
```

Sau đó cập nhật:

```sql
UPDATE users
SET password_hash = 'hash_mới'
WHERE username = 'admin';
```

---

## 7. Kiểm tra dữ liệu sau khi phân tích

### 7.1. Không có nhãn `None` bị lưu nhầm

```sql
SELECT COUNT(*) AS so_dong_none
FROM analysis_results ar
JOIN sentiment_labels sl ON sl.id = ar.sentiment_label_id
WHERE sl.is_mention = 0;
```

Kỳ vọng:

```text
0
```

### 7.2. Không có kết quả trùng cho cùng review + aspect

```sql
SELECT review_id, aspect_id, COUNT(*) AS so_lan
FROM analysis_results
GROUP BY review_id, aspect_id
HAVING COUNT(*) > 1;
```

Kỳ vọng: không có dòng nào.

### 7.3. Đếm số lượt đề cập theo khía cạnh

```sql
SELECT
    a.code AS aspect,
    COUNT(ar.id) AS so_luot_de_cap,
    SUM(sl.code = 'Positive') AS so_tich_cuc,
    SUM(sl.code = 'Negative') AS so_tieu_cuc,
    SUM(sl.code = 'Neutral')  AS so_trung_tinh
FROM aspects a
LEFT JOIN analysis_results ar ON ar.aspect_id = a.id
LEFT JOIN reviews r ON r.id = ar.review_id
    AND r.analysis_status = 'completed'
LEFT JOIN sentiment_labels sl ON sl.id = ar.sentiment_label_id
GROUP BY a.id, a.code, a.display_order
ORDER BY a.display_order;
```

### 7.4. Đếm theo khoảng thời gian

```sql
SET @date_from = '2026-01-01';
SET @date_to   = '2026-12-31';

SELECT
    a.code AS aspect,
    a.weight,
    COUNT(x.aspect_id) AS mentions,
    COALESCE(SUM(sl.code = 'Positive'), 0) AS positive_count,
    COALESCE(SUM(sl.code = 'Negative'), 0) AS negative_count,
    COALESCE(SUM(sl.code = 'Neutral'), 0)  AS neutral_count,
    ROUND(
        100 * COALESCE(SUM(sl.code = 'Positive'), 0)
        / NULLIF(COUNT(x.aspect_id), 0),
        2
    ) AS asr_percent
FROM aspects a
LEFT JOIN (
    SELECT ar.aspect_id, ar.sentiment_label_id
    FROM analysis_results ar
    JOIN reviews r ON r.id = ar.review_id
    WHERE r.analysis_status = 'completed'
      AND r.review_date BETWEEN @date_from AND @date_to
) x ON x.aspect_id = a.id
LEFT JOIN sentiment_labels sl ON sl.id = x.sentiment_label_id
GROUP BY a.id, a.code, a.weight, a.display_order
ORDER BY a.display_order;
```

---

## 8. Xóa và tạo lại database

Nếu muốn xóa sạch để làm lại demo:

```sql
DROP DATABASE brand_health_db;
```

Sau đó import lại file SQL.

Hoặc mở file SQL, bỏ comment 5 dòng:

```sql
-- DROP TABLE IF EXISTS analysis_results;
-- DROP TABLE IF EXISTS reviews;
-- DROP TABLE IF EXISTS aspects;
-- DROP TABLE IF EXISTS sentiment_labels;
-- DROP TABLE IF EXISTS users;
```

rồi import lại.

**Cảnh báo:** thao tác này xóa toàn bộ đánh giá và kết quả phân tích.

---

## 9. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `#1044 Access denied` | Đăng nhập MySQL không có quyền | Dùng user `root`, mật khẩu trống hoặc mật khẩu XAMPP hiện tại |
| `#1273 Unknown collation` | MariaDB quá cũ | Nâng XAMPP/MariaDB; file đã dùng `utf8mb4_unicode_ci` để tương thích |
| `#1064 Syntax error` gần `CHECK` | MySQL 5.7 quá cũ | Dùng MySQL 8+ hoặc MariaDB 10.4+ |
| `#1452 Cannot add or update a child row` | Dữ liệu tham chiếu sai `aspect_id` / `sentiment_label_id` | Chỉ dùng ID có trong bảng `aspects`, `sentiment_labels` |
| Tiếng Việt bị lỗi font | Charset kết nối sai | Kiểm tra PHP PDO dùng `charset=utf8mb4` |

---

## 10. Tiếp theo

Sau khi import và kiểm tra xong:

- `brand_health_db` có 5 bảng.
- `aspects` có 10 dòng.
- `sentiment_labels` có 4 dòng.
- `users` có tài khoản `admin`.

Giai đoạn 3 sẽ xây dựng Python API FastAPI, nạp đúng checkpoint `best_phobert_absa.pt`, tokenizer PhoBERT và trả JSON thật từ mô hình.