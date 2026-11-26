# Báo cáo Giai đoạn 7 — Kiểm thử tổng thể, tài liệu hóa và bàn giao

> Bản Markdown tóm tắt của `BAO_CAO_GIAI_DOAN_7_WEBSITE_BHI.docx` / `.pdf`.
> Đây là tài liệu **bổ sung, độc lập**; báo cáo chính `baocao_NVTM_CT255.docx`
> không bị chỉnh sửa (SHA-256 `905B6B43EA607574DC283E7C518DBF5CDD69571679BE2AC3C425112E4D5ED192`).
>
> Hệ thống: ABSA + theo dõi sức khỏe thương hiệu bằng PhoBERT trên UIT-ViSFD.
> Ngày kiểm thử: 14/11/2026 · Nhóm 2 — CT255.

## 1. Tóm tắt

| Hạng mục | Kết quả |
|---|---|
| Kịch bản end-to-end | 16/16 đạt |
| Kịch bản dữ liệu biên | 4/4 hiển thị đúng thông báo |
| Lớp bảo mật | 8/8 đạt |
| Sao lưu / phục hồi | Khớp 100% số dòng |
| Lỗi nghiêm trọng | 0 |
| Thời gian bộ E2E | 2,6 giây |

Hai vấn đề gặp phải đều thuộc công cụ kiểm thử, không phải website:

1. Toán tử `>` của PowerShell 5.1 ghi file UTF-16 làm backup không phục hồi được →
   chuyển sang `mysqldump --result-file`.
2. Kịch bản đầu dùng GET để đăng xuất, trong khi thiết kế chỉ nhận POST + CSRF →
   sửa kịch bản, xác nhận hành vi đúng.

## 2. Môi trường

| Thành phần | Giá trị |
|---|---|
| Web server | Apache 2.4.58 (XAMPP `D:\xamp`) |
| PHP | 8.0.30 — pdo_mysql, curl, mbstring |
| CSDL | MariaDB 10.4.32, utf8mb4_unicode_ci |
| Python | 3.10.22 (conda `vicloabsa_gpu`) |
| PyTorch / Transformers | 2.13.0+cu130 / 4.57.3 |
| GPU | NVIDIA RTX 2050 (CUDA) |
| Checkpoint | `best_phobert_absa.pt` — epoch 5, dev macro F1 0,781371 |
| max_len | 128 |
| Tách từ | VnCoreNLP + JDK 17 |
| API | `http://127.0.0.1:8000` — `phobert_official_ep5` |
| Website | `http://localhost/brand_health_demo/` |

`/health` trả `status=ok`, `model_loaded=true`, `segmenter_ready=true`,
`checkpoint_found=true`, `device=cuda`, `max_len=128`, `segmenter_backend=vncorenlp`.

## 3. Kiến trúc và luồng dữ liệu

```
Trình duyệt
   │ HTTP + session + CSRF
   ▼
Apache 2.4 ──► PHP 8.0 (chỉ public/)
                  │ PDO prepared statements        │ cURL JSON
                  ▼                                 ▼
           MariaDB 10.4                     FastAPI 127.0.0.1:8000
           brand_health_db                        │ PyTorch + Transformers
                                                  ▼
                                   best_phobert_absa.pt + VnCoreNLP
```

1. Nhập đánh giá → lưu `pending`.
2. Trang chi tiết gọi `POST /predict`.
3. API tách từ VnCoreNLP → PhoBERT → khía cạnh + nhãn + độ tin cậy.
4. PHP validate phản hồi, chỉ lưu khía cạnh khác `None` vào `analysis_results`.
5. Xóa kết quả cũ + ghi mới + cập nhật trạng thái trong một transaction.
6. Dashboard tính điểm khía cạnh, BHI, độ phủ và vẽ biểu đồ.

## 4. Ma trận kiểm thử end-to-end

| Mã | Kịch bản | Kết quả thực tế |
|---|---|---|
| E2E-01 | `/index.php` khi chưa đăng nhập | 302 → `/login.php` |
| E2E-02 | `/review_view.php?id=5` khi chưa đăng nhập | 302 → `/login.php` |
| E2E-03 | Đăng nhập admin | 302 → `/index.php` |
| E2E-04 | POST tạo đánh giá thiếu CSRF | 403 |
| E2E-05 | Tạo đánh giá hợp lệ (4 sao, 14/11/2026) | 302 → `review_view.php?id=12` |
| E2E-06 | Phân tích #12 bằng PhoBERT | `completed`, 3 khía cạnh, `inference_ms=293` |
| E2E-07 | Dashboard sau phân tích | BHI 60,0 · độ phủ 5/10 |
| E2E-08 | Nội dung rỗng | “Nội dung đánh giá không được để trống.” |
| E2E-09 | Ngày 01/01/2027 | “Ngày đánh giá không được ở tương lai.” |
| E2E-10 | Số sao = 9 | “Số sao phải từ 1 đến 5.” |
| E2E-11 | Ngày 30/02/2026 | “Ngày đánh giá không hợp lệ.” |
| E2E-12 | Nội dung chứa `<script>alert(1)</script>` | Hiển thị `&lt;script&gt;`, không có script thô |
| E2E-13 | Tìm kiếm `' OR 1=1 --` | 200, không `SQLSTATE`, không `Fatal error` |
| E2E-14 | `review_view.php?id=999999` | 404 |
| E2E-15 | `/app/config.php`, `/app/` | 404 cả hai |
| E2E-16 | Xóa 2 đánh giá tạm + đăng xuất POST | 302; logout → `/login.php`; `/index.php` → 302 |

Script tái lập: `docs/phase7_e2e_test.ps1` (20 kiểm tra, tự dọn dữ liệu tạm).

## 5. Kết quả ABSA thật

Gọi trực tiếp `/predict` với câu
“Máy chạy rất êm, pin trâu dùng cả ngày, nhưng loa ngoài hơi nhỏ.”:

| Khía cạnh | Cảm xúc | Độ tin cậy |
|---|---|---|
| BATTERY | Positive | 0,959246 |
| FEATURES | Negative | 0,667796 |
| GENERAL | Positive | 0,91974 |

Thời gian: tách từ 3,05 ms · suy luận 255,44 ms · tổng 260,86 ms (mô hình đã nạp sẵn).

Kết quả đang lưu cho hai đánh giá thật:

| Đánh giá | Khía cạnh | Cảm xúc | Độ tin cậy |
|---|---|---|---|
| #5 | CAMERA | Negative | 0,93519 |
| #5 | PERFORMANCE | Positive | 0,78479 |
| #11 | BATTERY | Positive | 0,51996 |
| #11 | CAMERA | Negative | 0,90280 |

## 6. BHI và độ phủ

Công thức: Tích cực = 1,0; Trung tính = 0,5; Tiêu cực = 0;
điểm khía cạnh = trung bình theo lượt nhắc × 100;
BHI = trung bình có trọng số 0,10 trên các khía cạnh có dữ liệu (trọng số chuẩn hóa lại).

| Thời điểm | Đánh giá | Độ phủ | BHI | Thành phần |
|---|---|---|---|---|
| Trước kiểm thử | #5, #11 | 3/10 | 66,7 | CAMERA 0; PERFORMANCE 100; BATTERY 100 |
| Sau khi thêm #12 | #5, #11, #12 | 5/10 | 60,0 | thêm FEATURES 0; GENERAL 100 |
| Sau khi xóa #12 | #5, #11 | 3/10 | 66,7 | trở về ban đầu |

Kiểm chứng: `66,7 = (0 + 100 + 100) / 3`; `60,0 = (0 + 100 + 100 + 0 + 100) / 5`.
Khía cạnh không có dữ liệu hiển thị **N/A**, không tham gia công thức.

### Bộ lọc thời gian

| Kịch bản | Kết quả |
|---|---|
| Không lọc | BHI 66,7 · 3/10 · đủ 3 biểu đồ |
| `from=2026-10-01` (dữ liệu 3 tháng thử) | BHI 25,0 |
| `from=2026-11-15` | BHI N/A · 0/10 · ẩn biểu đồ |
| `from=2026-11-14&to=2026-11-01` | Cảnh báo khoảng ngày ngược |
| `from=abc` | Cảnh báo ngày không hợp lệ |
| Dữ liệu thử 09–11/2026 | 100 · 0 · 50; tổng 62,5 |

Dữ liệu 3 tháng chỉ dùng kiểm tra công thức và đã được xóa.

## 7. Bảo mật

| Lớp | Cách áp dụng | Bằng chứng |
|---|---|---|
| Phân quyền | `require_login()` | Chưa đăng nhập → 302 |
| CSRF | Token 64 ký tự, `hash_equals` | Thiếu token → 403 |
| SQL injection | PDO prepared statements | `' OR 1=1 --` → 200, không lỗi |
| XSS | Escape output | `<script>` → `&lt;script&gt;` |
| Mật khẩu | bcrypt cost 12 | Không lưu mật khẩu thô |
| File riêng tư | Chỉ `public/` trong Alias | `/app/config.php` → 404 |
| Thông báo lỗi | Thân thiện, chi tiết ghi log | Không lộ stack trace |
| Dữ liệu nhạy cảm | Checkpoint ngoài web root | Không lưu đường dẫn vào CSDL |

Triển khai thật cần thêm HTTPS, cookie `Secure`/`HttpOnly`/`SameSite`, đổi mật khẩu admin,
giới hạn truy cập API và firewall cho cổng 8000.

## 8. Sao lưu, phục hồi, chuyển máy

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

Bản backup ngày 14/11/2026: 10.877 byte, 190 dòng, có `CREATE DATABASE` + `USE`,
không chứa mật khẩu thô. Phục hồi thử vào `brand_health_restore_test`:

| CSDL | reviews | analysis_results | aspects | users |
|---|---|---|---|---|
| `brand_health_db` | 2 | 4 | 10 | 1 |
| `brand_health_restore_test` | 2 | 4 | 10 | 1 |

> Không dùng `>` của PowerShell 5.1: file sẽ là UTF-16 và `mysql` báo
> `ASCII '\0' appeared in the statement`.

Checklist chuyển máy (12 mục): mã nguồn, file `.sql`, checkpoint (~1,6 GB), tokenizer,
VnCoreNLP + JDK 17, môi trường Python, `python_api/.env`, Apache Alias, quyền ghi log,
extension PHP, đổi mật khẩu admin, HTTPS/cookie bảo mật.

## 9. Kịch bản demo 10 phút

| Thời gian | Thao tác | Kết quả mong đợi |
|---|---|---|
| 0:00–1:00 | Giới thiệu ABSA, UIT-ViSFD, BHI | 10 khía cạnh, 4 nhãn |
| 1:00–2:00 | Đăng nhập, mở dashboard | BHI 66,7 · 3/10 · 3 biểu đồ |
| 2:00–3:30 | Thêm đánh giá mới | Trạng thái chờ phân tích |
| 3:30–5:00 | Bấm Phân tích bằng PhoBERT | Khía cạnh + độ tin cậy |
| 5:00–7:00 | Quay lại Tổng quan | BHI và biểu đồ cập nhật |
| 7:00–8:30 | Chip 30 ngày, trang Khía cạnh | Lọc theo ngày; 7 dòng N/A |
| 8:30–10:00 | Kiến trúc, bảo mật, hạn chế | Dùng sơ đồ mục 3, bảng mục 7 |

Dự phòng khi API chưa nạp xong (30–90 giây): dùng sẵn #5 và #11 đã phân tích.

## 10. Ảnh chụp cần chèn

| Hình | Trang | Nội dung |
|---|---|---|
| 1 | `/login.php` | Form đăng nhập |
| 2 | `/reviews.php` | Danh sách + bộ lọc |
| 3 | `/review_view.php?id=5` | Kết quả ABSA |
| 4 | `/index.php` | BHI + 3 biểu đồ |
| 5 | `/aspects.php` | 10 khía cạnh, N/A |
| 6 | `/index.php?from=2026-11-01` | Bộ lọc thời gian |
| 7 | `127.0.0.1:8000/docs` | Swagger UI (tuỳ chọn) |

## 11. Hạn chế

- **Không có opinion span**: checkpoint chỉ trả khía cạnh + nhãn, không có BIO/span.
- **Miền hẹp**: UIT-ViSFD là đánh giá điện thoại, 10 khía cạnh cố định.
- **Không có định danh thương hiệu**: không so sánh đa thương hiệu.
- **Trọng số BHI cố định 0,10**: không học từ dữ liệu, không phụ thuộc số lượt nhắc.
- **Độ phủ thấp** khi ít đánh giá → BHI dễ dao động.
- **Một tài khoản quản trị**, chưa có audit log đầy đủ.
- **Hạ tầng phát triển**: API một tiến trình, khởi động lạnh 30–90 giây, batch tối đa 20.
- **Vận hành thủ công**: chưa lên lịch backup, phiên PHP lưu trên file.

## 12. Hướng phát triển

1. Checkpoint có đầu span để trích đoạn bằng chứng.
2. Metadata thương hiệu/sản phẩm để so sánh đa thương hiệu.
3. Trọng số BHI theo ngành hoặc học từ dữ liệu.
4. Hàng đợi phân tích nền, tách tiến trình nạp mô hình.
5. Tự động hóa backup và kiểm tra phục hồi.
6. Docker Compose + HTTPS + giới hạn API.
7. Xuất báo cáo PDF/Excel, so sánh BHI giữa các kỳ.

## 13. Kết luận

Chuỗi nghiệp vụ đã hoàn chỉnh và được kiểm chứng trên hệ thống thật: nhập đánh giá →
PhoBERT → lưu transaction → BHI → biểu đồ → lọc thời gian. 16 kịch bản E2E,
4 kịch bản biên, 8 lớp bảo mật đều đạt; backup phục hồi khớp 100%.

Trạng thái cuối: 02 đánh giá, 04 kết quả khía cạnh, BHI 66,7, độ phủ 3/10.

Tài liệu đầy đủ (12 trang, 13 bảng): `BAO_CAO_GIAI_DOAN_7_WEBSITE_BHI.docx` / `.pdf`.
