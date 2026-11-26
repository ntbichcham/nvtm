# Thư mục sao lưu

Chứa file dump CSDL do `mysqldump --result-file` tạo ra:

```
brand_health_db_YYYYMMDD_HHMMSS.sql
```

- File `.sql` trong thư mục này **không được commit** (đã có trong `.gitignore`) vì chứa
  hash bcrypt của tài khoản và nội dung đánh giá.
- Lệnh sao lưu/phục hồi: xem mục 7 của `BAO_CAO_GIAI_DOAN_7_WEBSITE_BHI.docx`
  hoặc mục 12 trong `brand_health_demo/README.md`.
- Luôn dùng `--result-file` của `mysqldump`; không dùng toán tử `>` của PowerShell 5.1
  vì file sẽ bị ghi UTF-16 và không phục hồi được.