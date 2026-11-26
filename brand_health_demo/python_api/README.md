# Giai đoạn 3 — Python API PhoBERT ABSA

## 1. Mục tiêu

Python API có nhiệm vụ:

1. Nạp tokenizer PhoBERT.
2. Nạp đúng kiến trúc `MultiHeadPhoBERTForABSA`.
3. Nạp checkpoint thật `best_phobert_absa.pt`.
4. Tách từ bằng VnCoreNLP đúng pipeline huấn luyện.
5. Tokenize và chạy forward pass.
6. Trả JSON gồm khía cạnh, cảm xúc, độ tin cậy và xác suất 4 nhãn.
7. Xử lý lỗi checkpoint, model, segmenter, timeout.

API **không** tự tạo kết quả giả và **không** trả cụm ý kiến vì checkpoint hiện tại không có đầu ra BIO.

---

## 2. Cấu trúc file

```text
brand_health_demo/
├── run_api.ps1
└── python_api/
    ├── __init__.py
    ├── main.py
    ├── config.py
    ├── schemas.py
    ├── inference.py
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

| File | Chức năng |
|---|---|
| `main.py` | FastAPI app, endpoint `/health`, `/predict`, kiểm tra API key, timeout |
| `config.py` | Đọc biến môi trường và file `.env` |
| `schemas.py` | Pydantic request/response |
| `inference.py` | Nạp checkpoint, VnCoreNLP, tokenizer, chạy dự đoán |
| `requirements.txt` | Thư viện Python |
| `.env.example` | Mẫu cấu hình |
| `run_api.ps1` | Script chạy API trên Windows |

---

## 3. Môi trường đã cài

Trong giai đoạn này đã cài thêm vào môi trường hiện tại:

```text
D:\conda_envs\vicloabsa_gpu\python.exe
```

Các gói:

```text
fastapi
uvicorn[standard]
py_vncorenlp
pyjnius
python-dotenv
```

Không cài lại PyTorch/Transformers.

JDK 17 portable:

```text
D:\Ct255-NVTM(BI)\tools\jdk-17
```

VnCoreNLP:

```text
D:\Ct255-NVTM(BI)\nhom2\ABSA_Project\models\vncorenlp
├── VnCoreNLP-1.2.jar
└── models/
    └── wordsegmenter/
        ├── vi-vocab
        └── wordsegmenter.rdr
```

---

## 4. Tạo file cấu hình `.env`

Trong thư mục `brand_health_demo/python_api`, copy:

```text
.env.example  →  .env
```

Kiểm tra các đường dẫn trong `.env`:

```env
MODEL_CHECKPOINT_PATH=D:/Ct255-NVTM(BI)/nhom2/ABSA_Project/models/best_phobert_absa.pt
MODEL_CODE_DIR=D:/Ct255-NVTM(BI)/nhom2/ABSA_Project/src
TOKENIZER_NAME=vinai/phobert-base
MAX_LEN=128
DEVICE=auto
SEGMENTER_BACKEND=vncorenlp
VNCORENLP_DIR=D:/Ct255-NVTM(BI)/nhom2/ABSA_Project/models/vncorenlp
JAVA_HOME=D:/Ct255-NVTM(BI)/tools/jdk-17
API_KEY=
HF_LOCAL_FILES_ONLY=true
REQUEST_TIMEOUT_SECONDS=30
LOG_LEVEL=INFO
```

Lưu ý:

- Dùng dấu `/` trong đường dẫn Windows để tránh lỗi escaping.
- Không commit `.env` lên Git.
- `API_KEY` để trống khi chạy local; đặt chuỗi bí mật khi deploy.

Nếu chưa tạo `.env`, API vẫn dùng giá trị mặc định trong `config.py`. Tuy nhiên nên tạo `.env` để dễ chỉnh.

---

## 5. Chạy API

Mở PowerShell:

```powershell
cd "D:\Ct255-NVTM(BI)\nhom2\brand_health_demo"
.\run_api.bat
```

`run_api.bat` gọi PowerShell với `-ExecutionPolicy Bypass`, tránh lỗi
"running scripts is disabled on this system". Nếu muốn chạy trực tiếp:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_api.ps1
```

Hoặc chạy trực tiếp:

```powershell
$env:JAVA_HOME = "D:\Ct255-NVTM(BI)\tools\jdk-17"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:PYTHONIOENCODING = "utf-8"

cd "D:\Ct255-NVTM(BI)\nhom2\brand_health_demo"
& "D:\conda_envs\vicloabsa_gpu\python.exe" -m uvicorn python_api.main:app --host 127.0.0.1 --port 8000
```

Truy cập:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

Lần đầu nạp checkpoint có thể mất 30–90 giây (đã đo thực tế 87,6 giây trên máy này). API chỉ nhận request sau khi nạp xong.

---

## 6. Kiểm tra health

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
```

Kết quả mong đợi:

```json
{
  "status": "ok",
  "model_loaded": true,
  "segmenter_ready": true,
  "checkpoint_found": true,
  "device": "cuda",
  "max_len": 128,
  "segmenter_backend": "vncorenlp",
  "model_name": "PhoBERT-ABSA",
  "model_version": "phobert_official_ep5",
  "message": "API sẵn sàng."
}
```

Nếu `status = degraded`:

- `checkpoint_found = false`: sai đường dẫn checkpoint.
- `segmenter_ready = false`: Java hoặc VnCoreNLP lỗi.
- `model_loaded = false`: tokenizer/checkpoint/kiến trúc lỗi.

Xem log trong cửa sổ PowerShell để biết chi tiết.

---

## 7. Gọi API dự đoán

### PowerShell

```powershell
$body = @{
    text = "Pin dùng rất lâu nhưng camera chụp ảnh không được đẹp, thiết kế khá sang trọng."
    rating = 5
    include_all_aspects = $false
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
    | ConvertTo-Json -Depth 8
```

### curl

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@request.json"
```

Nội dung `request.json`:

```json
{
  "text": "Pin dùng rất lâu nhưng camera chụp ảnh không được đẹp, thiết kế khá sang trọng.",
  "rating": 5,
  "include_all_aspects": false
}
```

### Nếu có API key

Thêm header:

```text
X-API-Key: chuoi_bi_mat
```

---

## 8. Định dạng response

```json
{
  "success": true,
  "request_id": "4f6a...",
  "model": {
    "name": "PhoBERT-ABSA",
    "version": "phobert_official_ep5",
    "best_epoch": 5,
    "max_len": 128,
    "device": "cuda",
    "segmentation": "vncorenlp"
  },
  "text": "Pin dùng rất lâu nhưng camera chụp ảnh không được đẹp, thiết kế khá sang trọng.",
  "segmented_text": "Pin dùng rất lâu nhưng camera chụp ảnh không được đẹp , thiết_kế khá sang_trọng .",
  "aspects": [
    {
      "code": "BATTERY",
      "name_vi": "Pin",
      "model_index": 0,
      "sentiment": "Positive",
      "sentiment_vi": "Tích cực",
      "sentiment_index": 3,
      "confidence": 0.7,
      "confidence_percent": 70.0,
      "probabilities": {
        "None": 0.1,
        "Negative": 0.05,
        "Neutral": 0.15,
        "Positive": 0.7
      }
    }
  ],
  "all_aspects": null,
  "warnings": [],
  "segmentation_ms": 12.3,
  "inference_ms": 18.5,
  "total_ms": 31.2
}
```

`aspects` chỉ chứa các khía cạnh có nhãn khác `None`.

Nếu gửi `include_all_aspects: true`, trường `all_aspects` chứa đủ 10 khía cạnh.

---

## 9. Mã lỗi

| HTTP | Ý nghĩa |
|---:|---|
| `400` | Nội dung rỗng hoặc không hợp lệ |
| `401` | Sai/thiếu `X-API-Key` khi đã cấu hình API key |
| `422` | Request sai schema, ví dụ `rating` ngoài 1..5 |
| `503` | Model/segmenter chưa sẵn sàng |
| `504` | Suy luận quá thời gian `REQUEST_TIMEOUT_SECONDS` |
| `500` | Lỗi suy luận hoặc lỗi hệ thống |

Response lỗi mặc định của FastAPI:

```json
{
  "detail": "Mô hình chưa sẵn sàng. Kiểm tra log của Python API."
}
```

---

## 10. Kiểm tra bằng chế độ raw

Chỉ dùng khi chưa cài Java/VnCoreNLP hoặc để smoke test nhanh:

```powershell
$env:SEGMENTER_BACKEND = "raw"
.\run_api.ps1
```

Khi đó response có cảnh báo:

```text
SEGMENTER_BACKEND=raw: văn bản chưa được tách từ;
kết quả chỉ dùng để smoke test.
```

**Không dùng `raw` để lấy số liệu chính thức**, vì khác pipeline huấn luyện.

---

## 11. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Không tìm thấy checkpoint` | Kiểm tra `MODEL_CHECKPOINT_PATH` |
| `Không import được model_phobert.py` | Kiểm tra `MODEL_CODE_DIR` |
| `Không nạp được tokenizer PhoBERT` | Kiểm tra cache Hugging Face hoặc đặt `HF_LOCAL_FILES_ONLY=false` |
| `Không tìm thấy Java` | Kiểm tra `JAVA_HOME` trỏ tới thư mục có `bin\java.exe` |
| `Không nạp được VnCoreNLP` | Kiểm tra `VNCORENLP_DIR` có `VnCoreNLP-1.2.jar` và `models/wordsegmenter` |
| `CUDA out of memory` | Đặt `DEVICE=cpu` hoặc đóng ứng dụng khác dùng GPU |
| API chậm lần đầu | Lần đầu nạp checkpoint/VnCoreNLP; các request sau nhanh hơn |
| `status: degraded` | Xem log PowerShell; kiểm tra health từng trường |

---

## 12. Bảo mật

- API mặc định chỉ bind `127.0.0.1`, không mở ra mạng ngoài.
- Không trả đường dẫn checkpoint trong response.
- Log không ghi toàn bộ nội dung đánh giá.
- Có thể bật `API_KEY` để yêu cầu header `X-API-Key`.
- Không commit `.env`.
- Không đưa checkpoint vào `htdocs`.

---

## 13. Tiếp theo

Sau khi API chạy và `/health` trả `ok`:

1. Gọi thử `/predict` với câu mẫu.
2. Kiểm tra `aspects` có BATTERY/CAMERA/DESIGN.
3. Kiểm tra `segmented_text` có dấu `_` của VnCoreNLP.

Sau đó chuyển sang **Giai đoạn 4: xây dựng website PHP**.