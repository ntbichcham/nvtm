<div align="center">

# 📱 Vietnamese Aspect-Based Sentiment Analysis (UIT-ViSFD)

**Phân tích cảm xúc theo khía cạnh cho bình luận điện thoại tiếng Việt — PhoBERT, BiLSTM+fastText & 14 Mô hình Baseline**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/PhoBERT-vinai%2Fphobert--base-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/vinai/phobert-base)
[![Dataset](https://img.shields.io/badge/Dataset-UIT--ViSFD-green)](#1-tổng-quan-dữ-liệu)
[![Course](https://img.shields.io/badge/CT255-NVTM%20(BI)-purple)](#)

</div>

---

Kho lưu trữ mã nguồn, dữ liệu đã tiền xử lý, kết quả thực nghiệm (multi-seed) và ứng dụng web minh họa (**Brand Health Intelligence**) cho đề tài học phần **CT255 — Nghiệp vụ thông minh (Business Intelligence)**. Mọi số liệu báo cáo đều có khả năng tái lập trên cùng một bộ chia `Train / Dev / Test` (`7.786 / 1.112 / 2.224` bình luận), cùng quy ước nhãn và cùng hàm đánh giá thống nhất.

---

## 📁 1. Cấu trúc thư mục

```text
nhom2/
├── ABSA_Project/                        # Hệ thống lõi huấn luyện, đánh giá & thực nghiệm
│   ├── configs/                         # Cấu hình tiền xử lý & quy tắc teencode
│   │   ├── preprocessing.yaml
│   │   └── teencode_rules.csv
│   ├── data/
│   │   ├── original/UIT-ViSFD/          # Dữ liệu gốc (Train.csv, Dev.csv, Test.csv)
│   │   ├── processed/UIT-ViSFD/         # Dữ liệu tách từ VnCoreNLP & prepared_for_model/
│   │   └── literature_baselines.csv     # Bảng tổng hợp kết quả từ các nghiên cứu liên quan
│   ├── demo/                            # Ứng dụng Web Demo (PHP + Python FastAPI API)
│   ├── figures/                         # Biểu đồ EDA, sơ đồ kiến trúc & hình kết quả báo cáo
│   ├── models/                          # Trọng số mô hình (.pt) — bỏ qua trên Git (>1.5 GB/file)
│   ├── results/                         # Kết quả đánh giá, thống kê, multi-seed & báo cáo
│   │   ├── cli_experiments/             # Thực nghiệm 14 mô hình cơ sở (3-seed)
│   │   ├── cli_experiments_5seed/       # Thực nghiệm robustness (5-seed)
│   │   ├── experiments_2026/            # Kết quả PhoBERT & BiLSTM+fastText
│   │   ├── experiments_5seed/           # Đánh giá ổn định 5-seed cho Deep Learning
│   │   ├── dataset_histograms/          # Thống kê phân bố dữ liệu
│   │   ├── report_assets/               # Bảng so sánh & manifest hình ↔ code
│   │   └── statistics/                  # Báo cáo thống kê kiểm định dữ liệu
│   ├── src/                             # Mã nguồn huấn luyện, đánh giá & trực quan hóa
│   └── tests/                           # Unit tests kiểm chứng hàm tính độ đo
├── docs/                                # Tài liệu dự án được tổ chức gọn gàng
│   ├── papers/                          # Các bài báo tham khảo (PhoBERT, ViSoBERT, SA2SL...)
│   ├── references/                      # Bảng tổng hợp DOI & so sánh thực nghiệm
│   └── reports/                         # Báo cáo Word chính thức & báo cáo tiến độ
├── .gitignore                           # Loại trừ checkpoints (.pt), archives (.zip) & file dự đoán lớn
├── README.md
└── requirements.txt
```

---

## ⚙️ 2. Cài đặt môi trường

Môi trường đã kiểm chứng với **Python 3.10+**, **PyTorch 2.x**, **Transformers 4.57+**, **scikit-learn 1.7+**:

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `py_vncorenlp` và Java là **tùy chọn**. Máy không cài Java vẫn chạy và tái lập đầy đủ vì hệ thống tái sử dụng các file `*_segmented.csv` đã tách từ sẵn.
> Trên Windows, khuyến nghị thiết lập encoding UTF-8 trước khi chạy:
> ```powershell
> $env:PYTHONPATH="$PWD\ABSA_Project\src"
> $env:PYTHONIOENCODING="utf-8"
> ```

---

## 🏷️ 3. Quy ước nhãn (UIT-ViSFD)

Bài toán phân loại đồng thời **10 khía cạnh** (`BATTERY`, `CAMERA`, `DESIGN`, `FEATURES`, `GENERAL`, `PERFORMANCE`, `PRICE`, `SCREEN`, `SER&ACC`, `STORAGE`), mỗi khía cạnh gồm **4 lớp cảm xúc**:

| Nhãn | 0 | 1 | 2 | 3 |
| :--- | :---: | :---: | :---: | :---: |
| **Ý nghĩa** | `None` | `Negative` | `Neutral` | `Positive` |

Quy ước ánh xạ nhãn và hàm tính trọng số lớp được quản lý tập trung trong [`ABSA_Project/src/dataset_loader.py`](ABSA_Project/src/dataset_loader.py).

---

## 🚀 4. Hướng dẫn chạy & Tái lập thực nghiệm

Tất cả các lệnh bên dưới được chạy từ thư mục gốc của repository:

### 4.1. Kiểm định dữ liệu & Tiền xử lý
```bash
# Kiểm định chất lượng dữ liệu (sinh 19 file CSV thống kê)
python ABSA_Project/src/check_dataset.py --out-dir ABSA_Project/results/statistics/reproduced

# Tiền xử lý & tạo dữ liệu chuẩn hóa đầu vào mô hình
python ABSA_Project/src/preprocess.py --out-dir ABSA_Project/data/processed/UIT-ViSFD/reproduced

# Kiểm thử đơn vị cho bộ độ đo (9/9 test PASS)
python ABSA_Project/tests/test_metrics.py
```

### 4.2. Huấn luyện các mô hình Deep Learning
```bash
# Huấn luyện Multi-Head PhoBERT (max_len=160)
python ABSA_Project/src/train_phobert.py --max_len 160

# Huấn luyện BiLSTM đa đầu (Baseline)
python ABSA_Project/src/train_baseline.py --max_len 160

# Huấn luyện BiLSTM + fastText Embedding
python ABSA_Project/src/train_bilstm_fasttext.py
```

### 4.3. Đánh giá hợp nhất (Test split)
```bash
python ABSA_Project/src/evaluate_absa.py \
  --model_type phobert \
  --checkpoint ABSA_Project/models/best_phobert_absa.pt \
  --split Test --max_len 128 \
  --prefix phobert_absa_test
```

### 4.4. Chạy thực nghiệm 14 mô hình cơ sở qua CLI (multi-seed)
```bash
# Xem danh sách 14 mô hình cơ sở (Naive Bayes, SVM, Logistic Regression, MLP, kNN, Trees...)
python ABSA_Project/src/run_simple_experiments.py --list

# Chạy benchmark đa seed
python ABSA_Project/src/run_simple_experiments.py \
  --models default --seeds 42 123 2024 \
  --tag baselines --output-dir ABSA_Project/results/cli_experiments
```

---

## 📊 5. Kết quả thực nghiệm chính (tập Test)

| Độ đo đánh giá | PhoBERT (Multi-Head) | BiLSTM (Multi-Head) |
| :--- | :---: | :---: |
| **Micro F1 (ô nhãn)** | **0.9063** | 0.8835 |
| **Macro F1 (ô nhãn)** | **0.7879** | 0.7530 |
| **Sentiment Macro F1** | **0.7327** | 0.6909 |
| **Sentiment Micro F1** | **0.7997** | 0.7535 |
| **ACD F1** | **0.8950** | 0.8584 |
| **ACD Macro F1** | **0.8742** | 0.8106 |
| **Pair Micro F1** | 0.9697 | **0.9727** |
| **Pair Macro F1** | **0.8862** | 0.8839 |
| **Exact Match** | **0.4200** | 0.3395 |

> 📌 Kết quả tổng hợp 5-seed, phân tích dữ liệu và so sánh với các nghiên cứu trước (SA2SL, ViSoBERT, Thin et al.) nằm tại `ABSA_Project/results/report_assets/comparison_with_related_studies.md`.

---

## 🌐 6. Ứng dụng Web Demo (Brand Health Intelligence)

Hệ thống demo tích hợp mô hình PhoBERT ABSA vào dashboard theo dõi sức khỏe thương hiệu nằm trong thư mục [`ABSA_Project/demo/`](ABSA_Project/demo/):
- **Backend AI:** Python FastAPI (`ABSA_Project/demo/python_api/`)
- **Web Dashboard:** PHP / MySQL (`ABSA_Project/demo/public/`, `ABSA_Project/demo/app/`)
- Xem chi tiết hướng dẫn cài đặt tại [`ABSA_Project/demo/README.md`](ABSA_Project/demo/README.md).

---

## 📦 7. Lưu ý về Trọng số mô hình (Model Checkpoints)

Do giới hạn dung lượng tệp của GitHub (100 MB/tệp), các checkpoint PhoBERT (`~1.54 GB/tệp`) trong `ABSA_Project/models/` không được đẩy lên Git. Bạn có thể tự huấn luyện lại bằng lệnh tại mục **4.2** hoặc đặt tệp `best_phobert_absa.pt` đã huấn luyện vào `ABSA_Project/models/` trước khi chạy đánh giá hoặc web demo.
