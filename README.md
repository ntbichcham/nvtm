# ABSA UIT-ViSFD - PhoBERT vs BiLSTM

Mã nguồn và artifact tái lập cho đề tài **Phân tích cảm xúc dựa trên khía cạnh
(Aspect-Based Sentiment Analysis) cho bình luận điện thoại tiếng Việt trên bộ
UIT-ViSFD**, sử dụng PhoBERT làm mô hình chính và BiLSTM đa đầu làm baseline.

Kho lưu trữ phục vụ Chương 4 của báo cáo: mọi con số trong báo cáo đều được
sinh lại từ cùng một split, cùng nhãn, cùng class weights và cùng hàm tính
độ đo.

## 1. Cấu trúc

```
ABSA_Project/
├── configs/
│   ├── preprocessing.yaml       # đường dẫn, nhãn, tokenizer, segmentation
│   └── teencode_rules.csv       # bảng teencode mẫu (mặc định tắt)
├── data/
│   ├── original/UIT-ViSFD/      # Train.csv, Dev.csv, Test.csv (không sửa)
│   └── processed/UIT-ViSFD/
│       ├── *_segmented.csv      # đầu ra VnCoreNLP dùng chung
│       ├── prepared_for_model/  # *_prepared.csv
│       └── reproduced/          # bản tái tạo P0
├── src/
│   ├── check_dataset.py         # kiểm định chất lượng dữ liệu
│   ├── preprocess.py            # tiền xử lý tái lập
│   ├── eda_visualization.py     # 9 hình EDA
│   ├── dataset_loader.py        # nhãn, dataset, class weights
│   ├── metrics_utils.py         # độ đo lõi dùng chung
│   ├── evaluate_absa.py         # đánh giá hợp nhất + xuất dự đoán
│   ├── train_phobert.py         # huấn luyện PhoBERT
│   └── train_baseline.py        # huấn luyện BiLSTM
├── tests/
│   └── test_metrics.py          # 9 test tính tay cho bộ độ đo
├── results/
│   ├── statistics/              # báo cáo TV1 (bản gốc)
│   ├── statistics/reproduced/   # báo cáo TV1 tái tạo
│   └── evaluation/              # metrics, per-aspect, predictions, errors
├── figures/                     # hình báo cáo (bản gốc)
│   └── reproduced/              # hình EDA tái tạo
└── models/                      # checkpoint (KHÔNG commit, ~1,6 GB)
```

## 2. Môi trường

Phiên bản đã kiểm chứng:

| Gói | Phiên bản |
|---|---|
| Python | 3.10.22 |
| torch | 2.13.0 (CUDA 13.0) |
| transformers | 4.57.3 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| PyYAML | 6.0.3 |

```bash
pip install -r requirements.txt
```

`py_vncorenlp` và Java là **tùy chọn**. Máy không có Java vẫn tái lập được vì
`preprocess.py` mặc định dùng lại các file `*_segmented.csv` đã phát hành.

## 3. Quy ước nhãn

10 khía cạnh, 4 lớp cảm xúc:

```
None=0, Negative=1, Neutral=2, Positive=3
```

Thứ tự khía cạnh và ánh xạ nhãn được import trực tiếp từ
`dataset_loader.py`, dùng chung cho cả huấn luyện và đánh giá.

## 4. Tái lập số liệu

Tất cả lệnh chạy từ thư mục gốc của repo:

```bash
export PYTHONPATH="$PWD/ABSA_Project/src"
export PYTHONIOENCODING="utf-8"
```

### 4.1. Kiểm định dữ liệu

```bash
python ABSA_Project/src/check_dataset.py \
  --out-dir ABSA_Project/results/statistics/reproduced
```

Sinh 19 CSV. Bản tái tạo trùng khớp tuyệt đối với 18 CSV gốc về số dòng,
thứ tự và giá trị (file `segmentation_report.csv` khác đường dẫn tuyệt đối).

### 4.2. Tiền xử lý

```bash
python ABSA_Project/src/preprocess.py \
  --out-dir ABSA_Project/data/processed/UIT-ViSFD/reproduced
```

Sinh `*_segmented.csv`, `prepared_for_model/*_prepared.csv`,
`phobert_token_{details,statistics}.csv`, `segmentation_report.csv`,
`handoff_validation_report.csv`, `handoff_additional_validation.csv`.

`token_length` là độ dài PhoBERT **có** special tokens (`<s>`, `</s>`).
Kết quả tái tạo trùng khớp tuyệt đối với 3 file prepared gốc.

### 4.3. Hình EDA

```bash
python ABSA_Project/src/eda_visualization.py \
  --stats-dir ABSA_Project/results/statistics/reproduced \
  --out-dir ABSA_Project/figures/reproduced
```

Sinh 9 hình: phân bố khía cạnh, cảm xúc (số lượng và tỷ lệ), độ dài ký tự,
đơn vị khoảng trắng, độ dài token PhoBERT, đồng xuất hiện (count và Jaccard),
phân bố bình luận theo tháng.

### 4.4. Kiểm thử độ đo

```bash
python ABSA_Project/tests/test_metrics.py
# hoặc: pytest ABSA_Project/tests/test_metrics.py -v
```

9/9 test PASS. Các giá trị kỳ vọng được tính tay từ vector nhãn 10 khía cạnh.

### 4.5. Huấn luyện

PhoBERT được huấn luyện với `max_len=160`, đánh giá và triển khai với
`max_len=128`:

```bash
python ABSA_Project/src/train_phobert.py --max_len 160
python ABSA_Project/src/train_baseline.py --max_len 160
```

Baseline dùng cùng `comment_segmented`, cùng split và cùng class weights.

### 4.6. Đánh giá hợp nhất

```bash
python ABSA_Project/src/evaluate_absa.py \
  --model_type phobert \
  --checkpoint ABSA_Project/models/best_phobert_absa.pt \
  --split Test --max_len 128 \
  --prefix phobert_absa_test

python ABSA_Project/src/evaluate_absa.py \
  --model_type bilstm \
  --checkpoint ABSA_Project/models/best_bilstm_bilstm_official.pt \
  --vocab ABSA_Project/models/vocab_bilstm_official.json \
  --split Test --max_len 128 \
  --prefix bilstm_absa_test
```

Mỗi lần chạy sinh:

- `*_metrics.json`: micro/macro/weighted P/R/F1, ACD macro/micro,
  sentiment macro/micro, pair F1, per-class, exact match, confusion matrix;
- `*_per_aspect.csv`: bảng 10 khía cạnh;
- `*_predictions.csv`: 2.224 dòng x 33 cột (10 `true_`, 10 `pred_`,
  10 `prob_` và 3 cột định danh);
- `*_predictions.jsonl`: 2.224 dòng, mỗi dòng là một mẫu;
- `*_errors.csv`: các ô dự đoán sai;
- `*_classification_report.txt`: báo cáo sklearn toàn cục và từng khía cạnh.

Thêm `--reference-json` để bật kiểm tra hồi quy:

```bash
--reference-json ABSA_Project/results/evaluation/phobert_quality_test.json
```

Kết quả hồi quy: `max|delta| = 0.000e+00`, `all_match = True` cho cả hai mô hình.

## 5. Kết quả trên tập Test

| Độ đo | PhoBERT | BiLSTM |
|---|---:|---:|
| Micro F1 (ô nhãn) | 0,906340 | 0,883543 |
| Macro F1 (ô nhãn) | 0,787910 | 0,753001 |
| Sentiment macro F1 | 0,732706 | 0,690904 |
| Sentiment micro F1 | 0,799736 | 0,753521 |
| ACD F1 | 0,894995 | 0,858406 |
| ACD macro F1 | 0,874173 | 0,810588 |
| Pair micro F1 | 0,969737 | 0,972730 |
| Pair macro F1 | 0,886182 | 0,883864 |
| Exact Match | 0,419964 | 0,339478 |

## 6. Quy ước chia cho 0

- Độ đo mức ô nhãn: theo `zero_division=0` của scikit-learn.
- ACD macro: bỏ qua khía cạnh không xác định; trả 1,0 nếu không có khía cạnh
  nào được gán hoặc dự đoán.
- Sentiment macro: trung bình trên 3 lớp Negative/Neutral/Positive.
- Pair macro: trung bình trên hợp các cặp (khía cạnh, cực tính) xuất hiện ở
  nhãn thật hoặc dự đoán.

Quy ước này được ghi rõ trong `evaluate_absa.py` và kiểm chứng bằng
`tests/test_metrics.py`.

## 7. Ghi chú tái lập

- Không sửa dữ liệu gốc; mọi đầu ra tái tạo nằm trong thư mục `reproduced/`.
- Checkpoint PhoBERT (~1,6 GB) không được đưa vào Git; xem hướng dẫn huấn luyện
  ở mục 4.5 để sinh lại.
- Trên Windows, đặt `PYTHONIOENCODING=utf-8` trước khi chạy để tránh lỗi
  hiển thị tiếng Việt.
