# So sánh với các nghiên cứu liên quan — UIT-ViSFD ABSA

Sinh lúc: 2026-11-14T20:48:10+07:00

Quy ước so sánh:

- `same splits (Train/Dev/Test), reproduced`: chạy bởi nhóm trên đúng split 7786/1112/2224.
- `reproduced`: kết quả nhóm đã chạy trước đây trên cùng split (PhoBERT, BiLSTM).
- `reported`: số lấy từ bài báo gốc (có thể khác giao thức chia dữ liệu).
- `not_directly_comparable`: benchmark/dữ liệu khác — chỉ để tham khảo, KHÔNG so sánh trực tiếp.

## 1. So sánh trực tiếp trên UIT-ViSFD (ACD F1 và ASC F1)

| Nghiên cứu | Mô hình | Họ mô hình | ACD F1 (%) | ASC F1 (%) | Loại so sánh |
|---|---|---|---|---|---|
| Nhóm 2 — mô hình đơn giản (báo cáo này) | linear_svc | linear | 84.53 | 66.41 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | ridge | linear | 83.94 | 65.97 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | logistic_regression | linear | 83.24 | 65.29 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | sgd_log | linear | 81.82 | 64.58 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | decision_tree | tree | 79.32 | 50.99 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | mlp | neural | 75.67 | 58.96 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | extra_trees | ensemble | 73.90 | 57.19 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | random_forest | ensemble | 72.22 | 54.72 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | complement_nb | linear | 64.77 | 45.72 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | multinomial_nb | linear | 46.48 | 24.37 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | stratified | baseline | 40.80 | 16.34 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | majority | baseline | 30.87 | 11.31 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT ES 20/3 | Transformer | 90.71 | 76.95 | reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT 10 epochs | Transformer | 90.50 | 76.15 | reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT 5 epochs | Transformer | 89.50 | 72.71 | reproduced |
| Nhóm 2 (tái lập) | Bi-LSTM + fastText | DL | 86.53 | 69.14 | reproduced |
| SA2SL (Phan et al.) | Bi-LSTM (original) | DL | 84.48 | 63.06 | reported |
| SA2SL (Phan et al.) | LSTM | DL | 80.27 | 52.13 | reported |
| SA2SL (Phan et al.) | CNN | DL | 69.70 | 27.16 | reported |
| SA2SL (Phan et al.) | Naive Bayes | ML | 64.65 | 37.56 | reported |
| SA2SL (Phan et al.) | Random Forest | ML | 47.83 | 20.17 | reported |
| SA2SL (Phan et al.) | SVM | ML | 42.63 | 19.69 | reported |
| Thin et al. (Aspect-Category) | SVM | ML | — | 64.84 | reported |
| Thin et al. (Aspect-Category) | CNN | DL | — | 65.65 | reported |
| Thin et al. (Aspect-Category) | BiLSTM-CNN | DL | — | 63.06 | reported |
| Thin et al. (Aspect-Category) | PhoBERT | Transformer | — | 74.44 | reported |

## 2. So sánh trên độ đo 4 lớp / micro F1 / exact match

Chỉ các dòng có cùng giao thức 10 head (4 lớp gồm None) mới so sánh trực tiếp.

| Nghiên cứu | Mô hình | Macro F1 4 lớp (%) | Micro F1 (%) | Accuracy (%) | Exact match (%) | Loại |
|---|---|---|---|---|---|---|
| Nhóm 2 — mô hình đơn giản (báo cáo này) | linear_svc | 73.25 | 88.04 | 88.04 | 31.61 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | ridge | 72.79 | 87.28 | 87.28 | 29.45 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | logistic_regression | 72.11 | 86.15 | 86.15 | 25.36 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | sgd_log | 71.67 | 86.99 | 86.99 | 28.52 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | decision_tree | 60.95 | 80.61 | 80.61 | 14.15 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | mlp | 67.01 | 84.77 | 84.77 | 23.34 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | extra_trees | 65.67 | 84.58 | 84.58 | 23.20 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | random_forest | 63.72 | 83.88 | 83.88 | 20.37 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | complement_nb | 56.36 | 80.63 | 80.63 | 14.97 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | multinomial_nb | 39.72 | 76.29 | 76.29 | 7.55 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | stratified | 30.78 | 57.76 | 57.76 | 0.28 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | majority | 29.13 | 70.50 | 70.50 | 1.62 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT ES 20/3 | 81.68 | 91.79 | 91.79 | 46.79 | reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT 10 epochs | 81.07 | 91.58 | 91.58 | 46.18 | reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT 5 epochs | 78.37 | 90.50 | 90.50 | 41.91 | reproduced |
| Nhóm 2 (tái lập) | Bi-LSTM + fastText | 75.37 | 88.24 | 88.24 | 33.32 | reproduced |

## 3. Các kết quả KHÔNG so sánh trực tiếp

| Nghiên cứu | Mô hình | Accuracy (%) | Weighted F1 (%) | Macro F1 (%) | Ghi chú |
|---|---|---|---|---|---|
| ViSoBERT (Nguyen et al.) | PhoBERT-base | 75.72 | 75.52 | 75.52 | Benchmark khác (dữ liệu social media), KHÔNG so sánh trực tiếp với UIT-ViSFD. |
| ViSoBERT (Nguyen et al.) | PhoBERT-large | 76.52 | 76.36 | 76.22 | Benchmark khác, KHÔNG so sánh trực tiếp. |
| ViSoBERT (Nguyen et al.) | ViSoBERT | 77.83 | 77.75 | 77.75 | Mô hình ViSoBERT; benchmark khác, KHÔNG so sánh trực tiếp. |

## 4. Nhận xét

- Mô hình đơn giản (TF-IDF + tuyến tính/cây) là baseline minh bạch, chạy nhanh,
  nhưng thấp hơn PhoBERT đã fine-tune vì không có ngữ cảnh tiền huấn luyện.
- Các mô hình tuyến tính học tốt phần ACD (khía cạnh nào được nhắc) hơn phần ASC
  (phân cực cảm xúc), đúng với đặc điểm dữ liệu mất cân bằng mạnh (None chiếm đa số).
- So sánh với SA2SL/Thin et al. chỉ nên đọc theo từng cột độ đo, không gộp
  các độ đo khác định nghĩa vào một con số duy nhất.
- ViSoBERT dùng benchmark social-media khác, không đưa vào kết luận so sánh.

## 5. Tệp dữ liệu

- `comparison_with_related_studies.csv`: bảng đầy đủ, dễ lọc.
- `data/literature_baselines.csv`: nguồn số liệu gốc đã kiểm chứng.
