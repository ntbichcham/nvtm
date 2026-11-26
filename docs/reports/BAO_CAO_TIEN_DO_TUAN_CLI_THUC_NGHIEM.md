# BÁO CÁO TIẾN ĐỘ TUẦN — CLI THỰC NGHIỆM, ĐỘ ĐO TOÀN DIỆN VÀ SO SÁNH NGHIÊN CỨU LIÊN QUAN

**Thời điểm lập báo cáo:** 14/11/2026 20:50

## 1. Tóm tắt tuần

- Hoàn thiện **CLI thực nghiệm** cho 14 mô hình trong thư viện (12 mô hình chạy mặc định) trên đúng split UIT-ViSFD 7.786 / 1.112 / 2.224, giữ nguyên thứ tự 10 khía cạnh và 4 nhãn.
- Đã chạy **39 lượt** thực nghiệm thật; báo cáo chính dùng **12 mô hình** ở seed 42, mỗi lượt sinh `RESULTS.txt` + JSON/CSV + hình PNG.
- Bổ sung **bộ độ đo toàn diện**: accuracy, balanced accuracy, P/R/F1 micro/macro/weighted, Cohen's kappa (linear & quadratic), MCC, ROC-AUC OVR, PR-AUC/AP, log loss, Brier, ECE, top-2/top-3, confusion matrix, per-class, per-aspect, exact match, Hamming, subset accuracy, ACD, ASC.
- Thêm các độ đo quy đổi trung thực cho ABSA: **aspect-set IoU**, **CMC@k**, **mAP / mAP@k**, kèm định nghĩa trong từng `RESULTS.txt`.
- Sinh **11 histogram/heatmap dữ liệu** (PNG + CSV + `DATASET_ANALYSIS.txt`).
- Lập **bảng so sánh với SA2SL, Thin et al., ViSoBERT** và kết quả PhoBERT/BiLSTM của nhóm, có cột đánh dấu mức độ so sánh được.
- Lập **manifest hình/bảng ↔ code** cho toàn bộ artifact.
- Không tạo số liệu giả; các độ đo ẩn tin (PSNR/SSIM/BER/capacity) được ghi rõ **không áp dụng** cho bài toán ABSA.

## 2. Giao diện dòng lệnh (CLI)

Script chính: `ABSA_Project/src/run_simple_experiments.py`.

```bash
# Liệt kê mô hình
python ABSA_Project/src/run_simple_experiments.py --list

# Chạy đầy đủ 12 mô hình mặc định, 1 seed
python ABSA_Project/src/run_simple_experiments.py \
  --models default --seeds 42 --tag baselines_seed42 \
  --output-dir ABSA_Project/results/cli_experiments

# Chạy 3 seed cho nhóm mô hình nhanh (kiểm tra độ ổn định)
python ABSA_Project/src/run_simple_experiments.py \
  --models majority stratified multinomial_nb complement_nb \
          logistic_regression linear_svc ridge sgd_log decision_tree \
  --seeds 42 123 2024 --tag robustness_3seed

# Chỉ xem cấu hình, không huấn luyện
python ABSA_Project/src/run_simple_experiments.py --models mlp --dry-run
```

### 2.1. Tham số CLI quan trọng

| Tham số | Mặc định / giá trị đã dùng | Ý nghĩa |
|---|---|---|
| `--models` | `default` (nhận `all` / tên cụ thể) | Chọn mô hình; `default` = 12 mô hình chính. |
| `--seeds` | 42 | Một hoặc nhiều seed; truyền nhiều seed để tính mean ± std. |
| `--tag` | `simple_models` (báo cáo dùng `baselines_seed42`, `robustness_3seed`) | Nhóm thực nghiệm; quyết định thư mục và bảng tổng hợp. |
| `--data-dir` | …/prepared_for_model | Thư mục chứa Train/Dev/Test đã tiền xử lý. |
| `--text-col` | comment_segmented | Cột văn bản đã tách từ bằng VnCoreNLP. |
| `--label-col` | label | Cột nhãn dạng `{ASPECT#Polarity};…`. |
| `--max-features` | 30000 | Số đặc trưng TF-IDF word tối đa. |
| `--ngram-min/max` | 1 / 2 | Unigram + bigram. |
| `--char-ngram-max` | 0 (tắt) | Bật char n-gram để bắt lỗi chính tả teencode. |
| `--class-weight` | balanced | Cân bằng lớp; quan trọng vì None chiếm đa số. |
| `--C` | 1.0 | Hệ số điều chuẩn cho Logistic Regression / Linear SVC. |
| `--alpha` | 1.0 | Alpha cho Naive Bayes. |
| `--sgd-alpha` | 1e-4 | Alpha cho SGD logistic. |
| `--mlp-alpha` | 1e-4 | Alpha (L2) cho MLP. |
| `--epochs` | 20 (báo cáo dùng **8** cho MLP) | Số epoch cho `sgd_log` và `mlp` (có đường cong loss/acc). |
| `--hidden-layers` | 256 128 (báo cáo dùng **128** cho MLP) | Kiến trúc MLP. |
| `--batch-size` | 64 (báo cáo dùng **128** cho MLP) | Batch size MLP. |
| `--learning-rate` | 1e-3 | Learning rate MLP. |
| `--n-estimators` | 200 | Số cây cho Random Forest / Extra Trees. |
| `--knn-neighbors` | 15 | Số láng giềng cho kNN. |
| `--eval-train/--eval-dev` | bật | Đánh giá thêm Train/Dev ngoài Test. |
| `--save-predictions` | bật | Xuất dự đoán từng cặp đánh giá × khía cạnh. |
| `--save-probabilities` | bật | Xuất xác suất thô (N,10,4) dạng NPZ. |
| `--figures` | bật | Vẽ confusion matrix, ROC, PR, calibration, CMC, IoU… |
| `--force` | tắt | Ghi đè lượt chạy đã có. |
| `--dry-run` | tắt | Chỉ in cấu hình. |

## 3. Cấu trúc kết quả mỗi lần chạy

Thư mục: `results/cli_experiments/<tag>/<model>/seed_<n>/`

| Tệp | Nội dung |
|---|---|
| `RESULTS.txt` | Báo cáo văn bản: lệnh, môi trường, dữ liệu, siêu tham số, kiến trúc, độ đo Train/Dev/Test, confusion matrix, epoch, artifact, liên kết code. |
| `config.json` | Toàn bộ tham số dòng lệnh. |
| `environment.json` | Python, thư viện, CPU/GPU, git commit. |
| `data_summary.json` | Số dòng, phân bố nhãn, lượt nhắc khía cạnh, độ dài token. |
| `metrics_{train,dev,test}.json` | Bộ độ đo đầy đủ từng split. |
| `per_class_{split}.csv` | P/R/F1/support/AUC/AP từng lớp cảm xúc. |
| `per_aspect_{split}.csv` | Độ đo từng khía cạnh (10 dòng). |
| `aspect_detection_{split}.csv` | ACD, exact match, IoU, mAP, CMC@k. |
| `confusion_matrix_{split}.csv` | Ma trận 4×4. |
| `confusion_matrix_acd_{split}.csv` | Ma trận 2×2 None vs Mention. |
| `confusion_matrix_per_aspect_{split}.csv` | 10 ma trận 4×4 dạng long. |
| `predictions_{split}.csv` | Một dòng / (đánh giá × khía cạnh), kèm xác suất. |
| `probabilities_{split}.npz` | Mảng xác suất (N,10,4). |
| `train_log.csv` | Loss/accuracy/macro-F1 từng epoch (chỉ `sgd_log`, `mlp`). |
| `figures/*.png` | Confusion matrix, per-aspect F1, ROC, PR, calibration, CMC, IoU histogram, confidence histogram, learning curves. |
| `figure_manifest.csv` | Mỗi hình ↔ hàm/lệnh/dữ liệu/SHA-256. |

## 4. Dữ liệu và histogram

Bảng 1. Kích thước và phân bố nhãn (cặp đánh giá × khía cạnh).

| Split | Số đánh giá | None | Negative | Neutral | Positive |
|---|---|---|---|---|---|
| train | 7786 | 53988 | 7464 | 2903 | 13505 |
| dev | 1112 | 7804 | 1011 | 381 | 1924 |
| test | 2224 | 15518 | 2210 | 817 | 3695 |

Bảng 2. Danh mục histogram dữ liệu (mỗi hình có CSV số liệu gốc).

| Hình | Mô tả | PNG | CSV | Code |
|---|---|---|---|---|
| split_size_bar.png | Số đánh giá mỗi split | có | có | `analyze_dataset_histograms.py :: main()` |
| aspect_mention_counts_by_split.png | Số lượt nhắc từng khía cạnh theo split | có | có | `analyze_dataset_histograms.py :: main()` |
| aspect_mention_rate_heatmap.png | Tỉ lệ đánh giá nhắc từng khía cạnh | có | có | `analyze_dataset_histograms.py :: main()` |
| sentiment_distribution_by_split.png | Phân bố 4 nhãn cảm xúc | có | có | `analyze_dataset_histograms.py :: main()` |
| label_per_aspect_heatmap_train.png | Số lượng nhãn theo từng khía cạnh (Train) | có | có | `analyze_dataset_histograms.py :: main()` |
| sentiment_per_aspect_train.png | Tỉ lệ cảm xúc trong từng khía cạnh | có | có | `analyze_dataset_histograms.py :: main()` |
| aspect_count_per_review_hist.png | Số khía cạnh được nhắc trong một đánh giá | có | có | `analyze_dataset_histograms.py :: main()` |
| token_length_hist.png | Độ dài token | có | có | `analyze_dataset_histograms.py :: main()` |
| character_length_hist.png | Độ dài ký tự | có | có | `analyze_dataset_histograms.py :: main()` |
| aspect_cooccurrence_heatmap_train.png | Đồng xuất hiện khía cạnh (Jaccard) | có | có | `analyze_dataset_histograms.py :: main()` |
| monthly_volume.png | Số đánh giá theo tháng | có | có | `analyze_dataset_histograms.py :: main()` |

## 5. Kết quả thực nghiệm mô hình đơn giản

Bảng 3. Kết quả Test ở seed 42 (một dòng / mô hình). Đơn vị: %. Micro F1 = accuracy do mỗi cặp có đúng một nhãn.

| Mô hình | Loại | Runs | Acc | Macro F1 | Micro F1 | ACD F1 | ASC F1 | Exact | Kappa | MCC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear_svc | linear | 1 | 88.04 | 73.25 | 88.04 | 84.53 | 66.41 | 31.61 | 73.79 | 73.95 | 94.85 |
| ridge | linear | 1 | 87.28 | 72.79 | 87.28 | 83.94 | 65.97 | 29.45 | 72.75 | 72.78 | 94.47 |
| logistic_regression | linear | 1 | 86.15 | 72.11 | 86.15 | 83.24 | 65.29 | 25.36 | 71.10 | 71.12 | 94.48 |
| sgd_log | linear | 1 | 86.99 | 71.69 | 86.99 | 81.82 | 64.61 | 28.51 | 70.94 | 71.29 | 94.70 |
| mlp | neural | 1 | 84.77 | 67.01 | 84.77 | 75.67 | 58.96 | 23.34 | 64.68 | 65.59 | 92.35 |
| extra_trees | ensemble | 1 | 84.58 | 65.67 | 84.58 | 73.90 | 57.19 | 23.20 | 63.04 | 64.77 | 94.00 |
| random_forest | ensemble | 1 | 83.88 | 63.72 | 83.88 | 72.22 | 54.72 | 20.37 | 60.92 | 62.96 | 93.65 |
| decision_tree | tree | 1 | 80.56 | 61.13 | 80.56 | 79.13 | 51.27 | 14.03 | 59.58 | 59.60 | 76.61 |
| complement_nb | linear | 1 | 80.63 | 56.36 | 80.63 | 64.77 | 45.72 | 14.97 | 52.84 | 54.70 | 85.35 |
| multinomial_nb | linear | 1 | 76.29 | 39.72 | 76.29 | 46.48 | 24.37 | 7.55 | 35.12 | 40.69 | 83.10 |
| stratified | baseline | 1 | 57.75 | 30.90 | 57.75 | 40.83 | 16.53 | 0.18 | 11.45 | 11.45 | 73.77 |
| majority | baseline | 1 | 70.50 | 29.13 | 70.50 | 30.87 | 11.31 | 1.62 | 16.99 | 20.67 | 73.77 |

![Hình: so sánh 12 mô hình đơn giản trên 6 độ đo Test (seed 42).](ABSA_Project/results/report_assets/model_comparison_grouped.png)

*Hình: so sánh 12 mô hình đơn giản trên 6 độ đo Test (seed 42).* Code: `plot_model_comparison()` — `build_experiment_report.py`.

![Hình: xếp hạng mô hình đơn giản theo Macro F1 trên Test.](ABSA_Project/results/report_assets/model_comparison_macro_f1.png)

*Hình: xếp hạng mô hình đơn giản theo Macro F1 trên Test.* Code: `plot_model_comparison()` — `build_experiment_report.py`.

Bảng 4. Kiểm tra độ ổn định 3 seed (mean ± std, đơn vị %).

| Mô hình | Runs | Macro F1 | ACD F1 | ASC F1 | Exact | MCC |
|---|---|---|---|---|---|---|
| linear_svc | 3 | 73.25 ± 0.00 | 84.53 ± 0.00 | 66.41 ± 0.00 | 31.61 ± 0.00 | 73.95 ± 0.00 |
| ridge | 3 | 72.79 ± 0.00 | 83.94 ± 0.00 | 65.97 ± 0.00 | 29.45 ± 0.00 | 72.78 ± 0.00 |
| logistic_regression | 3 | 72.11 ± 0.00 | 83.24 ± 0.00 | 65.29 ± 0.00 | 25.36 ± 0.00 | 71.12 ± 0.00 |
| sgd_log | 3 | 71.67 ± 0.02 | 81.82 ± 0.01 | 64.58 ± 0.02 | 28.52 ± 0.09 | 71.29 ± 0.01 |
| decision_tree | 3 | 60.95 ± 0.20 | 79.32 ± 0.14 | 50.99 ± 0.27 | 14.15 ± 0.09 | 59.69 ± 0.19 |
| complement_nb | 3 | 56.36 ± 0.00 | 64.77 ± 0.00 | 45.72 ± 0.00 | 14.97 ± 0.00 | 54.70 ± 0.00 |
| multinomial_nb | 3 | 39.72 ± 0.00 | 46.48 ± 0.00 | 24.37 ± 0.00 | 7.55 ± 0.00 | 40.69 ± 0.00 |
| stratified | 3 | 30.78 ± 0.16 | 40.80 ± 0.04 | 16.34 ± 0.24 | 0.28 ± 0.08 | 11.28 ± 0.13 |
| majority | 3 | 29.13 ± 0.00 | 30.87 ± 0.00 | 11.31 ± 0.00 | 1.62 ± 0.00 | 20.67 ± 0.00 |

Ghi chú: các mô hình ensemble/MLP nặng chỉ chạy 1 seed; nhóm tuyến tính và baseline chạy 3 seed. Cột `Runs` cho biết số lượt thực tế.

`± 0.00` không phải là thiếu đo: các mô hình tất định (LinearSVC, Ridge, Logistic Regression, Naive Bayes, baseline majority) không dùng lấy mẫu ngẫu nhiên sau khi đã cố định TF-IDF, nên 3 seed cho ra cùng một kết quả. Chỉ `sgd_log`, `decision_tree` và `stratified` có phương sai khác 0.

### 5.1. Mô hình đơn giản tốt nhất (theo Macro F1 Test): `linear_svc` (seed 42)

| Độ đo | Giá trị |
|---|---|
| Accuracy | 88.04 |
| Balanced accuracy | 70.12 |
| Macro F1 (4 lớp) | 73.25 |
| Micro F1 | 88.04 |
| Weighted F1 | 87.55 |
| Cohen's kappa | 73.79 |
| Cohen's kappa (quadratic) | 79.80 |
| MCC | 73.95 |
| ROC-AUC OVR (macro) | 94.85 |
| PR-AUC (macro) | 77.82 |
| Log loss | 58.31 |
| Brier score | 27.29 |
| ECE (10 bins) | 25.63 |
| Top-2 accuracy | 95.98 |
| Top-3 accuracy | 98.63 |
| ASC macro F1 | 66.41 |
| ACD F1 (từ detection) | 84.53 |
| Exact match | 31.61 |
| Hamming loss (4 lớp) | 11.96 |
| Aspect-set IoU (mean) | 73.79 |
| mAP aspect detection (macro) | 88.91 |
| CMC Recall@5 | 96.28 |

Confusion matrix 4×4 trên Test (hàng = thật, cột = dự đoán):

| Thật \ Dự đoán | None | Negative | Neutral | Positive |
|---|---|---|---|---|
| None | 14850 | 285 | 51 | 332 |
| Negative | 548 | 1482 | 46 | 134 |
| Neutral | 172 | 133 | 313 | 199 |
| Positive | 592 | 101 | 67 | 2935 |

## 6. Hình tiêu biểu của mô hình tốt nhất

![Hình 1. Confusion matrix 4×4 trên Test.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/confusion_matrix_test.png)

*Hình 1. Confusion matrix 4×4 trên Test.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/confusion_matrix_test.png` — xem `figure_manifest.csv`.

![Hình 2. Macro F1 từng khía cạnh trên Test.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/per_aspect_macro_f1_test.png)

*Hình 2. Macro F1 từng khía cạnh trên Test.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/per_aspect_macro_f1_test.png` — xem `figure_manifest.csv`.

![Hình 3. F1 từng lớp cảm xúc trên Test.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/per_class_f1_test.png)

*Hình 3. F1 từng lớp cảm xúc trên Test.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/per_class_f1_test.png` — xem `figure_manifest.csv`.

![Hình 4. ROC one-vs-rest trên Test.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/roc_curves_test.png)

*Hình 4. ROC one-vs-rest trên Test.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/roc_curves_test.png` — xem `figure_manifest.csv`.

![Hình 5. Precision-Recall trên Test.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/pr_curves_test.png)

*Hình 5. Precision-Recall trên Test.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/pr_curves_test.png` — xem `figure_manifest.csv`.

![Hình 6. Reliability diagram (calibration).](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/calibration_test.png)

*Hình 6. Reliability diagram (calibration).* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/calibration_test.png` — xem `figure_manifest.csv`.

![Hình 7. Histogram độ tin cậy đúng/sai.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/confidence_hist_test.png)

*Hình 7. Histogram độ tin cậy đúng/sai.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/confidence_hist_test.png` — xem `figure_manifest.csv`.

![Hình 8. CMC@k cho xếp hạng khía cạnh.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/cmc_test.png)

*Hình 8. CMC@k cho xếp hạng khía cạnh.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/cmc_test.png` — xem `figure_manifest.csv`.

![Hình 9. Histogram aspect-set IoU.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/aspect_set_iou_hist_test.png)

*Hình 9. Histogram aspect-set IoU.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/aspect_set_iou_hist_test.png` — xem `figure_manifest.csv`.

![Hình 10. Average precision phát hiện từng khía cạnh.](ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/aspect_ap_test.png)

*Hình 10. Average precision phát hiện từng khía cạnh.* Dữ liệu & code: `ABSA_Project/results/cli_experiments/baselines_seed42/linear_svc/seed_42/figures/aspect_ap_test.png` — xem `figure_manifest.csv`.

- *Hình 11. Loss/accuracy/macro-F1 theo epoch (nếu có).* — không áp dụng cho mô hình này (không có hình).

### 6.1. Đường cong loss / accuracy / macro-F1 theo epoch

![Learning curves sgd_log](ABSA_Project/results/cli_experiments/baselines_seed42/sgd_log/seed_42/figures/learning_curves.png)

*Learning curves của `sgd_log` (Train vs Dev).* Dữ liệu: `seed_42/train_log.csv`; code: `plot_learning_curves()`.

![Learning curves mlp](ABSA_Project/results/cli_experiments/baselines_seed42/mlp/seed_42/figures/learning_curves.png)

*Learning curves của `mlp` (Train vs Dev).* Dữ liệu: `seed_42/train_log.csv`; code: `plot_learning_curves()`.


## 7. So sánh với các nghiên cứu liên quan

Bảng 5. Đối chiếu trên UIT-ViSFD. Dấu — nghĩa là bài báo không báo cáo độ đo đó.

| Nghiên cứu | Mô hình | Họ | ACD F1 | ASC F1 | Macro F1 4 lớp | Micro F1 | Mức so sánh |
|---|---|---|---|---|---|---|---|
| Nhóm 2 — mô hình đơn giản (báo cáo này) | linear_svc | linear | 84.53 | 66.41 | 73.25 | 88.04 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | ridge | linear | 83.94 | 65.97 | 72.79 | 87.28 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | logistic_regression | linear | 83.24 | 65.29 | 72.11 | 86.15 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | sgd_log | linear | 81.82 | 64.58 | 71.67 | 86.99 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | decision_tree | tree | 79.32 | 50.99 | 60.95 | 80.61 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | mlp | neural | 75.67 | 58.96 | 67.01 | 84.77 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | extra_trees | ensemble | 73.90 | 57.19 | 65.67 | 84.58 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | random_forest | ensemble | 72.22 | 54.72 | 63.72 | 83.88 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | complement_nb | linear | 64.77 | 45.72 | 56.36 | 80.63 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | multinomial_nb | linear | 46.48 | 24.37 | 39.72 | 76.29 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | stratified | baseline | 40.80 | 16.34 | 30.78 | 57.76 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 — mô hình đơn giản (báo cáo này) | majority | baseline | 30.87 | 11.31 | 29.13 | 70.50 | same splits (Train/Dev/Test), reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT ES 20/3 | Transformer | 90.71 | 76.95 | 81.68 | 91.79 | reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT 10 epochs | Transformer | 90.50 | 76.15 | 81.07 | 91.58 | reproduced |
| Nhóm 2 (tái lập) | Multi-Head PhoBERT 5 epochs | Transformer | 89.50 | 72.71 | 78.37 | 90.50 | reproduced |
| Nhóm 2 (tái lập) | Bi-LSTM + fastText | DL | 86.53 | 69.14 | 75.37 | 88.24 | reproduced |
| SA2SL (Phan et al.) | Bi-LSTM (original) | DL | 84.48 | 63.06 | — | — | reported |
| SA2SL (Phan et al.) | LSTM | DL | 80.27 | 52.13 | — | — | reported |
| SA2SL (Phan et al.) | CNN | DL | 69.70 | 27.16 | — | — | reported |
| SA2SL (Phan et al.) | Naive Bayes | ML | 64.65 | 37.56 | — | — | reported |
| SA2SL (Phan et al.) | Random Forest | ML | 47.83 | 20.17 | — | — | reported |
| SA2SL (Phan et al.) | SVM | ML | 42.63 | 19.69 | — | — | reported |
| Thin et al. (Aspect-Category) | SVM | ML | — | 64.84 | — | — | reported |
| Thin et al. (Aspect-Category) | CNN | DL | — | 65.65 | — | — | reported |
| Thin et al. (Aspect-Category) | BiLSTM-CNN | DL | — | 63.06 | — | — | reported |
| Thin et al. (Aspect-Category) | PhoBERT | Transformer | — | 74.44 | — | — | reported |

Với các mô hình của nhóm: nếu có nhiều seed thì lấy trung bình 3 seed, nếu chỉ chạy 1 seed thì lấy seed 42. Vì vậy một vài số có thể lệch rất nhỏ so với Bảng 3 (seed 42 đơn lẻ).

Các dòng sau **không so sánh trực tiếp** vì dùng benchmark social-media khác:

| Nghiên cứu | Mô hình | Accuracy | Weighted F1 | Macro F1 | Ghi chú |
|---|---|---|---|---|---|
| ViSoBERT (Nguyen et al.) | PhoBERT-base | 75.72 | 75.52 | 75.52 | Benchmark khác (dữ liệu social media), KHÔNG so sánh trực tiếp với UIT-ViSFD. |
| ViSoBERT (Nguyen et al.) | PhoBERT-large | 76.52 | 76.36 | 76.22 | Benchmark khác, KHÔNG so sánh trực tiếp. |
| ViSoBERT (Nguyen et al.) | ViSoBERT | 77.83 | 77.75 | 77.75 | Mô hình ViSoBERT; benchmark khác, KHÔNG so sánh trực tiếp. |

Diễn giải: mô hình đơn giản là baseline minh bạch và nhanh, nhưng thấp hơn PhoBERT fine-tune vì thiếu ngữ cảnh tiền huấn luyện. Khoảng cách lớn nhất ở ASC (phân cực cảm xúc) và exact match, đúng với dữ liệu mất cân bằng mạnh.

## 8. Độ đo nào áp dụng cho ABSA?

| Độ đo | Nhóm | Áp dụng | Nơi báo cáo |
|---|---|---|---|
| Accuracy | classification | yes | metrics_{split}.json → global.accuracy |
| Balanced accuracy | classification | yes | global.balanced_accuracy |
| Precision / Recall / F1 (micro) | classification | yes | global.micro_* |
| Precision / Recall / F1 (macro) | classification | yes | global.macro_* |
| Precision / Recall / F1 (weighted) | classification | yes | global.weighted_* |
| Cohen's kappa (linear) | agreement | yes | global.cohen_kappa |
| Cohen's kappa (quadratic) | agreement | yes | global.cohen_kappa_quadratic |
| Matthews correlation coefficient | classification | yes | global.mcc |
| ROC-AUC OVR (macro/micro/weighted) | ranking | yes | global.roc_auc_ovr_* |
| PR-AUC / Average Precision | ranking | yes | global.pr_auc_* + per_class.average_precision |
| Log loss | probability | yes | global.log_loss |
| Brier score (multi-class) | probability | yes | global.brier_score_multiclass |
| Expected Calibration Error (ECE) | probability | yes | global.ece_10bin |
| Top-2 / Top-3 accuracy | ranking | yes | global.top{k}_accuracy |
| Confusion matrix 4×4 | diagnostic | yes | confusion_matrix_{split}.csv |
| Confusion matrix ACD (None vs Mention) | diagnostic | yes | confusion_matrix_acd_{split}.csv |
| Per-class P/R/F1 + support | diagnostic | yes | per_class_{split}.csv |
| Per-aspect P/R/F1 + kappa + MCC | diagnostic | yes | per_aspect_{split}.csv |
| Exact match | multi-output | yes | aspect_detection_{split}.csv → exact_match |
| Hamming loss | multi-output | yes | aspect_detection_{split}.csv |
| Subset accuracy (aspect set) | multi-output | yes | aspect_detection_{split}.csv |
| ACD precision/recall/F1 | ABSA | yes | aspect_detection_{split}.csv → acd_* |
| ASC macro P/R/F1 | ABSA | yes | global.sentiment_macro_* |
| Aspect-set precision/recall/F1 | multi-output | adapted | aspect_detection_{split}.csv |
| IoU / Jaccard | adapted | adapted | aspect_detection_{split}.csv → aspect_set_iou_mean |
| CMC@k | adapted | adapted | aspect_detection_{split}.csv → cmc_* |
| mAP (aspect detection) | adapted | adapted | aspect_detection_{split}.csv → aspect_map_* |
| mAP@k | adapted | adapted | aspect_detection_{split}.csv → aspect_map@k |
| Loss / accuracy theo epoch | training | conditional | train_log.csv + figures/learning_curves.png |
| Histogram dữ liệu | descriptive | yes | results/dataset_histograms/ |
| Histogram độ tin cậy | diagnostic | yes | figures/confidence_hist_test.png |
| PSNR | steganography | no | Không áp dụng — không tạo số liệu. |
| SSIM | steganography | no | Không áp dụng. |
| BER | steganography | no | Không áp dụng. |
| Capacity | steganography | no | Không áp dụng. |
| Robustness (JPEG/crop/noise) | steganography | no | Không áp dụng. |

Các độ đo ẩn tin **PSNR / SSIM / BER / capacity / robustness** không áp dụng vì bài toán không có ảnh bìa và không nhúng payload. Báo cáo không tạo số liệu giả.

## 9. Liên kết hình/bảng ↔ code

Manifest đầy đủ có **1782 mục** tại `ABSA_Project/results/report_assets/figure_table_code_manifest.csv` (có SHA-256).

| Artifact | Loại | Hàm tạo | Nhóm |
|---|---|---|---|
| model_comparison_grouped.png | figure | plot_model_comparison | report_assets |
| model_comparison_macro_f1.png | figure | plot_model_comparison | report_assets |
| split_size_bar.png | figure | main | dataset_histograms |
| aspect_mention_counts_by_split.png | figure | main | dataset_histograms |
| aspect_mention_rate_heatmap.png | figure | main | dataset_histograms |
| sentiment_distribution_by_split.png | figure | main | dataset_histograms |
| label_per_aspect_heatmap_train.png | figure | main | dataset_histograms |
| sentiment_per_aspect_train.png | figure | main | dataset_histograms |
| aspect_count_per_review_hist.png | figure | main | dataset_histograms |
| token_length_hist.png | figure | main | dataset_histograms |
| character_length_hist.png | figure | main | dataset_histograms |
| aspect_cooccurrence_heatmap_train.png | figure | main | dataset_histograms |
| monthly_volume.png | figure | main | dataset_histograms |
| confusion_matrix_test.png | figure | plot_confusion | baselines_seed42/complement_nb/seed_42 |
| per_aspect_macro_f1_test.png | figure | plot_bar | baselines_seed42/complement_nb/seed_42 |
| per_class_f1_test.png | figure | plot_bar | baselines_seed42/complement_nb/seed_42 |
| roc_curves_test.png | figure | plot_roc | baselines_seed42/complement_nb/seed_42 |
| pr_curves_test.png | figure | plot_pr | baselines_seed42/complement_nb/seed_42 |
| calibration_test.png | figure | plot_calibration | baselines_seed42/complement_nb/seed_42 |
| confidence_hist_test.png | figure | plot_confidence_hist | baselines_seed42/complement_nb/seed_42 |

… (xem CSV để có toàn bộ danh mục).


## 10. Kiểm tra yêu cầu

| Yêu cầu | Trạng thái | Bằng chứng |
|---|---|---|
| Chỉ dùng công nghệ cho phép (HTML/CSS/Bootstrap/JS/Chart.js + PHP/PDO/MySQL; Python cho model) | Đạt | Website Phase 1–7 đã kiểm thử; không dùng React/Node/Laravel. |
| Dùng checkpoint thật `best_phobert_absa.pt`, không bịa kết quả | Đạt | FastAPI `/health` báo `model_loaded=true`, `device=cuda`, `max_len=128`. |
| Giữ 10 khía cạnh, thứ tự nhãn None/Negative/Neutral/Positive | Đạt | Import từ `dataset_loader.py`; ghi trong mọi `RESULTS.txt`. |
| Không lộ đường dẫn checkpoint trong UI/DB | Đạt | Phase 7 audit. |
| Prepared statements, `password_hash`/`password_verify`, session, CSRF, validation | Đạt | Phase 4–7 E2E 20/20. |
| CLI có tham số dòng lệnh | Đạt | Mục 2; `--help`, `--list`, `--dry-run`. |
| Đề xuất tham số CLI quan trọng | Đạt | Bảng tham số mục 2.1. |
| Lưu kết quả ra txt (kết quả, siêu tham số, dữ liệu, kiến trúc, lệnh) | Đạt | `RESULTS.txt` mục 1–13. |
| Đánh giá trên Train và Test | Đạt | `metrics_train.json`, `metrics_test.json`, cả Dev. |
| Độ đo tối đa phù hợp: acc, F1, kappa, MCC, AUC, confusion matrix, loss/acc/epoch | Đạt | Mục 5; bảng độ đo mục 8. |
| CMC, IoU, mAP và độ đo liên quan | Đạt (quy đổi trung thực) | Aspect-set IoU, CMC@k, mAP/mAP@k; định nghĩa trong `RESULTS.txt` mục 12. |
| Biểu đồ histogram | Đạt | 11 histogram/heatmap + histogram độ tin cậy/IoU. |
| So sánh với nghiên cứu liên quan | Đạt | Mục 7; `comparison_with_related_studies.csv`. |
| Liên kết code cho từng hình/bảng | Đạt | Mục 9; `figure_table_code_manifest.csv`. |
| Báo cáo tuần ở tệp riêng, không sửa báo cáo Word cũ | Đạt | Tệp này; `baocao_NVTM_CT255.docx` giữ nguyên SHA-256. |
| Độ đo ẩn tin (PSNR/SSIM/BER/capacity) | Không áp dụng | Ghi rõ lý do; không tạo số liệu. |

## 11. Hạn chế và việc tuần sau

1. Chạy đủ 3 seed cho Random Forest, Extra Trees và MLP nếu còn thời gian GPU/CPU.
2. Bổ sung biểu đồ so sánh trực quan (bar chart) giữa các mô hình trong báo cáo Word.
3. Đưa các bảng trong báo cáo tuần vào đúng chương của báo cáo chính (không sửa số liệu).
4. Kiểm tra lại dung lượng và thời gian chạy để chọn cấu hình trình diễn.
5. Giữ nguyên nguyên tắc: mọi số liệu phải sinh lại được từ CLI.

## 12. Tệp sinh tự động

- `BAO_CAO_TIEN_DO_TUAN_CLI_THUC_NGHIEM.md` / `.docx` — báo cáo này.
- `ABSA_Project/results/cli_experiments/summary.md` — bảng tổng hợp mean ± std.
- `ABSA_Project/results/dataset_histograms/DATASET_ANALYSIS.txt` — thống kê dữ liệu.
- `ABSA_Project/results/report_assets/comparison_with_related_studies.md` — so sánh.
- `ABSA_Project/results/report_assets/metric_applicability.csv` — độ đo áp dụng.
- `ABSA_Project/results/report_assets/figure_table_code_manifest.csv` — hình/bảng ↔ code.

*Sinh tự động lúc 14/11/2026 20:50 bởi `ABSA_Project/src/build_weekly_report.py`.*
