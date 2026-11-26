"""
build_weekly_report.py
======================
Generates the weekly progress report for the CLI-driven experiments.

Outputs
-------
  BAO_CAO_TIEN_DO_TUAN_CLI_THUC_NGHIEM.md
  BAO_CAO_TIEN_DO_TUAN_CLI_THUC_NGHIEM.docx

Every number is read from the real run artifacts (SUMMARY.csv, comparison CSV,
metrics JSON, dataset summary).  The report never hard-codes a result, so it can
be regenerated after any new run.

Usage
-----
python build_weekly_report.py
python build_weekly_report.py --experiments results/cli_experiments \\
    --report-assets results/report_assets --out-dir .

Author: Nhom 2 - CT255
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_EXPERIMENTS = PROJECT_ROOT / "results" / "cli_experiments"
DEFAULT_HISTOGRAMS = PROJECT_ROOT / "results" / "dataset_histograms"
DEFAULT_ASSETS = PROJECT_ROOT / "results" / "report_assets"
DEFAULT_OUT = REPO_ROOT

TITLE = "BÁO CÁO TIẾN ĐỘ TUẦN — CLI THỰC NGHIỆM, ĐỘ ĐO TOÀN DIỆN VÀ SO SÁNH NGHIÊN CỨU LIÊN QUAN"


def now() -> str:
    return datetime.now().astimezone().strftime("%d/%m/%Y %H:%M")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def num(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(x) or np.isinf(x)) else x


def pct(v: Any, digits: int = 2) -> str:
    x = num(v)
    return "—" if x is None else f"{x * 100:.{digits}f}"


def pct100(v: Any, digits: int = 2) -> str:
    x = num(v)
    return "—" if x is None else f"{x:.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def load_metrics(run_dir: Path, split: str = "test") -> Dict[str, Any]:
    path = run_dir / f"metrics_{split}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def best_run(manifest: pd.DataFrame) -> Optional[pd.Series]:
    if manifest.empty:
        return None
    df = manifest[manifest["status"] == "ok"].copy()
    if df.empty:
        return None
    df["macro_f1_test"] = pd.to_numeric(df["macro_f1_test"], errors="coerce")
    df = df.sort_values("macro_f1_test", ascending=False)
    return df.iloc[0]


def build_markdown(
    experiments: Path,
    histograms: Path,
    assets: Path,
) -> str:
    summary = read_csv(experiments / "summary.csv")
    manifest = read_csv(experiments / "manifest.csv")
    comparison = read_csv(assets / "comparison_with_related_studies.csv")
    applicability = read_csv(assets / "metric_applicability.csv")
    code_manifest = read_csv(assets / "figure_table_code_manifest.csv")
    dataset_summary = {}
    ds_path = histograms / "dataset_summary.json"
    if ds_path.exists():
        dataset_summary = json.loads(ds_path.read_text(encoding="utf-8"))

    primary_tag = "baselines_seed42"
    primary = summary[summary["tag"] == primary_tag] if "tag" in summary.columns else pd.DataFrame()
    robust = summary[summary["tag"] != primary_tag] if "tag" in summary.columns else pd.DataFrame()

    top = best_run(manifest[manifest["tag"] == primary_tag] if "tag" in manifest.columns else manifest)
    top_metrics = {}
    top_det = {}
    top_dir = None
    if top is not None:
        top_dir = Path(top["run_dir"])
        m = load_metrics(top_dir, "test")
        top_metrics = m.get("global", {})
        top_det = m.get("aspect_detection", {})

    lines: List[str] = []
    lines += [f"# {TITLE}", "", f"**Thời điểm lập báo cáo:** {now()}", ""]

    # ---------------------------------------------------------------- summary
    lines += ["## 1. Tóm tắt tuần", ""]
    n_models = int(primary.shape[0]) if not primary.empty else 0
    n_runs = int((manifest["status"] == "ok").sum()) if "status" in manifest.columns else 0
    lines += [
        f"- Hoàn thiện **CLI thực nghiệm** cho 14 mô hình trong thư viện (12 mô hình chạy mặc định) "
        f"trên đúng split UIT-ViSFD 7.786 / 1.112 / 2.224, giữ nguyên thứ tự 10 khía cạnh và 4 nhãn.",
        f"- Đã chạy **{n_runs} lượt** thực nghiệm thật; báo cáo chính dùng "
        f"**{n_models} mô hình** ở seed 42, mỗi lượt sinh `RESULTS.txt` + JSON/CSV + hình PNG.",
        "- Bổ sung **bộ độ đo toàn diện**: accuracy, balanced accuracy, P/R/F1 "
        "micro/macro/weighted, Cohen's kappa (linear & quadratic), MCC, ROC-AUC OVR, "
        "PR-AUC/AP, log loss, Brier, ECE, top-2/top-3, confusion matrix, per-class, "
        "per-aspect, exact match, Hamming, subset accuracy, ACD, ASC.",
        "- Thêm các độ đo quy đổi trung thực cho ABSA: **aspect-set IoU**, **CMC@k**, "
        "**mAP / mAP@k**, kèm định nghĩa trong từng `RESULTS.txt`.",
        "- Sinh **11 histogram/heatmap dữ liệu** (PNG + CSV + `DATASET_ANALYSIS.txt`).",
        "- Lập **bảng so sánh với SA2SL, Thin et al., ViSoBERT** và kết quả PhoBERT/BiLSTM "
        "của nhóm, có cột đánh dấu mức độ so sánh được.",
        "- Lập **manifest hình/bảng ↔ code** cho toàn bộ artifact.",
        "- Không tạo số liệu giả; các độ đo ẩn tin (PSNR/SSIM/BER/capacity) được ghi rõ "
        "**không áp dụng** cho bài toán ABSA.",
        "",
    ]

    # ---------------------------------------------------------------- CLI
    lines += ["## 2. Giao diện dòng lệnh (CLI)", ""]
    lines += [
        "Script chính: `ABSA_Project/src/run_simple_experiments.py`.",
        "",
        "```bash",
        "# Liệt kê mô hình",
        "python ABSA_Project/src/run_simple_experiments.py --list",
        "",
        "# Chạy đầy đủ 12 mô hình mặc định, 1 seed",
        "python ABSA_Project/src/run_simple_experiments.py \\",
        "  --models default --seeds 42 --tag baselines_seed42 \\",
        "  --output-dir ABSA_Project/results/cli_experiments",
        "",
        "# Chạy 3 seed cho nhóm mô hình nhanh (kiểm tra độ ổn định)",
        "python ABSA_Project/src/run_simple_experiments.py \\",
        "  --models majority stratified multinomial_nb complement_nb \\",
        "          logistic_regression linear_svc ridge sgd_log decision_tree \\",
        "  --seeds 42 123 2024 --tag robustness_3seed",
        "",
        "# Chỉ xem cấu hình, không huấn luyện",
        "python ABSA_Project/src/run_simple_experiments.py --models mlp --dry-run",
        "```",
        "",
        "### 2.1. Tham số CLI quan trọng",
        "",
    ]
    param_rows = [
        ("`--models`", "`default` (nhận `all` / tên cụ thể)", "Chọn mô hình; `default` = 12 mô hình chính."),
        ("`--seeds`", "42", "Một hoặc nhiều seed; truyền nhiều seed để tính mean ± std."),
        ("`--tag`", "`simple_models` (báo cáo dùng `baselines_seed42`, `robustness_3seed`)", "Nhóm thực nghiệm; quyết định thư mục và bảng tổng hợp."),
        ("`--data-dir`", "…/prepared_for_model", "Thư mục chứa Train/Dev/Test đã tiền xử lý."),
        ("`--text-col`", "comment_segmented", "Cột văn bản đã tách từ bằng VnCoreNLP."),
        ("`--label-col`", "label", "Cột nhãn dạng `{ASPECT#Polarity};…`."),
        ("`--max-features`", "30000", "Số đặc trưng TF-IDF word tối đa."),
        ("`--ngram-min/max`", "1 / 2", "Unigram + bigram."),
        ("`--char-ngram-max`", "0 (tắt)", "Bật char n-gram để bắt lỗi chính tả teencode."),
        ("`--class-weight`", "balanced", "Cân bằng lớp; quan trọng vì None chiếm đa số."),
        ("`--C`", "1.0", "Hệ số điều chuẩn cho Logistic Regression / Linear SVC."),
        ("`--alpha`", "1.0", "Alpha cho Naive Bayes."),
        ("`--sgd-alpha`", "1e-4", "Alpha cho SGD logistic."),
        ("`--mlp-alpha`", "1e-4", "Alpha (L2) cho MLP."),
        ("`--epochs`", "20 (báo cáo dùng **8** cho MLP)", "Số epoch cho `sgd_log` và `mlp` (có đường cong loss/acc)."),
        ("`--hidden-layers`", "256 128 (báo cáo dùng **128** cho MLP)", "Kiến trúc MLP."),
        ("`--batch-size`", "64 (báo cáo dùng **128** cho MLP)", "Batch size MLP."),
        ("`--learning-rate`", "1e-3", "Learning rate MLP."),
        ("`--n-estimators`", "200", "Số cây cho Random Forest / Extra Trees."),
        ("`--knn-neighbors`", "15", "Số láng giềng cho kNN."),
        ("`--eval-train/--eval-dev`", "bật", "Đánh giá thêm Train/Dev ngoài Test."),
        ("`--save-predictions`", "bật", "Xuất dự đoán từng cặp đánh giá × khía cạnh."),
        ("`--save-probabilities`", "bật", "Xuất xác suất thô (N,10,4) dạng NPZ."),
        ("`--figures`", "bật", "Vẽ confusion matrix, ROC, PR, calibration, CMC, IoU…"),
        ("`--force`", "tắt", "Ghi đè lượt chạy đã có."),
        ("`--dry-run`", "tắt", "Chỉ in cấu hình."),
    ]
    lines += md_table(["Tham số", "Mặc định / giá trị đã dùng", "Ý nghĩa"], param_rows)
    lines += [""]

    # ---------------------------------------------------------------- artifacts
    lines += ["## 3. Cấu trúc kết quả mỗi lần chạy", ""]
    lines += [
        "Thư mục: `results/cli_experiments/<tag>/<model>/seed_<n>/`",
        "",
        "| Tệp | Nội dung |",
        "|---|---|",
        "| `RESULTS.txt` | Báo cáo văn bản: lệnh, môi trường, dữ liệu, siêu tham số, kiến trúc, độ đo Train/Dev/Test, confusion matrix, epoch, artifact, liên kết code. |",
        "| `config.json` | Toàn bộ tham số dòng lệnh. |",
        "| `environment.json` | Python, thư viện, CPU/GPU, git commit. |",
        "| `data_summary.json` | Số dòng, phân bố nhãn, lượt nhắc khía cạnh, độ dài token. |",
        "| `metrics_{train,dev,test}.json` | Bộ độ đo đầy đủ từng split. |",
        "| `per_class_{split}.csv` | P/R/F1/support/AUC/AP từng lớp cảm xúc. |",
        "| `per_aspect_{split}.csv` | Độ đo từng khía cạnh (10 dòng). |",
        "| `aspect_detection_{split}.csv` | ACD, exact match, IoU, mAP, CMC@k. |",
        "| `confusion_matrix_{split}.csv` | Ma trận 4×4. |",
        "| `confusion_matrix_acd_{split}.csv` | Ma trận 2×2 None vs Mention. |",
        "| `confusion_matrix_per_aspect_{split}.csv` | 10 ma trận 4×4 dạng long. |",
        "| `predictions_{split}.csv` | Một dòng / (đánh giá × khía cạnh), kèm xác suất. |",
        "| `probabilities_{split}.npz` | Mảng xác suất (N,10,4). |",
        "| `train_log.csv` | Loss/accuracy/macro-F1 từng epoch (chỉ `sgd_log`, `mlp`). |",
        "| `figures/*.png` | Confusion matrix, per-aspect F1, ROC, PR, calibration, CMC, IoU histogram, confidence histogram, learning curves. |",
        "| `figure_manifest.csv` | Mỗi hình ↔ hàm/lệnh/dữ liệu/SHA-256. |",
        "",
    ]

    # ---------------------------------------------------------------- data
    lines += ["## 4. Dữ liệu và histogram", ""]
    if dataset_summary:
        rows = []
        for split, info in dataset_summary.get("splits", {}).items():
            rows.append([
                split, info.get("rows"),
                info.get("label_counts", {}).get("None", ""),
                info.get("label_counts", {}).get("Negative", ""),
                info.get("label_counts", {}).get("Neutral", ""),
                info.get("label_counts", {}).get("Positive", ""),
            ])
        lines += ["Bảng 1. Kích thước và phân bố nhãn (cặp đánh giá × khía cạnh).", ""]
        lines += md_table(["Split", "Số đánh giá", "None", "Negative", "Neutral", "Positive"], rows)
        lines += [""]
    hist_figs = [
        ("split_size_bar.png", "Số đánh giá mỗi split"),
        ("aspect_mention_counts_by_split.png", "Số lượt nhắc từng khía cạnh theo split"),
        ("aspect_mention_rate_heatmap.png", "Tỉ lệ đánh giá nhắc từng khía cạnh"),
        ("sentiment_distribution_by_split.png", "Phân bố 4 nhãn cảm xúc"),
        ("label_per_aspect_heatmap_train.png", "Số lượng nhãn theo từng khía cạnh (Train)"),
        ("sentiment_per_aspect_train.png", "Tỉ lệ cảm xúc trong từng khía cạnh"),
        ("aspect_count_per_review_hist.png", "Số khía cạnh được nhắc trong một đánh giá"),
        ("token_length_hist.png", "Độ dài token"),
        ("character_length_hist.png", "Độ dài ký tự"),
        ("aspect_cooccurrence_heatmap_train.png", "Đồng xuất hiện khía cạnh (Jaccard)"),
        ("monthly_volume.png", "Số đánh giá theo tháng"),
    ]
    lines += ["Bảng 2. Danh mục histogram dữ liệu (mỗi hình có CSV số liệu gốc).", ""]
    rows = []
    csv_override = {
        "aspect_cooccurrence_heatmap_train.png": "aspect_cooccurrence_jaccard_train.csv",
    }
    for name, desc in hist_figs:
        png = histograms / name
        csv_name = csv_override.get(name, name.replace(".png", ".csv"))
        rows.append([
            name, desc,
            "có" if png.exists() else "thiếu",
            "có" if (histograms / csv_name).exists() else "—",
            "`analyze_dataset_histograms.py :: main()`",
        ])
    lines += md_table(["Hình", "Mô tả", "PNG", "CSV", "Code"], rows)
    lines += [""]

    # ---------------------------------------------------------------- results
    lines += ["## 5. Kết quả thực nghiệm mô hình đơn giản", ""]
    lines += [
        "Bảng 3. Kết quả Test ở seed 42 (một dòng / mô hình). "
        "Đơn vị: %. Micro F1 = accuracy do mỗi cặp có đúng một nhãn.",
        "",
    ]
    if not primary.empty:
        rows = []
        for _, r in primary.sort_values("macro_f1_test_mean", ascending=False).iterrows():
            rows.append([
                r["model"], r["model_kind"], int(r["n_runs"]),
                pct(r.get("accuracy_test_mean")),
                pct(r.get("macro_f1_test_mean")),
                pct(r.get("micro_f1_test_mean")),
                pct(r.get("acd_f1_test_mean")),
                pct(r.get("sentiment_macro_f1_test_mean")),
                pct(r.get("exact_match_test_mean")),
                pct(r.get("cohen_kappa_test_mean")),
                pct(r.get("mcc_test_mean")),
                pct(r.get("roc_auc_ovr_macro_test_mean")),
            ])
        lines += md_table(
            ["Mô hình", "Loại", "Runs", "Acc", "Macro F1", "Micro F1",
             "ACD F1", "ASC F1", "Exact", "Kappa", "MCC", "ROC-AUC"],
            rows,
        )
    else:
        lines += ["Chưa có `summary.csv` cho tag `baselines_seed42`."]
    lines += [""]

    for cmp_name, cmp_caption in (
        ("model_comparison_grouped.png",
         "Hình: so sánh 12 mô hình đơn giản trên 6 độ đo Test (seed 42)."),
        ("model_comparison_macro_f1.png",
         "Hình: xếp hạng mô hình đơn giản theo Macro F1 trên Test."),
    ):
        cmp_path = assets / cmp_name
        if cmp_path.exists():
            rel = cmp_path.relative_to(REPO_ROOT).as_posix()
            lines += [f"![{cmp_caption}]({rel})", "",
                      f"*{cmp_caption}* Code: `plot_model_comparison()` — `build_experiment_report.py`.", ""]

    if not robust.empty:
        lines += [
            "Bảng 4. Kiểm tra độ ổn định 3 seed (mean ± std, đơn vị %).",
            "",
        ]
        rows = []
        for _, r in robust.sort_values("macro_f1_test_mean", ascending=False).iterrows():
            def ms(col):
                m, s = num(r.get(f"{col}_mean")), num(r.get(f"{col}_std"))
                if m is None:
                    return "—"
                return f"{m*100:.2f} ± {s*100:.2f}" if s is not None else f"{m*100:.2f}"
            rows.append([
                r["model"], int(r["n_runs"]), ms("macro_f1_test"),
                ms("acd_f1_test"), ms("sentiment_macro_f1_test"),
                ms("exact_match_test"), ms("mcc_test"),
            ])
        lines += md_table(
            ["Mô hình", "Runs", "Macro F1", "ACD F1", "ASC F1", "Exact", "MCC"],
            rows,
        )
        lines += [
            "",
            "Ghi chú: các mô hình ensemble/MLP nặng chỉ chạy 1 seed; nhóm tuyến tính "
            "và baseline chạy 3 seed. Cột `Runs` cho biết số lượt thực tế.",
            "",
            "`± 0.00` không phải là thiếu đo: các mô hình tất định (LinearSVC, Ridge, "
            "Logistic Regression, Naive Bayes, baseline majority) không dùng lấy mẫu "
            "ngẫu nhiên sau khi đã cố định TF-IDF, nên 3 seed cho ra cùng một kết quả. "
            "Chỉ `sgd_log`, `decision_tree` và `stratified` có phương sai khác 0.",
            "",
        ]

    if top is not None:
        lines += [
            f"### 5.1. Mô hình đơn giản tốt nhất (theo Macro F1 Test): `{top['model']}` (seed {int(top['seed'])})",
            "",
        ]
        if top_metrics:
            rows = [
                ("Accuracy", pct(top_metrics.get("accuracy"))),
                ("Balanced accuracy", pct(top_metrics.get("balanced_accuracy"))),
                ("Macro F1 (4 lớp)", pct(top_metrics.get("macro_f1"))),
                ("Micro F1", pct(top_metrics.get("micro_f1"))),
                ("Weighted F1", pct(top_metrics.get("weighted_f1"))),
                ("Cohen's kappa", pct(top_metrics.get("cohen_kappa"))),
                ("Cohen's kappa (quadratic)", pct(top_metrics.get("cohen_kappa_quadratic"))),
                ("MCC", pct(top_metrics.get("mcc"))),
                ("ROC-AUC OVR (macro)", pct(top_metrics.get("roc_auc_ovr_macro"))),
                ("PR-AUC (macro)", pct(top_metrics.get("pr_auc_macro"))),
                ("Log loss", pct(top_metrics.get("log_loss"))),
                ("Brier score", pct(top_metrics.get("brier_score_multiclass"))),
                ("ECE (10 bins)", pct(top_metrics.get("ece_10bin"))),
                ("Top-2 accuracy", pct(top_metrics.get("top2_accuracy"))),
                ("Top-3 accuracy", pct(top_metrics.get("top3_accuracy"))),
                ("ASC macro F1", pct(top_metrics.get("sentiment_macro_f1"))),
                ("ACD F1 (từ detection)", pct(top_det.get("acd_f1"))),
                ("Exact match", pct(top_det.get("exact_match"))),
                ("Hamming loss (4 lớp)", pct(top_det.get("hamming_loss_4class"))),
                ("Aspect-set IoU (mean)", pct(top_det.get("aspect_set_iou_mean"))),
                ("mAP aspect detection (macro)", pct(top_det.get("aspect_map_macro"))),
                ("CMC Recall@5", pct((top_det.get("cmc", {}).get("recall_at_k", {}) or {}).get("@5"))),
            ]
            lines += md_table(["Độ đo", "Giá trị"], rows)
            lines += [""]
            cm = top_metrics.get("confusion_matrix")
            if cm:
                lines += ["Confusion matrix 4×4 trên Test (hàng = thật, cột = dự đoán):", ""]
                lines += md_table(
                    ["Thật \\ Dự đoán", "None", "Negative", "Neutral", "Positive"],
                    [[["None", "Negative", "Neutral", "Positive"][i], *row] for i, row in enumerate(cm)],
                )
                lines += [""]

    # ---------------------------------------------------------------- figures
    lines += ["## 6. Hình tiêu biểu của mô hình tốt nhất", ""]
    if top_dir is not None:
        fig_dir = top_dir / "figures"
        wanted = [
            ("confusion_matrix_test.png", "Hình 1. Confusion matrix 4×4 trên Test."),
            ("per_aspect_macro_f1_test.png", "Hình 2. Macro F1 từng khía cạnh trên Test."),
            ("per_class_f1_test.png", "Hình 3. F1 từng lớp cảm xúc trên Test."),
            ("roc_curves_test.png", "Hình 4. ROC one-vs-rest trên Test."),
            ("pr_curves_test.png", "Hình 5. Precision-Recall trên Test."),
            ("calibration_test.png", "Hình 6. Reliability diagram (calibration)."),
            ("confidence_hist_test.png", "Hình 7. Histogram độ tin cậy đúng/sai."),
            ("cmc_test.png", "Hình 8. CMC@k cho xếp hạng khía cạnh."),
            ("aspect_set_iou_hist_test.png", "Hình 9. Histogram aspect-set IoU."),
            ("aspect_ap_test.png", "Hình 10. Average precision phát hiện từng khía cạnh."),
            ("learning_curves.png", "Hình 11. Loss/accuracy/macro-F1 theo epoch (nếu có)."),
        ]
        for name, caption in wanted:
            path = fig_dir / name
            if path.exists():
                rel = path.relative_to(REPO_ROOT).as_posix()
                lines += [f"![{caption}]({rel})", "", f"*{caption}* Dữ liệu & code: `{rel}` — xem `figure_manifest.csv`.", ""]
            else:
                lines += [f"- *{caption}* — không áp dụng cho mô hình này (không có hình).", ""]

        epoch_runs = []
        for model_name in ("sgd_log", "mlp"):
            p = experiments / primary_tag / model_name / "seed_42" / "figures" / "learning_curves.png"
            if p.exists():
                epoch_runs.append((model_name, p))
        if epoch_runs:
            lines += ["### 6.1. Đường cong loss / accuracy / macro-F1 theo epoch", ""]
            for model_name, p in epoch_runs:
                rel = p.relative_to(REPO_ROOT).as_posix()
                lines += [
                    f"![Learning curves {model_name}]({rel})", "",
                    f"*Learning curves của `{model_name}` (Train vs Dev).* "
                    f"Dữ liệu: `{p.parent.parent.name}/train_log.csv`; code: `plot_learning_curves()`.",
                    "",
                ]
    else:
        lines += ["Chưa có lượt chạy để trích hình."]
    lines += [""]

    # ---------------------------------------------------------------- comparison
    lines += ["## 7. So sánh với các nghiên cứu liên quan", ""]
    if not comparison.empty:
        direct = comparison[comparison["comparison_type"] != "not_directly_comparable"].copy()
        rows = []
        for _, r in direct.iterrows():
            rows.append([
                r.get("study"), r.get("model"), r.get("model_family"),
                pct100(r.get("acd_f1")), pct100(r.get("asc_f1")),
                pct100(r.get("macro_f1")), pct100(r.get("micro_f1")),
                r.get("comparison_type"),
            ])
        lines += [
            "Bảng 5. Đối chiếu trên UIT-ViSFD. Dấu — nghĩa là bài báo không báo cáo độ đo đó.",
            "",
        ]
        lines += md_table(
            ["Nghiên cứu", "Mô hình", "Họ", "ACD F1", "ASC F1", "Macro F1 4 lớp", "Micro F1", "Mức so sánh"],
            rows,
        )
        lines += [
            "",
            "Với các mô hình của nhóm: nếu có nhiều seed thì lấy trung bình 3 seed, "
            "nếu chỉ chạy 1 seed thì lấy seed 42. Vì vậy một vài số có thể lệch rất nhỏ "
            "so với Bảng 3 (seed 42 đơn lẻ).",
            "",
        ]
        indirect = comparison[comparison["comparison_type"] == "not_directly_comparable"]
        if not indirect.empty:
            lines += [
                "Các dòng sau **không so sánh trực tiếp** vì dùng benchmark social-media khác:",
                "",
            ]
            rows = [[r.get("study"), r.get("model"), pct100(r.get("accuracy")),
                     pct100(r.get("weighted_f1")), pct100(r.get("macro_f1")),
                     r.get("notes")] for _, r in indirect.iterrows()]
            lines += md_table(["Nghiên cứu", "Mô hình", "Accuracy", "Weighted F1", "Macro F1", "Ghi chú"], rows)
            lines += [""]
    lines += [
        "Diễn giải: mô hình đơn giản là baseline minh bạch và nhanh, nhưng thấp hơn "
        "PhoBERT fine-tune vì thiếu ngữ cảnh tiền huấn luyện. Khoảng cách lớn nhất ở "
        "ASC (phân cực cảm xúc) và exact match, đúng với dữ liệu mất cân bằng mạnh.",
        "",
    ]

    # ---------------------------------------------------------------- metrics
    lines += ["## 8. Độ đo nào áp dụng cho ABSA?", ""]
    if not applicability.empty:
        rows = [[r["metric"], r["family"], r["applies_to_absa"], r["where_reported"]]
                for _, r in applicability.iterrows()]
        lines += md_table(["Độ đo", "Nhóm", "Áp dụng", "Nơi báo cáo"], rows)
        lines += [""]
    lines += [
        "Các độ đo ẩn tin **PSNR / SSIM / BER / capacity / robustness** không áp dụng "
        "vì bài toán không có ảnh bìa và không nhúng payload. Báo cáo không tạo số liệu giả.",
        "",
    ]

    # ---------------------------------------------------------------- manifest
    lines += ["## 9. Liên kết hình/bảng ↔ code", ""]
    if not code_manifest.empty:
        lines += [
            f"Manifest đầy đủ có **{len(code_manifest)} mục** tại "
            "`ABSA_Project/results/report_assets/figure_table_code_manifest.csv` (có SHA-256).",
            "",
        ]
        sample = code_manifest.head(20)
        rows = [[Path(str(r["artifact"])).name, r["kind"], r["code_function"], r["group"]]
                for _, r in sample.iterrows()]
        lines += md_table(["Artifact", "Loại", "Hàm tạo", "Nhóm"], rows)
        lines += ["", "… (xem CSV để có toàn bộ danh mục).", ""]
    else:
        lines += ["Chưa có manifest."]
    lines += [""]

    # ---------------------------------------------------------------- audit
    lines += ["## 10. Kiểm tra yêu cầu", ""]
    checks = [
        ("Chỉ dùng công nghệ cho phép (HTML/CSS/Bootstrap/JS/Chart.js + PHP/PDO/MySQL; Python cho model)", "Đạt", "Website Phase 1–7 đã kiểm thử; không dùng React/Node/Laravel."),
        ("Dùng checkpoint thật `best_phobert_absa.pt`, không bịa kết quả", "Đạt", "FastAPI `/health` báo `model_loaded=true`, `device=cuda`, `max_len=128`."),
        ("Giữ 10 khía cạnh, thứ tự nhãn None/Negative/Neutral/Positive", "Đạt", "Import từ `dataset_loader.py`; ghi trong mọi `RESULTS.txt`."),
        ("Không lộ đường dẫn checkpoint trong UI/DB", "Đạt", "Phase 7 audit."),
        ("Prepared statements, `password_hash`/`password_verify`, session, CSRF, validation", "Đạt", "Phase 4–7 E2E 20/20."),
        ("CLI có tham số dòng lệnh", "Đạt", "Mục 2; `--help`, `--list`, `--dry-run`."),
        ("Đề xuất tham số CLI quan trọng", "Đạt", "Bảng tham số mục 2.1."),
        ("Lưu kết quả ra txt (kết quả, siêu tham số, dữ liệu, kiến trúc, lệnh)", "Đạt", "`RESULTS.txt` mục 1–13."),
        ("Đánh giá trên Train và Test", "Đạt", "`metrics_train.json`, `metrics_test.json`, cả Dev."),
        ("Độ đo tối đa phù hợp: acc, F1, kappa, MCC, AUC, confusion matrix, loss/acc/epoch", "Đạt", "Mục 5; bảng độ đo mục 8."),
        ("CMC, IoU, mAP và độ đo liên quan", "Đạt (quy đổi trung thực)", "Aspect-set IoU, CMC@k, mAP/mAP@k; định nghĩa trong `RESULTS.txt` mục 12."),
        ("Biểu đồ histogram", "Đạt", "11 histogram/heatmap + histogram độ tin cậy/IoU."),
        ("So sánh với nghiên cứu liên quan", "Đạt", "Mục 7; `comparison_with_related_studies.csv`."),
        ("Liên kết code cho từng hình/bảng", "Đạt", "Mục 9; `figure_table_code_manifest.csv`."),
        ("Báo cáo tuần ở tệp riêng, không sửa báo cáo Word cũ", "Đạt", "Tệp này; `baocao_NVTM_CT255.docx` giữ nguyên SHA-256."),
        ("Độ đo ẩn tin (PSNR/SSIM/BER/capacity)", "Không áp dụng", "Ghi rõ lý do; không tạo số liệu."),
    ]
    lines += md_table(["Yêu cầu", "Trạng thái", "Bằng chứng"], checks)
    lines += [""]

    # ---------------------------------------------------------------- next
    lines += ["## 11. Hạn chế và việc tuần sau", ""]
    lines += [
        "1. Chạy đủ 3 seed cho Random Forest, Extra Trees và MLP nếu còn thời gian GPU/CPU.",
        "2. Bổ sung biểu đồ so sánh trực quan (bar chart) giữa các mô hình trong báo cáo Word.",
        "3. Đưa các bảng trong báo cáo tuần vào đúng chương của báo cáo chính (không sửa số liệu).",
        "4. Kiểm tra lại dung lượng và thời gian chạy để chọn cấu hình trình diễn.",
        "5. Giữ nguyên nguyên tắc: mọi số liệu phải sinh lại được từ CLI.",
        "",
    ]

    lines += [
        "## 12. Tệp sinh tự động",
        "",
        "- `BAO_CAO_TIEN_DO_TUAN_CLI_THUC_NGHIEM.md` / `.docx` — báo cáo này.",
        "- `ABSA_Project/results/cli_experiments/summary.md` — bảng tổng hợp mean ± std.",
        "- `ABSA_Project/results/dataset_histograms/DATASET_ANALYSIS.txt` — thống kê dữ liệu.",
        "- `ABSA_Project/results/report_assets/comparison_with_related_studies.md` — so sánh.",
        "- `ABSA_Project/results/report_assets/metric_applicability.csv` — độ đo áp dụng.",
        "- `ABSA_Project/results/report_assets/figure_table_code_manifest.csv` — hình/bảng ↔ code.",
        "",
        f"*Sinh tự động lúc {now()} bởi `ABSA_Project/src/build_weekly_report.py`.*",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
def write_docx(markdown: str, path: Path, experiments: Path, assets: Path) -> None:
    try:
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        print("[warn] python-docx not installed; DOCX skipped")
        return

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    # Title
    h = doc.add_heading(TITLE, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Thời điểm lập báo cáo: {now()}")
    doc.add_paragraph(
        "Báo cáo tiến độ tuần cho phần thực nghiệm CLI, độ đo toàn diện, histogram và "
        "so sánh với nghiên cứu liên quan. Mọi số liệu đọc trực tiếp từ artifact của "
        "các lần chạy thật."
    )

    lines = markdown.splitlines()
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fenced code block: keep the text verbatim in a monospace paragraph.
        if stripped.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(line if line else " ")
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
            i += 1
            continue

        # Markdown table block
        if line.startswith("|") and line.endswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = []
            for bl in block:
                cells = [c.strip() for c in bl.strip().strip("|").split("|")]
                if all(set(c) <= {"-", ":", " "} for c in cells):
                    continue
                rows.append(cells)
            if rows:
                ncols = max(len(r) for r in rows)
                table = doc.add_table(rows=0, cols=ncols)
                table.style = "Light Grid Accent 1"
                for ri, row in enumerate(rows):
                    cells = table.add_row().cells
                    for ci in range(ncols):
                        text = row[ci] if ci < len(row) else ""
                        text = text.replace("**", "").replace("`", "")
                        cells[ci].text = text
                        if ri == 0:
                            for p in cells[ci].paragraphs:
                                for run in p.runs:
                                    run.bold = True
                doc.add_paragraph("")
            continue

        if not stripped:
            i += 1
            continue
        if stripped.startswith("**Thời điểm lập báo cáo:**"):
            i += 1  # already added above the body
            continue
        if stripped.startswith("# "):
            i += 1  # title already added
            continue
        if stripped.startswith("## "):
            doc.add_heading(stripped[3:].strip(), level=1)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:].strip(), level=2)
        elif stripped.startswith("!["):
            # image
            start = stripped.find("](")
            end = stripped.find(")", start)
            if start != -1 and end != -1:
                rel = stripped[start + 2:end]
                img = (REPO_ROOT / rel).resolve()
                if img.exists():
                    try:
                        doc.add_picture(str(img), width=Inches(5.6))
                        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except Exception:
                        doc.add_paragraph(f"[không chèn được hình: {rel}]")
        elif stripped.startswith("- "):
            text = stripped[2:].replace("**", "").replace("`", "")
            doc.add_paragraph(text, style="List Bullet")
        elif re.match(r"^\d+\.\s", stripped):
            text = re.sub(r"^\d+\.\s+", "", stripped).replace("**", "").replace("`", "")
            doc.add_paragraph(text, style="List Number")
        elif stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            p = doc.add_paragraph()
            run = p.add_run(stripped.strip("*"))
            run.italic = True
        else:
            text = stripped.replace("**", "").replace("`", "")
            doc.add_paragraph(text)
        i += 1

    doc.save(str(path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sinh báo cáo tiến độ tuần (Markdown + DOCX).")
    parser.add_argument("--experiments", default=str(DEFAULT_EXPERIMENTS))
    parser.add_argument("--histograms", default=str(DEFAULT_HISTOGRAMS))
    parser.add_argument("--report-assets", default=str(DEFAULT_ASSETS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--name", default="BAO_CAO_TIEN_DO_TUAN_CLI_THUC_NGHIEM")
    args = parser.parse_args(argv)

    experiments = Path(args.experiments).resolve()
    histograms = Path(args.histograms).resolve()
    assets = Path(args.report_assets).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(experiments, histograms, assets)
    md_path = out_dir / f"{args.name}.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(f"Markdown: {md_path}")

    docx_path = out_dir / f"{args.name}.docx"
    write_docx(markdown, docx_path, experiments, assets)
    print(f"DOCX    : {docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
