"""
build_experiment_report.py
==========================
Aggregates the CLI experiment artifacts into report-ready tables:

  comparison_with_related_studies.csv / .md
      Our simple-model runs next to the numbers reported by SA2SL, Thin et al.
      and ViSoBERT, plus the group's earlier PhoBERT/BiLSTM results.  Every row
      carries a comparability flag, so a reader can never mistake a different
      benchmark for a direct comparison.

  metric_applicability.csv
      Which metric is native to ABSA, which is an honest adaptation (IoU, CMC,
      mAP) and which does not apply at all (PSNR/SSIM/BER/capacity).

  figure_table_code_manifest.csv
      Every figure and every result table mapped to the function that produced
      it, the command that regenerates it, its data source and its SHA-256.

Usage
-----
python build_experiment_report.py
python build_experiment_report.py --experiments results/cli_experiments \\
    --histograms results/dataset_histograms \\
    --literature data/literature_baselines.csv \\
    --output results/report_assets

Author: Nhom 2 - CT255
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_EXPERIMENTS = PROJECT_ROOT / "results" / "cli_experiments"
DEFAULT_HISTOGRAMS = PROJECT_ROOT / "results" / "dataset_histograms"
DEFAULT_LITERATURE = PROJECT_ROOT / "data" / "literature_baselines.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "report_assets"

RUNNER = SRC_DIR / "run_simple_experiments.py"
HISTOGRAM_SCRIPT = SRC_DIR / "analyze_dataset_histograms.py"
METRICS_FILE = SRC_DIR / "experiment_metrics.py"
ZOO_FILE = SRC_DIR / "simple_model_zoo.py"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[warn] missing {path}")
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def num(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else v


def fmt(value: Any, digits: int = 2) -> str:
    v = num(value)
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# Related-study comparison
# ---------------------------------------------------------------------------
def load_group_runs(experiments_dir: Path) -> pd.DataFrame:
    """One comparison row per model, using the multi-seed mean when available."""
    summary = read_csv(experiments_dir / "summary.csv")
    if summary.empty:
        return summary
    # Prefer the tag with the most runs for each model (3-seed robustness wins).
    summary = summary.copy()
    summary["n_runs"] = pd.to_numeric(summary["n_runs"], errors="coerce").fillna(0)
    summary = summary.sort_values("n_runs", ascending=False).drop_duplicates("model")

    def metric(row: pd.Series, col: str) -> Optional[float]:
        v = num(row.get(f"{col}_mean"))
        return v * 100 if v is not None else None

    rows = []
    for _, r in summary.iterrows():
        n = int(r.get("n_runs") or 1)
        rows.append(
            {
                "study_id": f"this_run::{r['model']}",
                "study": "Nhóm 2 — mô hình đơn giản (báo cáo này)",
                "year": 2026,
                "venue": "internal CLI experiment",
                "dataset": "UIT-ViSFD",
                "task": "ACD+ASC+4-class (10 heads)",
                "model": r["model"],
                "model_family": r.get("model_kind"),
                "acd_precision": None,
                "acd_recall": None,
                "acd_f1": metric(r, "acd_f1_test"),
                "asc_precision": None,
                "asc_recall": None,
                "asc_f1": metric(r, "sentiment_macro_f1_test"),
                "macro_f1": metric(r, "macro_f1_test"),
                "macro_f1_scope": "4-class incl None (flattened 10 heads)",
                "accuracy": metric(r, "accuracy_test"),
                "micro_f1": metric(r, "micro_f1_test"),
                "weighted_f1": metric(r, "weighted_f1_test"),
                "exact_match": metric(r, "exact_match_test"),
                "comparison_type": "same splits (Train/Dev/Test), reproduced",
                "source": r.get("tag", ""),
                "notes": (
                    f"tag={r.get('tag')}; n_runs={n}; seeds={r.get('seeds', '')}; "
                    f"kappa={fmt(num(r.get('cohen_kappa_test_mean')))}; "
                    f"mcc={fmt(num(r.get('mcc_test_mean')))}; "
                    f"IoU={fmt(num(r.get('aspect_set_iou_mean_test_mean')))}"
                ),
                "tag": r.get("tag", ""),
                "seed": r.get("seeds", ""),
            }
        )
    return pd.DataFrame(rows)


def build_comparison(runs: pd.DataFrame, literature: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not literature.empty:
        lit = literature.copy()
        lit["tag"] = ""
        lit["seed"] = ""
        frames.append(lit)
    if not runs.empty:
        frames.append(runs)
    if not frames:
        return pd.DataFrame()
    cols = [
        "study_id", "study", "year", "venue", "dataset", "task", "model",
        "model_family", "acd_precision", "acd_recall", "acd_f1",
        "asc_precision", "asc_recall", "asc_f1",
        "macro_f1", "macro_f1_scope", "accuracy", "micro_f1", "weighted_f1",
        "exact_match", "comparison_type", "source", "notes", "tag", "seed",
    ]
    combined = []
    for frame in frames:
        for c in cols:
            if c not in frame.columns:
                frame[c] = ""
        combined.append(frame[cols])
    out = pd.concat(combined, ignore_index=True)

    # Sort: direct comparisons first (reproduced > reported > not comparable),
    # then by ACD F1 descending inside each block.
    order = {
        "same splits (Train/Dev/Test), reproduced": 0,
        "reproduced": 1,
        "reported": 2,
        "not_directly_comparable": 3,
    }
    out["_cmp"] = out["comparison_type"].map(lambda x: order.get(str(x), 9))
    out["_sort"] = pd.to_numeric(out["acd_f1"], errors="coerce").fillna(-1)
    out = out.sort_values(["_cmp", "_sort"], ascending=[True, False]).drop(
        columns=["_cmp", "_sort"]
    )
    return out.reset_index(drop=True)


def comparison_markdown(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "# So sánh với các nghiên cứu liên quan\n\nChưa có dữ liệu.\n"

    numeric_cols = {
        "acd_precision", "acd_recall", "acd_f1", "asc_precision", "asc_recall",
        "asc_f1", "macro_f1", "accuracy", "micro_f1", "weighted_f1", "exact_match",
    }

    def cell(value: Any, column: str) -> str:
        if column in numeric_cols:
            v = num(value)
            return "—" if v is None else f"{v:.2f}"
        if value is None:
            return "—"
        text = str(value)
        if text.lower() in {"nan", "none", ""}:
            return "—"
        return text

    def table(df: pd.DataFrame, cols: Sequence[str], headers: Sequence[str]) -> List[str]:
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(cell(r.get(c, ""), c) for c in cols) + " |")
        return lines

    lines = [
        "# So sánh với các nghiên cứu liên quan — UIT-ViSFD ABSA",
        "",
        f"Sinh lúc: {now_iso()}",
        "",
        "Quy ước so sánh:",
        "",
        "- `same splits (Train/Dev/Test), reproduced`: chạy bởi nhóm trên đúng split 7786/1112/2224.",
        "- `reproduced`: kết quả nhóm đã chạy trước đây trên cùng split (PhoBERT, BiLSTM).",
        "- `reported`: số lấy từ bài báo gốc (có thể khác giao thức chia dữ liệu).",
        "- `not_directly_comparable`: benchmark/dữ liệu khác — chỉ để tham khảo, KHÔNG so sánh trực tiếp.",
        "",
        "## 1. So sánh trực tiếp trên UIT-ViSFD (ACD F1 và ASC F1)",
        "",
    ]
    direct = comparison[comparison["comparison_type"] != "not_directly_comparable"]
    lines += table(
        direct,
        ["study", "model", "model_family", "acd_f1", "asc_f1", "comparison_type"],
        ["Nghiên cứu", "Mô hình", "Họ mô hình", "ACD F1 (%)", "ASC F1 (%)", "Loại so sánh"],
    )
    lines += [
        "",
        "## 2. So sánh trên độ đo 4 lớp / micro F1 / exact match",
        "",
        "Chỉ các dòng có cùng giao thức 10 head (4 lớp gồm None) mới so sánh trực tiếp.",
        "",
    ]
    four = comparison[
        comparison["macro_f1_scope"].astype(str).str.contains("4-class", na=False)
    ]
    lines += table(
        four,
        ["study", "model", "macro_f1", "micro_f1", "accuracy", "exact_match", "comparison_type"],
        ["Nghiên cứu", "Mô hình", "Macro F1 4 lớp (%)", "Micro F1 (%)", "Accuracy (%)", "Exact match (%)", "Loại"],
    )
    lines += [
        "",
        "## 3. Các kết quả KHÔNG so sánh trực tiếp",
        "",
    ]
    indirect = comparison[comparison["comparison_type"] == "not_directly_comparable"]
    if indirect.empty:
        lines.append("Không có.")
    else:
        lines += table(
            indirect,
            ["study", "model", "accuracy", "weighted_f1", "macro_f1", "notes"],
            ["Nghiên cứu", "Mô hình", "Accuracy (%)", "Weighted F1 (%)", "Macro F1 (%)", "Ghi chú"],
        )
    lines += [
        "",
        "## 4. Nhận xét",
        "",
        "- Mô hình đơn giản (TF-IDF + tuyến tính/cây) là baseline minh bạch, chạy nhanh,",
        "  nhưng thấp hơn PhoBERT đã fine-tune vì không có ngữ cảnh tiền huấn luyện.",
        "- Các mô hình tuyến tính học tốt phần ACD (khía cạnh nào được nhắc) hơn phần ASC",
        "  (phân cực cảm xúc), đúng với đặc điểm dữ liệu mất cân bằng mạnh (None chiếm đa số).",
        "- So sánh với SA2SL/Thin et al. chỉ nên đọc theo từng cột độ đo, không gộp",
        "  các độ đo khác định nghĩa vào một con số duy nhất.",
        "- ViSoBERT dùng benchmark social-media khác, không đưa vào kết luận so sánh.",
        "",
        "## 5. Tệp dữ liệu",
        "",
        "- `comparison_with_related_studies.csv`: bảng đầy đủ, dễ lọc.",
        "- `data/literature_baselines.csv`: nguồn số liệu gốc đã kiểm chứng.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric applicability
# ---------------------------------------------------------------------------
METRICS = [
    # metric, family, applies, definition, why, where
    ("Accuracy", "classification", "yes", "Tỉ lệ cặp (đánh giá × khía cạnh) dự đoán đúng nhãn.", "Độ đo cơ bản, dễ hiểu.", "metrics_{split}.json → global.accuracy"),
    ("Balanced accuracy", "classification", "yes", "Trung bình recall từng lớp.", "Chống ảo giác do None chiếm đa số.", "global.balanced_accuracy"),
    ("Precision / Recall / F1 (micro)", "classification", "yes", "Gộp toàn bộ cặp trước khi tính P/R/F1.", "Phù hợp khi mỗi cặp có đúng một nhãn.", "global.micro_*"),
    ("Precision / Recall / F1 (macro)", "classification", "yes", "Trung bình không trọng số trên 4 lớp.", "Đánh giá công bằng lớp hiếm (Neutral).", "global.macro_*"),
    ("Precision / Recall / F1 (weighted)", "classification", "yes", "Trung bình theo support từng lớp.", "Phản ánh phân bố thực tế.", "global.weighted_*"),
    ("Cohen's kappa (linear)", "agreement", "yes", "Đồng thuận có hiệu chỉnh ngẫu nhiên.", "Kiểm tra mô hình có học thật hay đoán mò.", "global.cohen_kappa"),
    ("Cohen's kappa (quadratic)", "agreement", "yes", "Kappa có trọng số bậc hai theo khoảng cách nhãn.", "Phù hợp thứ tự Negative<Neutral<Positive.", "global.cohen_kappa_quadratic"),
    ("Matthews correlation coefficient", "classification", "yes", "Tương quan giữa nhãn thật và dự đoán trên 4 lớp.", "Cân bằng, ổn định với mất cân bằng.", "global.mcc"),
    ("ROC-AUC OVR (macro/micro/weighted)", "ranking", "yes", "Diện tích dưới đường ROC one-vs-rest.", "Đo khả năng xếp hạng xác suất.", "global.roc_auc_ovr_*"),
    ("PR-AUC / Average Precision", "ranking", "yes", "Diện tích dưới đường precision-recall.", "Tốt hơn ROC-AUC khi lớp hiếm.", "global.pr_auc_* + per_class.average_precision"),
    ("Log loss", "probability", "yes", "Cross-entropy của xác suất dự đoán.", "Đo chất lượng xác suất, không chỉ nhãn.", "global.log_loss"),
    ("Brier score (multi-class)", "probability", "yes", "Trung bình bình phương sai lệch xác suất.", "Đo hiệu chỉnh xác suất.", "global.brier_score_multiclass"),
    ("Expected Calibration Error (ECE)", "probability", "yes", "Sai lệch trung bình giữa confidence và accuracy.", "Kiểm tra độ tin cậy có phản ánh đúng.", "global.ece_10bin"),
    ("Top-2 / Top-3 accuracy", "ranking", "yes", "Nhãn thật nằm trong top-k xác suất.", "Hữu ích khi nhãn lân cận dễ nhầm.", "global.top{k}_accuracy"),
    ("Confusion matrix 4×4", "diagnostic", "yes", "Ma trận đếm nhãn thật × nhãn dự đoán.", "Chỉ rõ nhầm lẫn giữa các nhãn.", "confusion_matrix_{split}.csv"),
    ("Confusion matrix ACD (None vs Mention)", "diagnostic", "yes", "Ma trận 2×2 cho phát hiện khía cạnh.", "Tách lỗi phát hiện khỏi lỗi phân cực.", "confusion_matrix_acd_{split}.csv"),
    ("Per-class P/R/F1 + support", "diagnostic", "yes", "Chỉ số từng lớp cảm xúc.", "Phát hiện lớp yếu.", "per_class_{split}.csv"),
    ("Per-aspect P/R/F1 + kappa + MCC", "diagnostic", "yes", "Chỉ số đầy đủ cho từng trong 10 khía cạnh.", "Phát hiện khía cạnh khó (STORAGE).", "per_aspect_{split}.csv"),
    ("Exact match", "multi-output", "yes", "Cả 10 khía cạnh đều đúng nhãn.", "Đo mức đúng trọn vẹn một đánh giá.", "aspect_detection_{split}.csv → exact_match"),
    ("Hamming loss", "multi-output", "yes", "Tỉ lệ cặp dự đoán sai.", "Đo sai số trung bình trên 10 head.", "aspect_detection_{split}.csv"),
    ("Subset accuracy (aspect set)", "multi-output", "yes", "Tập khía cạnh dự đoán trùng khít tập thật.", "Đo riêng phần phát hiện khía cạnh.", "aspect_detection_{split}.csv"),
    ("ACD precision/recall/F1", "ABSA", "yes", "Phát hiện khía cạnh (None vs mention).", "Bài toán con chuẩn của ABSA.", "aspect_detection_{split}.csv → acd_*"),
    ("ASC macro P/R/F1", "ABSA", "yes", "Phân cực cảm xúc chỉ trên lớp 1..3.", "Bài toán con chuẩn thứ hai của ABSA.", "global.sentiment_macro_*"),
    ("Aspect-set precision/recall/F1", "multi-output", "adapted", "P/R/F1 theo tập khía cạnh của từng đánh giá.", "Đánh giá mức khớp tập khía cạnh.", "aspect_detection_{split}.csv"),
    ("IoU / Jaccard", "adapted", "adapted", "Jaccard giữa tập khía cạnh thật và dự đoán của từng đánh giá.", "IoU gốc là độ đo phân đoạn ảnh; ở ABSA dùng Jaccard tập khía cạnh.", "aspect_detection_{split}.csv → aspect_set_iou_mean"),
    ("CMC@k", "adapted", "adapted", "Tỉ lệ khía cạnh thật nằm trong top-k khía cạnh theo mention score = 1 − P(None).", "CMC gốc là nhận dạng sinh trắc; ở ABSA dùng xếp hạng khía cạnh.", "aspect_detection_{split}.csv → cmc_*"),
    ("mAP (aspect detection)", "adapted", "adapted", "Average precision của phát hiện khía cạnh nhị phân, macro/micro.", "mAP gốc là phát hiện đối tượng; ở ABSA dùng AP phát hiện khía cạnh.", "aspect_detection_{split}.csv → aspect_map_*"),
    ("mAP@k", "adapted", "adapted", "AP@k trong xếp hạng khía cạnh của từng đánh giá.", "Phiên bản giới hạn top-k của mAP khía cạnh.", "aspect_detection_{split}.csv → aspect_map@k"),
    ("Loss / accuracy theo epoch", "training", "conditional", "Đường cong huấn luyện từng epoch, chỉ cho sgd_log và mlp.", "Các mô hình closed-form/ensemble không có epoch.", "train_log.csv + figures/learning_curves.png"),
    ("Histogram dữ liệu", "descriptive", "yes", "Phân bố nhãn, khía cạnh, độ dài, đồng xuất hiện.", "Mô tả dữ liệu, không phải kết quả mô hình.", "results/dataset_histograms/"),
    ("Histogram độ tin cậy", "diagnostic", "yes", "Phân bố max probability cho dự đoán đúng/sai.", "Chỉ ra mô hình quá tự tin hay không.", "figures/confidence_hist_test.png"),
    ("PSNR", "steganography", "no", "Peak signal-to-noise ratio giữa ảnh gốc và ảnh nhúng.", "Bài toán ABSA không có ảnh bìa/payload nhúng.", "Không áp dụng — không tạo số liệu."),
    ("SSIM", "steganography", "no", "Độ tương đồng cấu trúc ảnh.", "Không có ảnh để so sánh.", "Không áp dụng."),
    ("BER", "steganography", "no", "Bit error rate của payload.", "Không có payload.", "Không áp dụng."),
    ("Capacity", "steganography", "no", "Số bit nhúng được.", "Không có kênh nhúng.", "Không áp dụng."),
    ("Robustness (JPEG/crop/noise)", "steganography", "no", "Độ bền payload trước biến đổi ảnh.", "Không có ảnh/payload.", "Không áp dụng."),
]


def build_metric_applicability() -> pd.DataFrame:
    return pd.DataFrame(
        METRICS,
        columns=["metric", "family", "applies_to_absa", "definition", "why", "where_reported"],
    )


# ---------------------------------------------------------------------------
# Comparison figures
# ---------------------------------------------------------------------------
def plot_model_comparison(experiments_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Bar charts comparing every simple model on the primary tag."""
    summary = read_csv(experiments_dir / "summary.csv")
    if summary.empty:
        return []
    tag = "baselines_seed42"
    df = summary[summary["tag"] == tag].copy()
    if df.empty:
        df = summary.copy()
    # Keep one row per model (the tag with the most runs if duplicated).
    df = df.sort_values("n_runs", ascending=False).drop_duplicates("model")
    df = df.sort_values("macro_f1_test_mean", ascending=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("accuracy_test_mean", "Accuracy"),
        ("micro_f1_test_mean", "Micro F1"),
        ("macro_f1_test_mean", "Macro F1"),
        ("acd_f1_test_mean", "ACD F1"),
        ("sentiment_macro_f1_test_mean", "ASC F1"),
        ("exact_match_test_mean", "Exact match"),
    ]
    models = df["model"].tolist()
    y = np.arange(len(models))
    records: List[Dict[str, Any]] = []

    # Grouped horizontal bars
    fig, ax = plt.subplots(figsize=(11, 6.2))
    height = 0.13
    for i, (col, label) in enumerate(metrics):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0) * 100
        ax.barh(y + (i - (len(metrics) - 1) / 2) * height, vals, height=height, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Score (%)")
    ax.set_title("So sánh 12 mô hình đơn giản trên Test (seed 42)")
    ax.set_xlim(0, 100)
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "model_comparison_grouped.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    records.append(
        {
            "group": "report_assets",
            "artifact": str(path),
            "kind": "figure",
            "description": "So sánh 12 mô hình đơn giản trên 6 độ đo Test",
            "code_file": str(Path(__file__).resolve()),
            "code_function": "plot_model_comparison",
            "command": "python src/build_experiment_report.py",
            "data_source": "summary.csv",
            "generated_at": now_iso(),
            "sha256": sha256(path),
        }
    )

    # Single-metric bar with value labels
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    vals = pd.to_numeric(df["macro_f1_test_mean"], errors="coerce").fillna(0) * 100
    bars = ax.barh(models, vals, color="#2a9d8f")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("Macro F1 trên Test (%)")
    ax.set_title("Xếp hạng mô hình đơn giản theo Macro F1 (seed 42)")
    ax.set_xlim(0, max(100, float(vals.max()) + 8))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "model_comparison_macro_f1.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    records.append(
        {
            "group": "report_assets",
            "artifact": str(path),
            "kind": "figure",
            "description": "Xếp hạng mô hình đơn giản theo Macro F1 Test",
            "code_file": str(Path(__file__).resolve()),
            "code_function": "plot_model_comparison",
            "command": "python src/build_experiment_report.py",
            "data_source": "summary.csv",
            "generated_at": now_iso(),
            "sha256": sha256(path),
        }
    )
    return records


# ---------------------------------------------------------------------------
# Figure/table → code manifest
# ---------------------------------------------------------------------------
TABLE_SPECS = [
    ("metrics_train.json", "table", "Bộ độ đo đầy đủ trên Train", "evaluate_split"),
    ("metrics_dev.json", "table", "Bộ độ đo đầy đủ trên Dev", "evaluate_split"),
    ("metrics_test.json", "table", "Bộ độ đo đầy đủ trên Test", "evaluate_split"),
    ("per_class_train.csv", "table", "P/R/F1/support/AUC/AP từng lớp trên Train", "save_split_artifacts"),
    ("per_class_dev.csv", "table", "P/R/F1/support/AUC/AP từng lớp trên Dev", "save_split_artifacts"),
    ("per_class_test.csv", "table", "P/R/F1/support/AUC/AP từng lớp trên Test", "save_split_artifacts"),
    ("per_aspect_train.csv", "table", "Độ đo từng khía cạnh trên Train", "save_split_artifacts"),
    ("per_aspect_dev.csv", "table", "Độ đo từng khía cạnh trên Dev", "save_split_artifacts"),
    ("per_aspect_test.csv", "table", "Độ đo từng khía cạnh trên Test", "save_split_artifacts"),
    ("aspect_detection_train.csv", "table", "ACD, exact match, IoU, mAP, CMC@k trên Train", "save_split_artifacts"),
    ("aspect_detection_dev.csv", "table", "ACD, exact match, IoU, mAP, CMC@k trên Dev", "save_split_artifacts"),
    ("aspect_detection_test.csv", "table", "ACD, exact match, IoU, mAP, CMC@k trên Test", "save_split_artifacts"),
    ("confusion_matrix_train.csv", "table", "Confusion matrix 4×4 trên Train", "plot_confusion"),
    ("confusion_matrix_dev.csv", "table", "Confusion matrix 4×4 trên Dev", "plot_confusion"),
    ("confusion_matrix_test.csv", "table", "Confusion matrix 4×4 trên Test", "plot_confusion"),
    ("confusion_matrix_acd_train.csv", "table", "Confusion matrix ACD trên Train", "save_split_artifacts"),
    ("confusion_matrix_acd_dev.csv", "table", "Confusion matrix ACD trên Dev", "save_split_artifacts"),
    ("confusion_matrix_acd_test.csv", "table", "Confusion matrix ACD trên Test", "save_split_artifacts"),
    ("confusion_matrix_per_aspect_train.csv", "table", "10 confusion matrix theo khía cạnh, Train", "save_split_artifacts"),
    ("confusion_matrix_per_aspect_dev.csv", "table", "10 confusion matrix theo khía cạnh, Dev", "save_split_artifacts"),
    ("confusion_matrix_per_aspect_test.csv", "table", "10 confusion matrix theo khía cạnh, Test", "save_split_artifacts"),
    ("predictions_train.csv", "table", "Dự đoán từng cặp đánh giá × khía cạnh, Train", "save_split_artifacts"),
    ("predictions_dev.csv", "table", "Dự đoán từng cặp đánh giá × khía cạnh, Dev", "save_split_artifacts"),
    ("predictions_test.csv", "table", "Dự đoán từng cặp đánh giá × khía cạnh, Test", "save_split_artifacts"),
    ("probabilities_train.npz", "table", "Xác suất thô (N,10,4) trên Train", "save_split_artifacts"),
    ("probabilities_dev.npz", "table", "Xác suất thô (N,10,4) trên Dev", "save_split_artifacts"),
    ("probabilities_test.npz", "table", "Xác suất thô (N,10,4) trên Test", "save_split_artifacts"),
    ("train_log.csv", "table", "Loss/accuracy/macro-F1 từng epoch", "fit_simple_model"),
    ("epoch_log.json", "table", "Log epoch dạng JSON", "fit_simple_model"),
    ("config.json", "table", "Toàn bộ tham số dòng lệnh", "build_parser"),
    ("environment.json", "table", "Môi trường chạy (Python, thư viện, GPU, git)", "environment_info"),
    ("data_summary.json", "table", "Thống kê dữ liệu từng split", "split_summary"),
    ("RESULTS.txt", "table", "Báo cáo văn bản đầy đủ của lần chạy", "run_single"),
    ("run_record.json", "table", "Bản ghi một dòng để tổng hợp manifest", "run_single"),
    ("figure_manifest.csv", "table", "Danh mục hình của lần chạy", "register_figure"),
]


def collect_code_manifest(
    experiments_dir: Path, histograms_dir: Path, output_dir: Path
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    # 1. Dataset histogram figures (already carry their own manifest).
    hist_manifest = read_csv(histograms_dir / "figure_manifest.csv")
    if not hist_manifest.empty:
        for _, r in hist_manifest.iterrows():
            records.append(
                {
                    "group": "dataset_histograms",
                    "artifact": r.get("artifact"),
                    "kind": r.get("kind", "figure"),
                    "description": r.get("description"),
                    "code_file": r.get("code_file", str(HISTOGRAM_SCRIPT)),
                    "code_function": r.get("code_function", "main"),
                    "command": r.get("command"),
                    "data_source": r.get("data_source"),
                    "generated_at": r.get("generated_at"),
                    "sha256": r.get("sha256"),
                }
            )

    # 2. Per-run figures and tables.
    for run_dir in sorted(experiments_dir.glob("*/*/seed_*")):
        if not run_dir.is_dir():
            continue
        group = str(run_dir.relative_to(experiments_dir)).replace("\\", "/")
        fig_manifest = read_csv(run_dir / "figure_manifest.csv")
        if not fig_manifest.empty:
            for _, r in fig_manifest.iterrows():
                records.append(
                    {
                        "group": group,
                        "artifact": r.get("artifact"),
                        "kind": r.get("kind", "figure"),
                        "description": r.get("description"),
                        "code_file": r.get("code_file", str(RUNNER)),
                        "code_function": r.get("code_function", ""),
                        "command": r.get("command"),
                        "data_source": r.get("data_source"),
                        "generated_at": r.get("generated_at"),
                        "sha256": r.get("sha256"),
                    }
                )
        for filename, kind, description, function in TABLE_SPECS:
            path = run_dir / filename
            if not path.exists():
                continue
            data_source = filename
            if filename == "metrics_test.json":
                data_source = "run output (in-memory predictions)"
            records.append(
                {
                    "group": group,
                    "artifact": str(path),
                    "kind": kind,
                    "description": description,
                    "code_file": str(RUNNER),
                    "code_function": function,
                    "command": "see config.json / RESULTS.txt section 1",
                    "data_source": data_source,
                    "generated_at": now_iso(),
                    "sha256": sha256(path),
                }
            )
        model_file = run_dir / "model.joblib"
        if model_file.exists():
            records.append(
                {
                    "group": group,
                    "artifact": str(model_file),
                    "kind": "model",
                    "description": "Vectorizer + 10 head đã huấn luyện (joblib)",
                    "code_file": str(RUNNER),
                    "code_function": "run_single",
                    "command": "see config.json",
                    "data_source": "TF-IDF + heads",
                    "generated_at": now_iso(),
                    "sha256": sha256(model_file),
                }
            )

    # 3. Global aggregate tables.
    globals_spec = [
        ("manifest.csv", "table", "Mọi lần chạy, một dòng / run", "rebuild_manifest", "run_record.json"),
        ("manifest.json", "table", "Mọi lần chạy, JSON", "rebuild_manifest", "run_record.json"),
        ("summary.csv", "table", "Tổng hợp mean ± std theo mô hình × tag", "build_summary", "manifest.csv"),
        ("summary.json", "table", "Tổng hợp JSON", "build_summary", "manifest.csv"),
        ("summary.md", "table", "Bảng tổng hợp Markdown", "build_summary", "manifest.csv"),
    ]
    for filename, kind, description, function, source in globals_spec:
        path = experiments_dir / filename
        if path.exists():
            records.append(
                {
                    "group": "global",
                    "artifact": str(path),
                    "kind": kind,
                    "description": description,
                    "code_file": str(RUNNER),
                    "code_function": function,
                    "command": "python src/run_simple_experiments.py --models ... ",
                    "data_source": source,
                    "generated_at": now_iso(),
                    "sha256": sha256(path),
                }
            )

    report_spec = [
        ("comparison_with_related_studies.csv", "table", "Bảng so sánh với tài liệu tham khảo", "build_comparison"),
        ("comparison_with_related_studies.md", "table", "Bảng so sánh dạng Markdown", "comparison_markdown"),
        ("metric_applicability.csv", "table", "Độ đo nào áp dụng / không áp dụng cho ABSA", "build_metric_applicability"),
        ("figure_table_code_manifest.csv", "table", "Chính danh mục này", "collect_code_manifest"),
    ]
    for filename, kind, description, function in report_spec:
        path = output_dir / filename
        records.append(
            {
                "group": "report_assets",
                "artifact": str(path),
                "kind": kind,
                "description": description,
                "code_file": str(Path(__file__).resolve()),
                "code_function": function,
                "command": "python src/build_experiment_report.py",
                "data_source": "manifest.csv + data/literature_baselines.csv",
                "generated_at": now_iso(),
                "sha256": sha256(path),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tổng hợp kết quả CLI thành bảng so sánh, bảng độ đo và manifest hình/bảng.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python build_experiment_report.py\n"
            "  python build_experiment_report.py --experiments results/cli_experiments\n"
        ),
    )
    parser.add_argument("--experiments", default=str(DEFAULT_EXPERIMENTS))
    parser.add_argument("--histograms", default=str(DEFAULT_HISTOGRAMS))
    parser.add_argument("--literature", default=str(DEFAULT_LITERATURE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    experiments_dir = Path(args.experiments).resolve()
    histograms_dir = Path(args.histograms).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    literature = read_csv(Path(args.literature).resolve())
    runs = load_group_runs(experiments_dir)

    comparison = build_comparison(runs, literature)
    if not comparison.empty:
        comparison.to_csv(
            output_dir / "comparison_with_related_studies.csv",
            index=False, encoding="utf-8-sig",
        )
    (output_dir / "comparison_with_related_studies.md").write_text(
        comparison_markdown(comparison), encoding="utf-8"
    )

    applicability = build_metric_applicability()
    applicability.to_csv(
        output_dir / "metric_applicability.csv", index=False, encoding="utf-8-sig"
    )

    comparison_figures = plot_model_comparison(experiments_dir, output_dir)

    manifest = collect_code_manifest(experiments_dir, histograms_dir, output_dir)
    if comparison_figures:
        manifest = pd.concat(
            [pd.DataFrame(comparison_figures), manifest], ignore_index=True
        )
    manifest.to_csv(
        output_dir / "figure_table_code_manifest.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Literature rows : {len(literature)}")
    print(f"Reproduced runs : {len(runs)}")
    print(f"Comparison rows : {len(comparison)}")
    print(f"Metric rows     : {len(applicability)}")
    print(f"Figure/table    : {len(manifest)}")
    print(f"Output          : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
