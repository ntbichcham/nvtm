"""
analyze_dataset_histograms.py
=============================
Histogram / distribution analysis for the UIT-ViSFD data used by the ABSA
experiments.  Every figure is written as a PNG **and** as a CSV with the exact
numbers behind the plot, so a report can link each figure to data and code.

Figures produced
----------------
  split_size_bar.png                  rows per split
  aspect_mention_counts_by_split.png  mentions per aspect, grouped by split
  aspect_mention_rate_heatmap.png     aspect x split mention rate (%)
  sentiment_distribution_by_split.png label counts (None/Negative/Neutral/Positive)
  label_per_aspect_heatmap_train.png  aspect x label count on Train
  sentiment_per_aspect_train.png      stacked label distribution per aspect
  aspect_count_per_review_hist.png    number of mentioned aspects per review
  token_length_hist.png               token-length histogram by split
  character_length_hist.png           character-length histogram by split
  aspect_cooccurrence_heatmap_train.png  aspect co-occurrence (Jaccard)
  monthly_volume.png                  reviews per month (if dates parse)

Usage
-----
python analyze_dataset_histograms.py
python analyze_dataset_histograms.py --output-dir results/dataset_histograms --dpi 180

Author: Nhom 2 - CT255
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset_loader import ASPECT_LIST, parse_label_string  # noqa: E402

LABEL_NAMES = ["None", "Negative", "Neutral", "Positive"]
COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "UIT-ViSFD" / "prepared_for_model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_histograms"

FIGURES: List[Dict[str, Any]] = []


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def finish(fig, path: Path, description: str, function: str, data_source: str, command: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(
        {
            "artifact": str(path),
            "kind": "figure",
            "description": description,
            "code_file": str(Path(__file__).resolve()),
            "code_function": function,
            "command": command,
            "data_source": data_source,
            "generated_at": now_iso(),
            "sha256": sha256(path),
        }
    )


def load_split(path: Path, text_col: str, label_col: str, limit: int = 0) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit).copy()
    labels = np.stack([parse_label_string(x) for x in df[label_col].tolist()])
    return df, labels


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vẽ histogram/phân tích dữ liệu UIT-ViSFD và lưu PNG + CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python analyze_dataset_histograms.py\n"
            "  python analyze_dataset_histograms.py --output-dir results/dataset_histograms --dpi 180\n"
        ),
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--train-file", default="Train_prepared.csv")
    parser.add_argument("--dev-file", default="Dev_prepared.csv")
    parser.add_argument("--test-file", default="Test_prepared.csv")
    parser.add_argument("--text-col", default="comment_segmented")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số dòng mỗi split (0 = tất cả).")
    parser.add_argument("--formats", nargs="+", default=["png", "csv"], choices=["png", "csv"])
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])

    splits: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}
    for name, fname in (("train", args.train_file), ("dev", args.dev_file), ("test", args.test_file)):
        path = data_dir / fname
        if not path.exists():
            print(f"[warn] missing {name}: {path}")
            continue
        splits[name] = load_split(path, args.text_col, args.label_col, args.limit)
        print(f"Loaded {name}: {len(splits[name][0])} rows")

    if not splits:
        raise SystemExit("No data loaded.")

    split_names = list(splits.keys())

    # ------------------------------------------------------------------
    # 1. Split sizes
    # ------------------------------------------------------------------
    sizes = [len(splits[s][0]) for s in split_names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(split_names, sizes, color=["#2a9d8f", "#e9c46a", "#457b9d"][: len(split_names)])
    for i, v in enumerate(sizes):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of reviews")
    ax.set_title("Dataset split sizes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finish(fig, output_dir / "split_size_bar.png", "Số đánh giá mỗi split", "main", "Train/Dev/Test CSV", command, args.dpi)
    write_csv(output_dir / "split_size_bar.csv", ["split", "rows"], list(zip(split_names, sizes)))

    # ------------------------------------------------------------------
    # 2. Aspect mention counts by split
    # ------------------------------------------------------------------
    rows = []
    for s in split_names:
        labels = splits[s][1]
        for i, aspect in enumerate(ASPECT_LIST):
            rows.append((aspect, s, int(np.sum(labels[:, i] > 0)), float(np.mean(labels[:, i] > 0) * 100)))
    df_aspect = pd.DataFrame(rows, columns=["aspect", "split", "count", "percent"])
    pivot = df_aspect.pivot(index="aspect", columns="split", values="count").reindex(ASPECT_LIST)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(ASPECT_LIST))
    width = 0.8 / max(1, len(split_names))
    for j, s in enumerate(split_names):
        ax.bar(x + (j - (len(split_names) - 1) / 2) * width, pivot[s].values, width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(ASPECT_LIST, rotation=35, ha="right")
    ax.set_ylabel("Mentions (label != None)")
    ax.set_title("Aspect mention counts by split")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finish(fig, output_dir / "aspect_mention_counts_by_split.png", "Số lượt nhắc từng khía cạnh theo split", "main", "aspect_mention_counts_by_split.csv", command, args.dpi)
    write_csv(output_dir / "aspect_mention_counts_by_split.csv", ["aspect", "split", "count", "percent"], rows)

    # ------------------------------------------------------------------
    # 3. Aspect mention rate heatmap (aspect x split)
    # ------------------------------------------------------------------
    rate = df_aspect.pivot(index="aspect", columns="split", values="percent").reindex(ASPECT_LIST)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rate.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(split_names)), split_names)
    ax.set_yticks(np.arange(len(ASPECT_LIST)), ASPECT_LIST)
    ax.set_title("Aspect mention rate (% of reviews) by split")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            ax.text(j, i, f"{rate.values[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")
    fig.tight_layout()
    finish(fig, output_dir / "aspect_mention_rate_heatmap.png", "Tỉ lệ đánh giá nhắc từng khía cạnh theo split", "main", "aspect_mention_counts_by_split.csv", command, args.dpi)
    write_csv(
        output_dir / "aspect_mention_rate_heatmap.csv",
        ["aspect", *split_names],
        [[a, *[f"{rate.loc[a, s]:.4f}" for s in split_names]] for a in ASPECT_LIST],
    )

    # ------------------------------------------------------------------
    # 4. Sentiment distribution by split
    # ------------------------------------------------------------------
    rows = []
    for s in split_names:
        labels = splits[s][1]
        for k, name in enumerate(LABEL_NAMES):
            count = int(np.sum(labels == k))
            rows.append((s, name, count, count / max(1, labels.size) * 100))
    df_sent = pd.DataFrame(rows, columns=["split", "label", "count", "percent"])
    pivot = df_sent.pivot(index="label", columns="split", values="count").reindex(LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(LABEL_NAMES))
    width = 0.8 / max(1, len(split_names))
    for j, s in enumerate(split_names):
        ax.bar(x + (j - (len(split_names) - 1) / 2) * width, pivot[s].values, width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylabel("Count (review x aspect pairs)")
    ax.set_title("Sentiment label distribution by split")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finish(fig, output_dir / "sentiment_distribution_by_split.png", "Phân bố 4 nhãn cảm xúc theo split", "main", "sentiment_distribution_by_split.csv", command, args.dpi)
    write_csv(output_dir / "sentiment_distribution_by_split.csv", ["split", "label", "count", "percent"], rows)

    # ------------------------------------------------------------------
    # 5. Label distribution per aspect on Train
    # ------------------------------------------------------------------
    train_labels = splits[split_names[0]][1]
    matrix = np.zeros((len(ASPECT_LIST), 4), dtype=int)
    for i in range(len(ASPECT_LIST)):
        for k in range(4):
            matrix[i, k] = int(np.sum(train_labels[:, i] == k))
    fig, ax = plt.subplots(figsize=(8.4, 6))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(4), LABEL_NAMES)
    ax.set_yticks(np.arange(len(ASPECT_LIST)), ASPECT_LIST)
    ax.set_title(f"Label distribution per aspect — {split_names[0]}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    finish(fig, output_dir / "label_per_aspect_heatmap_train.png", "Số lượng từng nhãn cho từng khía cạnh trên Train", "main", "label_per_aspect_heatmap_train.csv", command, args.dpi)
    write_csv(output_dir / "label_per_aspect_heatmap_train.csv", ["aspect", *LABEL_NAMES], [[a, *[int(v) for v in matrix[i]]] for i, a in enumerate(ASPECT_LIST)])

    # ------------------------------------------------------------------
    # 6. Sentiment per aspect, stacked (Train)
    # ------------------------------------------------------------------
    percentages = matrix / np.maximum(1, matrix.sum(axis=1, keepdims=True)) * 100
    fig, ax = plt.subplots(figsize=(12, 5.2))
    bottom = np.zeros(len(ASPECT_LIST))
    for k, name in enumerate(LABEL_NAMES):
        ax.bar(ASPECT_LIST, percentages[:, k], bottom=bottom, label=name, color=COLORS[k])
        bottom += percentages[:, k]
    ax.set_ylabel("Percentage of reviews mentioning the aspect (%)")
    ax.set_title(f"Sentiment distribution per aspect — {split_names[0]}")
    ax.legend(ncol=4, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finish(fig, output_dir / "sentiment_per_aspect_train.png", "Tỉ lệ 4 nhãn trong từng khía cạnh trên Train", "main", "label_per_aspect_heatmap_train.csv", command, args.dpi)
    write_csv(output_dir / "sentiment_per_aspect_train.csv", ["aspect", *LABEL_NAMES], [[a, *[f"{percentages[i, k]:.4f}" for k in range(4)]] for i, a in enumerate(ASPECT_LIST)])

    # ------------------------------------------------------------------
    # 7. Aspect count per review
    # ------------------------------------------------------------------
    rows = []
    max_k = len(ASPECT_LIST)
    for s in split_names:
        counts = np.sum(splits[s][1] > 0, axis=1)
        hist = np.bincount(counts, minlength=max_k + 1)
        for k in range(max_k + 1):
            rows.append((s, k, int(hist[k]), hist[k] / max(1, len(counts)) * 100))
    df_counts = pd.DataFrame(rows, columns=["split", "n_aspects", "count", "percent"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / max(1, len(split_names))
    for j, s in enumerate(split_names):
        sub = df_counts[df_counts.split == s]
        ax.bar(sub["n_aspects"] + (j - (len(split_names) - 1) / 2) * width, sub["count"], width, label=s)
    ax.set_xlabel("Number of mentioned aspects in a review")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Aspect count per review")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finish(fig, output_dir / "aspect_count_per_review_hist.png", "Histogram số khía cạnh được nhắc trong một đánh giá", "main", "aspect_count_per_review_hist.csv", command, args.dpi)
    write_csv(output_dir / "aspect_count_per_review_hist.csv", ["split", "n_aspects", "count", "percent"], rows)

    # ------------------------------------------------------------------
    # 8. Token length histogram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    length_rows = []
    for s in split_names:
        df, _ = splits[s]
        if "token_length" in df.columns:
            values = pd.to_numeric(df["token_length"], errors="coerce").dropna().values
        else:
            values = df[args.text_col].fillna("").astype(str).str.split().str.len().values
        ax.hist(values, bins=40, alpha=0.55, label=f"{s} (n={len(values)}, median={np.median(values):.0f})")
        for q in (0.25, 0.5, 0.75):
            length_rows.append((s, q, float(np.quantile(values, q))))
    ax.set_xlabel("Token length (word-segmented text)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Token-length distribution by split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finish(fig, output_dir / "token_length_hist.png", "Histogram độ dài token theo split", "main", "token_length_hist.csv", command, args.dpi)
    write_csv(output_dir / "token_length_hist.csv", ["split", "quantile", "value"], length_rows)

    # ------------------------------------------------------------------
    # 9. Character length histogram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for s in split_names:
        df, _ = splits[s]
        values = df[args.text_col].fillna("").astype(str).str.len().values
        ax.hist(values, bins=40, alpha=0.55, label=f"{s} (median={np.median(values):.0f})")
    ax.set_xlabel("Character length")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Character-length distribution by split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    finish(fig, output_dir / "character_length_hist.png", "Histogram độ dài ký tự theo split", "main", "character_length_hist.csv", command, args.dpi)
    rows = []
    for s in split_names:
        df, _ = splits[s]
        values = df[args.text_col].fillna("").astype(str).str.len().values
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows.append((s, q, float(np.quantile(values, q))))
    write_csv(output_dir / "character_length_hist.csv", ["split", "quantile", "value"], rows)

    # ------------------------------------------------------------------
    # 10. Aspect co-occurrence (Jaccard) on Train
    # ------------------------------------------------------------------
    bin_labels = (train_labels > 0).astype(int)
    counts = bin_labels.T @ bin_labels
    diag = np.diag(counts).astype(float)
    jaccard = np.zeros_like(counts, dtype=float)
    for i in range(len(ASPECT_LIST)):
        for j in range(len(ASPECT_LIST)):
            union = diag[i] + diag[j] - counts[i, j]
            jaccard[i, j] = 0.0 if union == 0 else counts[i, j] / union
    fig, ax = plt.subplots(figsize=(8.6, 7))
    im = ax.imshow(jaccard, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(ASPECT_LIST)), ASPECT_LIST, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ASPECT_LIST)), ASPECT_LIST)
    ax.set_title(f"Aspect co-occurrence — Jaccard ({split_names[0]})")
    for i in range(len(ASPECT_LIST)):
        for j in range(len(ASPECT_LIST)):
            ax.text(j, i, f"{jaccard[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    fig.tight_layout()
    finish(fig, output_dir / "aspect_cooccurrence_heatmap_train.png", "Ma trận đồng xuất hiện khía cạnh (Jaccard) trên Train", "main", "aspect_cooccurrence_jaccard_train.csv", command, args.dpi)
    write_csv(
        output_dir / "aspect_cooccurrence_jaccard_train.csv",
        ["aspect", *ASPECT_LIST],
        [[a, *[f"{jaccard[i, j]:.4f}" for j in range(len(ASPECT_LIST))]] for i, a in enumerate(ASPECT_LIST)],
    )
    write_csv(
        output_dir / "aspect_cooccurrence_count_train.csv",
        ["aspect", *ASPECT_LIST],
        [[a, *[int(counts[i, j]) for j in range(len(ASPECT_LIST))]] for i, a in enumerate(ASPECT_LIST)],
    )

    # ------------------------------------------------------------------
    # 11. Monthly volume (if dates are available)
    # ------------------------------------------------------------------
    all_dates = []
    for s in split_names:
        df, _ = splits[s]
        series = None
        if "date_parsed" in df.columns:
            series = pd.to_datetime(df["date_parsed"], errors="coerce")
        if series is None or series.notna().sum() == 0:
            if "date_time" in df.columns:
                series = pd.to_datetime(df["date_time"], errors="coerce", dayfirst=True)
        if series is not None and series.notna().sum() > 0:
            all_dates.append(pd.DataFrame({"split": s, "date": series.dropna()}))
    if all_dates:
        dates = pd.concat(all_dates, ignore_index=True)
        dates["month"] = dates["date"].dt.to_period("M").astype(str)
        monthly = dates.groupby(["month", "split"]).size().reset_index(name="count")
        wide = monthly.pivot(index="month", columns="split", values="count").fillna(0).sort_index()
        fig, ax = plt.subplots(figsize=(12, 4.8))
        for s in wide.columns:
            ax.plot(wide.index, wide[s], marker="o", label=s)
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of reviews")
        ax.set_title("Review volume by month and split")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        finish(fig, output_dir / "monthly_volume.png", "Số đánh giá theo tháng và split", "main", "monthly_volume.csv", command, args.dpi)
        write_csv(output_dir / "monthly_volume.csv", ["month", *wide.columns], [[m, *[int(wide.loc[m, s]) for s in wide.columns]] for m in wide.index])
    else:
        print("[warn] no parseable date column; monthly_volume.png skipped")

    # ------------------------------------------------------------------
    # 12. Text report + manifests
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 90)
    lines.append("PHÂN TÍCH DỮ LIỆU UIT-ViSFD — HISTOGRAM / DISTRIBUTION")
    lines.append("=" * 90)
    lines.append(f"Thời điểm: {now_iso()}")
    lines.append(f"Lệnh: {command}")
    lines.append(f"Thư mục dữ liệu: {data_dir}")
    lines.append(f"Thư mục kết quả: {output_dir}")
    lines.append("")
    lines.append("Cột văn bản: " + args.text_col)
    lines.append("Cột nhãn: " + args.label_col)
    lines.append("Thứ tự khía cạnh cố định: " + ", ".join(ASPECT_LIST))
    lines.append("Thứ tự nhãn cố định: " + ", ".join(LABEL_NAMES))
    lines.append("")
    for s in split_names:
        df, labels = splits[s]
        lines.append(f"[{s}] rows={len(df)}")
        lines.append("  label counts: " + ", ".join(f"{n}={int(np.sum(labels == k))}" for k, n in enumerate(LABEL_NAMES)))
        lines.append("  aspect mentions: " + ", ".join(f"{a}={int(np.sum(labels[:, i] > 0))}" for i, a in enumerate(ASPECT_LIST)))
        counts = np.sum(labels > 0, axis=1)
        lines.append(f"  aspects/review: mean={counts.mean():.3f}, median={np.median(counts):.0f}, max={counts.max()}")
        if "token_length" in df.columns:
            vals = pd.to_numeric(df["token_length"], errors="coerce").dropna()
            lines.append(f"  token length: mean={vals.mean():.2f}, median={vals.median():.0f}, max={vals.max():.0f}")
        lines.append("")
    lines.append("DANH MỤC HÌNH VÀ CSV:")
    for rec in FIGURES:
        lines.append(f"  - {Path(rec['artifact']).name:<42} -> {rec['code_function']}()")
        lines.append(f"      code: {rec['code_file']}")
        lines.append(f"      data: {rec['data_source']}")
        lines.append(f"      sha256: {rec['sha256']}")
    lines.append("")
    lines.append("GHI CHÚ:")
    lines.append("  * Mỗi hình đều có file CSV cùng tên chứa số liệu gốc.")
    lines.append("  * Histogram/heatmap chỉ mô tả dữ liệu, không phải kết quả mô hình.")
    lines.append("  * Các độ đo ẩn tin (PSNR/SSIM/BER) không áp dụng cho dữ liệu văn bản ABSA.")
    (output_dir / "DATASET_ANALYSIS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(
        output_dir / "figure_manifest.csv",
        ["artifact", "kind", "description", "code_file", "code_function", "command", "data_source", "generated_at", "sha256"],
        [
            [r["artifact"], r["kind"], r["description"], r["code_file"], r["code_function"],
             r["command"], r["data_source"], r["generated_at"], r["sha256"]]
            for r in FIGURES
        ],
    )
    write_json(output_dir / "dataset_summary.json", {
        "generated_at": now_iso(),
        "command": command,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "splits": {
            s: {
                "rows": int(len(splits[s][0])),
                "label_counts": {n: int(np.sum(splits[s][1] == k)) for k, n in enumerate(LABEL_NAMES)},
                "aspect_mentions": {a: int(np.sum(splits[s][1][:, i] > 0)) for i, a in enumerate(ASPECT_LIST)},
            }
            for s in split_names
        },
        "figures": FIGURES,
    })

    print(f"\nSaved {len(FIGURES)} figures to {output_dir}")
    print(f"Report: {output_dir / 'DATASET_ANALYSIS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
