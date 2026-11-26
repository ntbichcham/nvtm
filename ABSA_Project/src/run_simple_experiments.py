"""
run_simple_experiments.py
=========================
Command-line experiment runner for the UIT-ViSFD ABSA task.

It trains classical / simple models on the exact same Train/Dev/Test splits,
aspect order and 4-class label order used by the PhoBERT checkpoint, evaluates
on Train and Test (Dev optional), and writes a complete, reproducible record:

  RESULTS.txt              human-readable report (command, data, hyper-params,
                           architecture, metrics, confusion matrices, curves,
                           per-aspect tables, artifact paths, code links)
  config.json              every CLI argument
  environment.json         Python / library / OS / GPU / git commit
  metrics_train.json       full metric bundle on Train
  metrics_dev.json         full metric bundle on Dev
  metrics_test.json        full metric bundle on Test
  per_class_<split>.csv    P/R/F1/support/AUC/AP per sentiment class
  per_aspect_<split>.csv   metrics for each of the 10 aspect heads
  aspect_detection_<split>.csv  ACD, exact match, IoU, mAP, CMC@k, mAP@k
  confusion_matrix_<split>.csv  global 4x4 matrix
  confusion_matrix_acd_<split>.csv  None-vs-mention 2x2 matrix
  confusion_matrix_per_aspect_<split>.csv  long format, 10 matrices
  predictions_<split>.csv  long format, one row per review x aspect
  probabilities_<split>.npz  raw (N, 10, 4) probabilities
  train_log.csv            epoch, train/dev loss, accuracy, macro-F1
  figures/*.png             confusion matrix, per-aspect F1, ROC, PR,
                            calibration, CMC, IoU histogram, confidence hist
  figure_manifest.csv      every figure/table -> code function + command
  run_record.json          one-row summary used to rebuild the global manifest

Usage
-----
python run_simple_experiments.py --list
python run_simple_experiments.py --models majority logistic_regression --seeds 42
python run_simple_experiments.py --models default --seeds 42 123 2024
python run_simple_experiments.py --models all --tag full_baselines --force
python run_simple_experiments.py --help

Author: Nhom 2 - CT255
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Make sibling modules importable when the script is launched by absolute path.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_curve,
    auc,
)

from dataset_loader import ASPECT_LIST, parse_label_string  # noqa: E402
from experiment_metrics import (  # noqa: E402
    LABELS,
    LABEL_NAMES,
    build_confusion_matrix_rows,
    classification_metrics,
    multilabel_aspect_metrics,
    per_aspect_metrics,
)
from simple_model_zoo import (  # noqa: E402
    ALL_MODELS,
    DEFAULT_MODELS,
    EPOCH_MODELS,
    MODEL_REGISTRY,
    MultiHeadModel,
    build_vectorizer,
    describe_models,
    fit_simple_model,
    vectorizer_description,
)

VERSION = "1.0.0"
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "UIT-ViSFD" / "prepared_for_model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cli_experiments"
FIGURE_DPI = 150


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        if abs(value) <= 1.0 and digits >= 4:
            return f"{value:.4f} ({value * 100:.2f}%)"
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Sequence[Any]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def write_csv_dicts(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else json_safe(v)) for k, v in row.items()})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> Dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": run("describe", "--tags", "--always", "--dirty"),
        "status_short": run("status", "--short"),
    }


def environment_info() -> Dict[str, Any]:
    import sklearn
    import scipy
    import matplotlib as mpl

    info: Dict[str, Any] = {
        "timestamp": now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": mpl.__version__,
        "git": git_info(),
        "command": " ".join([sys.executable, *sys.argv]),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
        info["cuda_available"] = False
    return info


def resolve_models(raw: Sequence[str]) -> List[str]:
    models: List[str] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            if part == "default":
                models.extend(DEFAULT_MODELS)
            elif part == "all":
                models.extend(ALL_MODELS)
            else:
                models.append(part)
    # stable unique order
    seen = set()
    out = []
    for m in models:
        if m not in seen:
            seen.add(m)
            out.append(m)
    unknown = [m for m in out if m not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown model(s): {', '.join(unknown)}\nAvailable:\n{describe_models()}"
        )
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read_split(
    path: Path, text_col: str, label_col: str, limit: int = 0
) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    if limit and limit > 0:
        df = df.head(limit).copy()
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not in {path.name}: {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not in {path.name}: {list(df.columns)}")
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = np.stack([parse_label_string(x) for x in df[label_col].tolist()]) if len(df) else np.zeros((0, len(ASPECT_LIST)), dtype=int)
    return df, texts, labels


def split_summary(name: str, df: pd.DataFrame, labels: np.ndarray, text_col: str, path: Path) -> Dict[str, Any]:
    label_counts = {
        label_name: int(np.sum(labels == idx)) for idx, label_name in enumerate(LABEL_NAMES)
    }
    aspect_mentions = {
        aspect: int(np.sum(labels[:, i] > 0)) for i, aspect in enumerate(ASPECT_LIST)
    }
    aspect_counts_per_review = np.sum(labels > 0, axis=1)
    hist, edges = np.histogram(aspect_counts_per_review, bins=np.arange(-0.5, len(ASPECT_LIST) + 1.5, 1))
    length_values = None
    if "token_length" in df.columns:
        length_values = pd.to_numeric(df["token_length"], errors="coerce").dropna()
    elif text_col in df.columns:
        length_values = df[text_col].fillna("").astype(str).str.split().str.len()
    length_stats = None
    if length_values is not None and len(length_values) > 0:
        length_stats = {
            "count": int(length_values.shape[0]),
            "mean": float(length_values.mean()),
            "std": float(length_values.std(ddof=0)),
            "min": float(length_values.min()),
            "p25": float(length_values.quantile(0.25)),
            "median": float(length_values.median()),
            "p75": float(length_values.quantile(0.75)),
            "max": float(length_values.max()),
        }
    return {
        "split": name,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "text_column": text_col,
        "label_column": "label",
        "empty_text_rows": int(sum(1 for t in df[text_col].fillna("").astype(str) if not t.strip())),
        "label_counts": label_counts,
        "label_percent": {
            k: (v / max(1, labels.size) * 100.0) for k, v in label_counts.items()
        },
        "aspect_mentions": aspect_mentions,
        "aspect_mention_percent": {
            k: (v / max(1, len(df)) * 100.0) for k, v in aspect_mentions.items()
        },
        "aspect_count_per_review": {
            "mean": float(aspect_counts_per_review.mean()) if len(df) else 0.0,
            "min": int(aspect_counts_per_review.min()) if len(df) else 0,
            "max": int(aspect_counts_per_review.max()) if len(df) else 0,
            "histogram": {str(int(edges[i])): int(hist[i]) for i in range(len(hist))},
        },
        "token_length": length_stats,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIGURE_RECORDS: List[Dict[str, Any]] = []


def register_figure(
    path: Path,
    description: str,
    code_function: str,
    data_source: str,
    command: str,
    kind: str = "figure",
) -> None:
    FIGURE_RECORDS.append(
        {
            "artifact": str(path),
            "kind": kind,
            "description": description,
            "code_file": str(Path(__file__).resolve()),
            "code_function": code_function,
            "command": command,
            "data_source": data_source,
            "generated_at": now_iso(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    )


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: Sequence[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    _finish(fig, path)


def plot_bar(
    values: Sequence[float],
    labels: Sequence[str],
    title: str,
    ylabel: str,
    path: Path,
    color: str = "#2a9d8f",
    horizontal: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2) if not horizontal else (8, 6))
    y = np.asarray(values, dtype=float)
    if horizontal:
        ax.barh(labels, y, color=color)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
    else:
        ax.bar(labels, y, color=color)
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.25)
    fig.tight_layout()
    _finish(fig, path)


def plot_learning_curves(log: Sequence[Dict[str, Any]], path: Path) -> None:
    epochs = [r["epoch"] for r in log]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(epochs, [r["train_loss"] for r in log], marker="o", label="Train")
    if "dev_loss" in log[0]:
        axes[0].plot(epochs, [r["dev_loss"] for r in log], marker="s", label="Dev")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Log loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [r["train_accuracy"] for r in log], marker="o", label="Train")
    if "dev_accuracy" in log[0]:
        axes[1].plot(epochs, [r["dev_accuracy"] for r in log], marker="s", label="Dev")
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, [r["train_macro_f1"] for r in log], marker="o", label="Train")
    if "dev_macro_f1" in log[0]:
        axes[2].plot(epochs, [r["dev_macro_f1"] for r in log], marker="s", label="Dev")
    axes[2].set_title("Macro F1 per epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    _finish(fig, path)


def plot_roc(y_true: np.ndarray, proba: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for i, name in enumerate(LABEL_NAMES):
        y_bin = (y_true == i).astype(int)
        if y_bin.min() == y_bin.max():
            continue
        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest, Test)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _finish(fig, path)


def plot_pr(y_true: np.ndarray, proba: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for i, name in enumerate(LABEL_NAMES):
        y_bin = (y_true == i).astype(int)
        if y_bin.min() == y_bin.max():
            continue
        precision, recall, _ = precision_recall_curve(y_bin, proba[:, i])
        ap = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (one-vs-rest, Test)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _finish(fig, path)


def plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, path: Path) -> None:
    conf = np.max(proba, axis=1)
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    centers, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(correct[mask].mean())
        counts.append(mask.sum())
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(centers, accs, marker="o", label="Model")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram (Test)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _finish(fig, path)


def plot_cmc(cmc: Dict[str, Dict[str, Optional[float]]], path: Path) -> None:
    ks = sorted(int(k[1:]) for k in cmc["recall_at_k"].keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(ks, [cmc["recall_at_k"][f"@{k}"] for k in ks], marker="o", label="Recall@k (all true aspects)")
    ax.plot(ks, [cmc["hit_at_k"][f"@{k}"] for k in ks], marker="s", label="Hit@k (>=1 true aspect)")
    ax.plot(ks, [cmc["all_aspects_at_k"][f"@{k}"] for k in ks], marker="^", label="All-aspects@k")
    ax.set_xlabel("k (number of top-ranked aspects)")
    ax.set_ylabel("Score")
    ax.set_title("CMC / cumulative aspect-detection curve (Test)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _finish(fig, path)


def plot_hist(values: Sequence[float], title: str, xlabel: str, path: Path, bins: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(values, bins=bins, color="#457b9d", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of reviews")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _finish(fig, path)


def plot_confidence_hist(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, path: Path
) -> None:
    conf = np.max(proba, axis=1)
    correct = y_true == y_pred
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf[correct], bins=bins, alpha=0.7, label="Correct", color="#2a9d8f")
    ax.hist(conf[~correct], bins=bins, alpha=0.7, label="Incorrect", color="#e76f51")
    ax.set_xlabel("Maximum predicted probability")
    ax.set_ylabel("Number of review-aspect pairs")
    ax.set_title("Confidence distribution (Test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _finish(fig, path)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def line(self, text: str = "") -> None:
        self.lines.append(text)

    def rule(self, char: str = "=", width: int = 100) -> None:
        self.line(char * width)

    def heading(self, text: str) -> None:
        self.line()
        self.rule()
        self.line(text)
        self.rule()

    def kv(self, key: str, value: Any, width: int = 34) -> None:
        self.line(f"  {key:<{width}}: {value}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def metrics_table_lines(metrics: Dict[str, Any]) -> List[str]:
    out = []
    keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("micro_precision", "Precision (micro)"),
        ("micro_recall", "Recall (micro)"),
        ("micro_f1", "F1 (micro)"),
        ("macro_precision", "Precision (macro)"),
        ("macro_recall", "Recall (macro)"),
        ("macro_f1", "F1 (macro)"),
        ("weighted_precision", "Precision (weighted)"),
        ("weighted_recall", "Recall (weighted)"),
        ("weighted_f1", "F1 (weighted)"),
        ("cohen_kappa", "Cohen's kappa (linear)"),
        ("cohen_kappa_quadratic", "Cohen's kappa (quadratic)"),
        ("mcc", "Matthews correlation coefficient"),
        ("roc_auc_ovr_macro", "ROC-AUC OVR (macro)"),
        ("roc_auc_ovr_micro", "ROC-AUC OVR (micro)"),
        ("roc_auc_ovr_weighted", "ROC-AUC OVR (weighted)"),
        ("pr_auc_macro", "PR-AUC / AP (macro)"),
        ("pr_auc_micro", "PR-AUC / AP (micro)"),
        ("log_loss", "Log loss"),
        ("brier_score_multiclass", "Brier score (multi-class)"),
        ("ece_10bin", "Expected calibration error (10 bins)"),
        ("top2_accuracy", "Top-2 accuracy"),
        ("top3_accuracy", "Top-3 accuracy"),
        ("sentiment_macro_precision", "ASC precision (macro, labels 1..3)"),
        ("sentiment_macro_recall", "ASC recall (macro, labels 1..3)"),
        ("sentiment_macro_f1", "ASC F1 (macro, labels 1..3)"),
        ("acd_precision", "ACD precision (mention vs None)"),
        ("acd_recall", "ACD recall (mention vs None)"),
        ("acd_f1", "ACD F1 (mention vs None)"),
    ]
    for key, label in keys:
        if key in metrics:
            out.append(f"  {label:<42}: {fmt(metrics[key])}")
    return out


def aspect_table_lines(per_aspect: Dict[str, Dict[str, Any]]) -> List[str]:
    header = (
        f"  {'Aspect':<12} {'N':>6} {'Acc':>8} {'MacroP':>8} {'MacroR':>8} "
        f"{'MacroF1':>8} {'WeightF1':>9} {'Kappa':>8} {'MCC':>8} {'AUC':>8} {'ACD-F1':>8}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for aspect, m in per_aspect.items():
        lines.append(
            f"  {aspect:<12} {m.get('n_samples', 0):>6} "
            f"{fmt(m.get('accuracy')):>8} {fmt(m.get('macro_precision')):>8} "
            f"{fmt(m.get('macro_recall')):>8} {fmt(m.get('macro_f1')):>8} "
            f"{fmt(m.get('weighted_f1')):>9} {fmt(m.get('cohen_kappa')):>8} "
            f"{fmt(m.get('mcc')):>8} {fmt(m.get('roc_auc_ovr_macro')):>8} "
            f"{fmt(m.get('acd_f1')):>8}"
        )
    return lines


def detection_table_lines(det: Dict[str, Any]) -> List[str]:
    keys = [
        ("exact_match", "Exact match (all 10 aspects correct)"),
        ("hamming_loss_4class", "Hamming loss (4-class)"),
        ("hamming_loss_mention", "Hamming loss (mention)"),
        ("subset_accuracy", "Subset accuracy (aspect set)"),
        ("mention_micro_precision", "Mention precision (micro)"),
        ("mention_micro_recall", "Mention recall (micro)"),
        ("mention_micro_f1", "Mention F1 (micro) = ACD F1"),
        ("mention_macro_f1", "Mention F1 (macro)"),
        ("mention_weighted_f1", "Mention F1 (weighted)"),
        ("aspect_set_precision_mean", "Set precision (mean per review)"),
        ("aspect_set_recall_mean", "Set recall (mean per review)"),
        ("aspect_set_f1_mean", "Set F1 (mean per review)"),
        ("aspect_set_iou_mean", "Aspect-set IoU / Jaccard (mean)"),
        ("aspect_set_iou_std", "Aspect-set IoU / Jaccard (std)"),
        ("aspect_map_macro", "mAP aspect detection (macro over aspects)"),
        ("aspect_map_micro", "mAP aspect detection (micro)"),
    ]
    lines = []
    for key, label in keys:
        if key in det:
            lines.append(f"  {label:<48}: {fmt(det[key])}")
    if "aspect_map_at_k" in det:
        lines.append("  mAP@k inside each review:")
        for k, v in det["aspect_map_at_k"].items():
            lines.append(f"    mAP{k:<4}: {fmt(v)}")
    if "cmc" in det:
        cmc = det["cmc"]
        lines.append("  CMC@k (aspect-detection ranking):")
        lines.append(f"    {'k':>3} {'Recall@k':>12} {'Hit@k':>12} {'All-aspects@k':>16}")
        for k in sorted(cmc["recall_at_k"], key=lambda x: int(x[1:])):
            lines.append(
                f"    {k:>3} {fmt(cmc['recall_at_k'][k]):>12} "
                f"{fmt(cmc['hit_at_k'][k]):>12} {fmt(cmc['all_aspects_at_k'][k]):>16}"
            )
    return lines


def class_table_lines(per_class: Dict[str, Dict[str, Any]]) -> List[str]:
    header = (
        f"  {'Class':<12} {'Support':>9} {'Precision':>10} {'Recall':>9} "
        f"{'F1':>9} {'AUC-OVR':>9} {'AP-OVR':>9}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for name, m in per_class.items():
        lines.append(
            f"  {name:<12} {m.get('support', 0):>9} {fmt(m.get('precision')):>10} "
            f"{fmt(m.get('recall')):>9} {fmt(m.get('f1')):>9} "
            f"{fmt(m.get('roc_auc')):>9} {fmt(m.get('average_precision')):>9}"
        )
    return lines


def confusion_lines(cm: Sequence[Sequence[int]], labels: Sequence[str] = LABEL_NAMES) -> List[str]:
    width = 10
    lines = ["  " + " " * 12 + "".join(f"{l:>{width}}" for l in labels)]
    for i, row in enumerate(cm):
        lines.append(f"  {labels[i]:<12}" + "".join(f"{int(v):>{width}}" for v in row))
    return lines


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def evaluate_split(
    model: MultiHeadModel,
    X,
    Y: np.ndarray,
    split_name: str,
    args,
) -> Dict[str, Any]:
    pred = model.predict(X)
    proba = model.predict_proba(X)
    flat_true = Y.ravel()
    flat_pred = pred.ravel()
    flat_proba = proba.reshape(-1, 4)

    global_metrics = classification_metrics(flat_true, flat_pred, flat_proba)
    per_aspect = per_aspect_metrics(Y, pred, proba, ASPECT_LIST)
    detection = multilabel_aspect_metrics(Y, pred, proba, ASPECT_LIST)
    cm = confusion_matrix(flat_true, flat_pred, labels=LABELS)
    cm_acd = confusion_matrix((flat_true > 0).astype(int), (flat_pred > 0).astype(int), labels=[0, 1])

    return {
        "split": split_name,
        "n_reviews": int(Y.shape[0]),
        "n_review_aspect_pairs": int(Y.size),
        "global": global_metrics,
        "per_aspect": per_aspect,
        "aspect_detection": detection,
        "confusion_matrix_global": cm.tolist(),
        "confusion_matrix_acd": cm_acd.tolist(),
        "predictions": pred,
        "probabilities": proba,
    }


def save_split_artifacts(
    run_dir: Path,
    result: Dict[str, Any],
    df: pd.DataFrame,
    split_name: str,
    args,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    metrics = {
        "split": split_name,
        "n_reviews": result["n_reviews"],
        "n_review_aspect_pairs": result["n_review_aspect_pairs"],
        "global": result["global"],
        "aspect_detection": result["aspect_detection"],
        "per_aspect": result["per_aspect"],
        "confusion_matrix_global": result["confusion_matrix_global"],
        "confusion_matrix_acd": result["confusion_matrix_acd"],
        "confusion_matrix_labels": LABEL_NAMES,
    }
    path = run_dir / f"metrics_{split_name}.json"
    write_json(path, metrics)
    out["metrics"] = str(path)

    # Per-class table
    rows = []
    for name, m in result["global"]["per_class"].items():
        rows.append(
            {
                "split": split_name,
                "class": name,
                "support": m.get("support"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "roc_auc_ovr": m.get("roc_auc"),
                "average_precision_ovr": m.get("average_precision"),
            }
        )
    path = run_dir / f"per_class_{split_name}.csv"
    write_csv_dicts(path, rows)
    out["per_class"] = str(path)

    # Per-aspect table
    rows = []
    for aspect, m in result["per_aspect"].items():
        row = {"split": split_name, "aspect": aspect, "n_samples": m.get("n_samples")}
        for key, value in m.items():
            if key in {"confusion_matrix", "per_class", "n_samples"}:
                continue
            row[key] = value
        rows.append(row)
    path = run_dir / f"per_aspect_{split_name}.csv"
    write_csv_dicts(path, rows)
    out["per_aspect"] = str(path)

    # Aspect detection table (one row)
    det = result["aspect_detection"]
    row = {k: v for k, v in det.items() if not isinstance(v, (dict, list))}
    for k, v in det.get("aspect_map_at_k", {}).items():
        row[f"aspect_map{k}"] = v
    for group in ("recall_at_k", "hit_at_k", "all_aspects_at_k"):
        for k, v in det.get("cmc", {}).get(group, {}).items():
            row[f"cmc_{group}{k}"] = v
    row["split"] = split_name
    path = run_dir / f"aspect_detection_{split_name}.csv"
    write_csv_dicts(path, [row])
    out["aspect_detection"] = str(path)

    # Confusion matrices
    path = run_dir / f"confusion_matrix_{split_name}.csv"
    write_csv(
        path,
        [
            ["true\\pred", *LABEL_NAMES],
            *[
                [LABEL_NAMES[i], *[int(v) for v in row]]
                for i, row in enumerate(result["confusion_matrix_global"])
            ],
        ],
        ["true\\pred", *LABEL_NAMES],
    )
    out["confusion_matrix"] = str(path)

    path = run_dir / f"confusion_matrix_acd_{split_name}.csv"
    write_csv(
        path,
        [["true\\pred", "None", "Mention"],
         ["None", *[int(v) for v in result["confusion_matrix_acd"][0]]],
         ["Mention", *[int(v) for v in result["confusion_matrix_acd"][1]]]],
        ["true\\pred", "None", "Mention"],
    )
    out["confusion_matrix_acd"] = str(path)

    # Per-aspect confusion matrices, long format
    rows = []
    for aspect, m in result["per_aspect"].items():
        cm = np.asarray(m["confusion_matrix"], dtype=int)
        for i, true_name in enumerate(LABEL_NAMES):
            for j, pred_name in enumerate(LABEL_NAMES):
                rows.append(
                    {
                        "split": split_name,
                        "aspect": aspect,
                        "true_label": true_name,
                        "pred_label": pred_name,
                        "count": int(cm[i, j]),
                    }
                )
    path = run_dir / f"confusion_matrix_per_aspect_{split_name}.csv"
    write_csv_dicts(path, rows)
    out["confusion_matrix_per_aspect"] = str(path)

    # Predictions, long format
    if args.save_predictions:
        pred = result["predictions"]
        proba = result["probabilities"]
        true_labels = result["true_labels"]
        text_col = args.text_col
        rows = []
        for r in range(len(df)):
            text = str(df.iloc[r].get(text_col, ""))
            review_index = df.iloc[r].get("index", r)
            for a, aspect in enumerate(ASPECT_LIST):
                p = proba[r, a]
                rows.append(
                    {
                        "split": split_name,
                        "review_row": r,
                        "review_index": review_index,
                        "text": text,
                        "aspect": aspect,
                        "true_label": int(true_labels[r, a]),
                        "true_name": LABEL_NAMES[int(true_labels[r, a])],
                        "pred_label": int(pred[r, a]),
                        "pred_name": LABEL_NAMES[int(pred[r, a])],
                        "p_None": float(p[0]),
                        "p_Negative": float(p[1]),
                        "p_Neutral": float(p[2]),
                        "p_Positive": float(p[3]),
                        "mention_score": float(1.0 - p[0]),
                        "correct": bool(int(true_labels[r, a]) == int(pred[r, a])),
                    }
                )
        path = run_dir / f"predictions_{split_name}.csv"
        write_csv_dicts(path, rows)
        out["predictions"] = str(path)

    if args.save_probabilities:
        path = run_dir / f"probabilities_{split_name}.npz"
        np.savez_compressed(
            path,
            probabilities=result["probabilities"].astype(np.float32),
            predictions=result["predictions"].astype(np.int8),
            true_labels=result.get("true_labels", np.zeros_like(result["predictions"])).astype(np.int8),
        )
        out["probabilities"] = str(path)

    return out


def run_single(
    model_name: str,
    seed: int,
    args,
    splits: Dict[str, Tuple[pd.DataFrame, List[str], np.ndarray]],
    vectorizer,
    X: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    tag = args.tag
    run_dir = output_dir / tag / model_name / f"seed_{seed}"
    results_txt = run_dir / "RESULTS.txt"
    if results_txt.exists() and not args.force:
        print(f"[skip] {run_dir} already exists (use --force to overwrite)")
        return {"status": "skipped", "run_dir": str(run_dir), "model": model_name, "seed": seed}

    if run_dir.exists() and args.force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_RECORDS.clear()

    command = " ".join([sys.executable, *sys.argv])
    env = environment_info()
    data_dir = Path(args.data_dir).resolve()
    file_map = {"train": args.train_file, "dev": args.dev_file, "test": args.test_file}
    data_summary = {
        name: split_summary(name, df, labels, args.text_col, data_dir / file_map[name])
        for name, (df, texts, labels) in splits.items()
    }

    print(f"\n=== {model_name} | seed {seed} | tag {args.tag} ===")
    print(f"Train rows: {len(splits['train'][0])}; Test rows: {len(splits['test'][0])}")
    start = time.perf_counter()
    Y_train = splits["train"][2]
    Y_dev = splits["dev"][2] if "dev" in splits else None
    model, epoch_log = fit_simple_model(
        model_name,
        X["train"],
        Y_train,
        X.get("dev"),
        Y_dev,
        args,
        seed,
    )
    model.vectorizer = vectorizer
    model.n_features = int(X["train"].shape[1])
    fit_seconds = time.perf_counter() - start
    print(f"Fit time: {fit_seconds:.2f}s | features: {model.n_features}")

    if args.save_model:
        import joblib

        model_path = run_dir / "model.joblib"
        joblib.dump(model, model_path, compress=3)
        print(f"Saved model: {model_path}")

    report = Report()
    report.rule()
    report.line("UIT-ViSFD ABSA — KẾT QUẢ THỰC NGHIỆM MÔ HÌNH ĐƠN GIẢN")
    report.line("Báo cáo sinh tự động, số liệu lấy từ lần chạy thực tế (không mô phỏng).")
    report.rule()

    run_id = f"{tag}/{model_name}/seed_{seed}"
    report.heading("1. THÔNG TIN LẦN CHẠY")
    report.kv("Run ID", run_id)
    report.kv("Model", model_name)
    report.kv("Model kind", MODEL_REGISTRY[model_name]["kind"])
    report.kv("Model description", MODEL_REGISTRY[model_name]["description"])
    report.kv("Seed", seed)
    report.kv("Tag", tag)
    report.kv("Timestamp", now_iso())
    report.kv("Command", command)
    report.kv("Wall time (fit)", f"{fit_seconds:.4f} s")
    report.kv("Output directory", str(run_dir))

    report.heading("2. MÔI TRƯỜNG")
    report.kv("Python", env["python"].splitlines()[0])
    report.kv("Executable", env["python_executable"])
    report.kv("Platform", env["platform"])
    report.kv("Hostname", env["hostname"])
    report.kv("CPU count", env["cpu_count"])
    report.kv("scikit-learn", env["sklearn"])
    report.kv("NumPy", env["numpy"])
    report.kv("Pandas", env["pandas"])
    report.kv("SciPy", env["scipy"])
    report.kv("Matplotlib", env["matplotlib"])
    report.kv("PyTorch", env.get("torch"))
    report.kv("CUDA available", env.get("cuda_available"))
    report.kv("GPU", env.get("gpu", "N/A"))
    report.kv("Git commit", env["git"].get("commit") or "N/A")
    report.kv("Git branch", env["git"].get("branch") or "N/A")
    report.kv("Git describe", env["git"].get("describe") or "N/A")

    report.heading("3. THÔNG TIN DỮ LIỆU")
    report.kv("Data directory", args.data_dir)
    report.kv("Text column", args.text_col)
    report.kv("Label column", args.label_col)
    report.kv("Label format", "{ASPECT#Polarity};... (e.g. {CAMERA#Positive};{BATTERY#Negative};)")
    report.kv("Fixed aspect order (10)", ", ".join(ASPECT_LIST))
    report.kv("Fixed label order (4)", ", ".join(LABEL_NAMES))
    for split_name, summary in data_summary.items():
        report.line()
        report.line(f"  [{split_name}] {summary['path']}")
        report.kv(f"  {split_name} rows", summary["rows"])
        report.kv(f"  {split_name} empty text", summary["empty_text_rows"])
        report.kv(
            f"  {split_name} label counts",
            ", ".join(f"{k}={v}" for k, v in summary["label_counts"].items()),
        )
        report.kv(
            f"  {split_name} aspect mentions",
            ", ".join(f"{k}={v}" for k, v in summary["aspect_mentions"].items()),
        )
        report.kv(
            f"  {split_name} aspects/review",
            f"mean={summary['aspect_count_per_review']['mean']:.3f}, "
            f"min={summary['aspect_count_per_review']['min']}, "
            f"max={summary['aspect_count_per_review']['max']}",
        )
        if summary["token_length"]:
            ts = summary["token_length"]
            report.kv(
                f"  {split_name} token length",
                f"mean={ts['mean']:.2f}, median={ts['median']:.0f}, "
                f"p25={ts['p25']:.0f}, p75={ts['p75']:.0f}, max={ts['max']:.0f}",
            )

    report.heading("4. SIÊU THAM SỐ (HYPERPARAMETERS)")
    report.kv("Vectorizer", json.dumps(vectorizer_description(args), ensure_ascii=False))
    report.kv("Number of features", model.n_features)
    report.kv("Number of heads", len(ASPECT_LIST))
    report.kv("Classes per head", ", ".join(LABEL_NAMES))
    report.kv("Class weight", args.class_weight)
    report.kv("C (LR/SVC)", args.C)
    report.kv("NB alpha", args.alpha)
    report.kv("Ridge alpha", args.ridge_alpha)
    report.kv("SGD alpha", args.sgd_alpha)
    report.kv("MLP alpha", args.mlp_alpha)
    report.kv("Max iterations (LR/SVC)", args.max_iter)
    report.kv("Solver (LR)", args.solver)
    report.kv("Epochs (SGD/MLP)", args.epochs)
    report.kv("Batch size (MLP)", args.batch_size)
    report.kv("Learning rate (MLP)", args.learning_rate)
    report.kv("Hidden layers (MLP)", args.hidden_layers)
    report.kv("Trees (RF/ET)", args.n_estimators)
    report.kv("Max depth (tree/RF/ET)", args.max_depth)
    report.kv("Min samples leaf", args.min_samples_leaf)
    report.kv("kNN neighbours", args.knn_neighbors)
    report.kv("Random seed", seed)

    report.heading("5. KIẾN TRÚC")
    report.line(
        "  Multi-head classifier: one independent 4-class classifier per aspect,\n"
        "  all sharing the same TF-IDF feature matrix.  The 10 heads preserve the\n"
        "  official UIT-ViSFD aspect order and the 4-class order None/Negative/\n"
        "  Neutral/Positive.  Head outputs are concatenated to (N, 10, 4)."
    )
    report.line()
    report.kv("Feature extraction", vectorizer_description(args)["type"])
    report.kv("Per-head estimator", model_name)
    report.kv("Loss / objective", "log loss (SGD/MLP) or the estimator's native objective")
    report.kv("Output shape", "(N reviews, 10 aspects, 4 labels)")
    report.kv("Constant heads", sum(1 for h in model.heads if h.is_constant))
    report.kv(
        "Head n_iter",
        ", ".join(
            f"{ASPECT_LIST[i]}={h.n_iter if h.n_iter is not None else 'N/A'}"
            for i, h in enumerate(model.heads)
        ),
    )

    # Evaluate splits
    results: Dict[str, Dict[str, Any]] = {}
    artifacts: Dict[str, Dict[str, str]] = {}
    for split_name in ("train", "dev", "test"):
        if split_name not in splits:
            continue
        if split_name == "train" and not args.eval_train:
            continue
        if split_name == "dev" and not args.eval_dev:
            continue
        df, texts, Y = splits[split_name]
        print(f"Evaluating {split_name} ...")
        res = evaluate_split(model, X[split_name], Y, split_name, args)
        res["true_labels"] = Y
        results[split_name] = res
        artifacts[split_name] = save_split_artifacts(run_dir, res, df, split_name, args)

    # Train log
    train_log_path = run_dir / "train_log.csv"
    if epoch_log:
        write_csv_dicts(train_log_path, epoch_log)
    else:
        write_csv_dicts(
            train_log_path,
            [{
                "epoch": "N/A",
                "note": "This estimator is not trained epoch-by-epoch; per-epoch loss/accuracy is not defined.",
            }],
        )

    # Figures
    test = results.get("test")
    figures_dir = run_dir / "figures"
    figures: Dict[str, str] = {}
    if test is not None and args.figures:
        y_true = test["true_labels"].ravel()
        y_pred = test["predictions"].ravel()
        proba = test["probabilities"].reshape(-1, 4)
        cm = np.asarray(test["confusion_matrix_global"])

        p = figures_dir / "confusion_matrix_test.png"
        plot_confusion(cm, LABEL_NAMES, "Confusion matrix — Test (review × aspect pairs)", p)
        register_figure(p, "Global 4x4 confusion matrix on Test", "plot_confusion", "metrics_test.json / confusion_matrix_test.csv", command)
        figures["confusion_matrix_test"] = str(p)

        aspect_f1 = {a: test["per_aspect"][a]["macro_f1"] for a in ASPECT_LIST}
        p = figures_dir / "per_aspect_macro_f1_test.png"
        plot_bar(list(aspect_f1.values()), list(aspect_f1.keys()), "Macro F1 per aspect — Test", "Macro F1", p)
        register_figure(p, "Per-aspect macro F1 on Test", "plot_bar", "per_aspect_test.csv", command)
        figures["per_aspect_macro_f1_test"] = str(p)

        class_f1 = [test["global"]["per_class"][n]["f1"] for n in LABEL_NAMES]
        p = figures_dir / "per_class_f1_test.png"
        plot_bar(class_f1, LABEL_NAMES, "F1 per sentiment class — Test", "F1", p)
        register_figure(p, "Per-class F1 on Test", "plot_bar", "per_class_test.csv", command)
        figures["per_class_f1_test"] = str(p)

        p = figures_dir / "roc_curves_test.png"
        plot_roc(y_true, proba, p)
        register_figure(p, "One-vs-rest ROC curves on Test", "plot_roc", "probabilities_test.npz", command)
        figures["roc_curves_test"] = str(p)

        p = figures_dir / "pr_curves_test.png"
        plot_pr(y_true, proba, p)
        register_figure(p, "One-vs-rest precision-recall curves on Test", "plot_pr", "probabilities_test.npz", command)
        figures["pr_curves_test"] = str(p)

        p = figures_dir / "calibration_test.png"
        plot_calibration(y_true, y_pred, proba, p)
        register_figure(p, "Reliability diagram / calibration on Test", "plot_calibration", "probabilities_test.npz", command)
        figures["calibration_test"] = str(p)

        p = figures_dir / "confidence_hist_test.png"
        plot_confidence_hist(y_true, y_pred, proba, p)
        register_figure(p, "Confidence histogram (correct vs incorrect) on Test", "plot_confidence_hist", "probabilities_test.npz", command)
        figures["confidence_hist_test"] = str(p)

        p = figures_dir / "cmc_test.png"
        plot_cmc(test["aspect_detection"]["cmc"], p)
        register_figure(p, "CMC@k aspect-detection curve on Test", "plot_cmc", "aspect_detection_test.csv", command)
        figures["cmc_test"] = str(p)

        ious = []
        for t, pr in zip((test["true_labels"] > 0).astype(int), (test["predictions"] > 0).astype(int)):
            union = int(np.sum(t | pr))
            ious.append(1.0 if union == 0 else int(np.sum(t & pr)) / union)
        p = figures_dir / "aspect_set_iou_hist_test.png"
        plot_hist(ious, "Aspect-set IoU distribution — Test", "IoU (Jaccard) between true and predicted aspect sets", p)
        register_figure(p, "Histogram of per-review aspect-set IoU on Test", "plot_hist", "computed from predictions_test.csv", command)
        figures["aspect_set_iou_hist_test"] = str(p)

        ap_map = test["aspect_detection"].get("aspect_ap_per_aspect", {})
        p = figures_dir / "aspect_ap_test.png"
        plot_bar(
            [ap_map[a] if ap_map.get(a) is not None else 0.0 for a in ASPECT_LIST],
            ASPECT_LIST,
            "Average precision of aspect detection per aspect — Test",
            "Average precision (AP)",
            p,
            horizontal=True,
        )
        register_figure(p, "Per-aspect AP for aspect detection on Test", "plot_bar", "aspect_detection_test.csv", command)
        figures["aspect_ap_test"] = str(p)

    if epoch_log:
        p = figures_dir / "learning_curves.png"
        plot_learning_curves(epoch_log, p)
        register_figure(p, "Train/Dev loss, accuracy and macro-F1 per epoch", "plot_learning_curves", "train_log.csv", command)
        figures["learning_curves"] = str(p)

    # Figure manifest
    if FIGURE_RECORDS:
        write_csv_dicts(run_dir / "figure_manifest.csv", FIGURE_RECORDS)

    # Write metrics results sections
    for split_name in ("train", "dev", "test"):
        if split_name not in results:
            continue
        res = results[split_name]
        report.heading(f"{'6' if split_name == 'train' else '7' if split_name == 'dev' else '8'}. KẾT QUẢ TRÊN TẬP {split_name.upper()}")
        report.kv("Số đánh giá", res["n_reviews"])
        report.kv("Số cặp (đánh giá × khía cạnh)", res["n_review_aspect_pairs"])
        report.line()
        report.line("  --- Độ đo phân loại tổng hợp (flatten toàn bộ 10 head) ---")
        report.lines.extend(metrics_table_lines(res["global"]))
        report.line()
        report.line("  --- Độ đo theo từng lớp cảm xúc ---")
        report.lines.extend(class_table_lines(res["global"]["per_class"]))
        report.line()
        report.line("  --- Độ đo theo từng khía cạnh ---")
        report.lines.extend(aspect_table_lines(res["per_aspect"]))
        report.line()
        report.line("  --- Độ đo mức khía cạnh / tập khía cạnh / xếp hạng ---")
        report.lines.extend(detection_table_lines(res["aspect_detection"]))
        report.line()
        report.line("  --- Confusion matrix 4x4 (toàn bộ cặp đánh giá × khía cạnh) ---")
        report.lines.extend(confusion_lines(res["confusion_matrix_global"]))
        report.line()
        report.line("  --- Confusion matrix ACD: None vs Mention ---")
        report.lines.extend(confusion_lines(res["confusion_matrix_acd"], ["None", "Mention"]))

    report.heading("9. LOSS / ACCURACY QUA TỪNG EPOCH")
    if epoch_log:
        header = f"  {'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} {'TrainF1':>9}"
        if "dev_loss" in epoch_log[0]:
            header += f" {'DevLoss':>10} {'DevAcc':>9} {'DevF1':>9}"
        report.line(header)
        report.line("  " + "-" * (len(header) - 2))
        for r in epoch_log:
            line = (
                f"  {r['epoch']:>5} {r['train_loss']:>10.4f} {r['train_accuracy']:>9.4f} "
                f"{r['train_macro_f1']:>9.4f}"
            )
            if "dev_loss" in r:
                line += f" {r['dev_loss']:>10.4f} {r['dev_accuracy']:>9.4f} {r['dev_macro_f1']:>9.4f}"
            report.line(line)
        report.kv("Best epoch (Dev macro F1)", max(epoch_log, key=lambda r: r.get("dev_macro_f1", r["train_macro_f1"]))["epoch"])
    else:
        report.line(
            "  Không áp dụng: mô hình này không huấn luyện theo epoch (closed-form,\n"
            "  batch hoặc ensemble).  Chỉ báo cáo kết quả cuối cùng.  Các mô hình có\n"
            "  đường cong epoch là sgd_log và mlp."
        )

    report.heading("10. ARTIFACTS (ĐƯỜNG DẪN FILE KẾT QUẢ)")
    report.kv("RESULTS.txt", str(results_txt))
    report.kv("config.json", str(run_dir / "config.json"))
    report.kv("environment.json", str(run_dir / "environment.json"))
    report.kv("train_log.csv", str(train_log_path))
    if args.save_model:
        report.kv("model.joblib", str(run_dir / "model.joblib"))
    for split_name, paths in artifacts.items():
        for key, value in paths.items():
            report.kv(f"{split_name}.{key}", value)
    for key, value in figures.items():
        report.kv(f"figure.{key}", value)
    if FIGURE_RECORDS:
        report.kv("figure_manifest.csv", str(run_dir / "figure_manifest.csv"))

    report.heading("11. LIÊN KẾT HÌNH/BẢNG ↔ CODE")
    report.line(
        "  Mỗi hình/bảng dưới đây được tạo bởi hàm trong file code và có thể tái lập\n"
        "  bằng đúng lệnh ở mục 1.  Manifest đầy đủ: figure_manifest.csv."
    )
    for rec in FIGURE_RECORDS:
        report.line(f"  - {Path(rec['artifact']).name:<40} -> {rec['code_function']}()")
        report.line(f"      code: {rec['code_file']}")
        report.line(f"      data: {rec['data_source']}")

    report.heading("12. GHI CHÚ VỀ ĐỘ ĐO VÀ GIỚI HẠN")
    report.line(
        "  * Accuracy/F1 micro/macro/weighted, Cohen's kappa (linear & quadratic),\n"
        "    MCC, ROC-AUC OVR, PR-AUC/AP, log loss, Brier, ECE, top-2/top-3,\n"
        "    confusion matrix: tính trực tiếp từ dự đoán 10 head.\n"
        "  * ACD F1: mention (label 1..3) vs None (label 0).\n"
        "  * ASC macro F1: chỉ tính trên các lớp 1..3 (bỏ None).\n"
        "  * Exact match: cả 10 khía cạnh đều đúng nhãn.\n"
        "  * Aspect-set IoU: Jaccard giữa tập khía cạnh thật và tập dự đoán của\n"
        "    từng đánh giá; đây là cách diễn giải IoU phù hợp cho ABSA, không phải\n"
        "    IoU phân đoạn ảnh.\n"
        "  * CMC@k: tỉ lệ khía cạnh thật được xếp trong top-k khía cạnh theo\n"
        "    mention score = 1 - P(None); đây là CMC xếp hạng khía cạnh, không phải\n"
        "    CMC nhận dạng khuôn mặt.\n"
        "  * mAP: average precision của phát hiện khía cạnh nhị phân; mAP@k là AP@k\n"
        "    trong từng đánh giá.\n"
        "  * Các độ đo ẩn tin (PSNR, SSIM, BER, capacity, robustness) KHÔNG áp dụng\n"
        "    vì bài toán không có ảnh bìa / payload nhúng.  Không tạo số liệu giả.\n"
        "  * Mô hình đơn giản chỉ dùng TF-IDF; không dùng checkpoint PhoBERT.\n"
        "    Kết quả PhoBERT trong báo cáo so sánh là kết quả đã chạy trước đó,\n"
        "    không trộn với lần chạy này."
    )

    report.heading("13. LỆNH TÁI LẬP")
    report.line(f"  {command}")
    report.line()
    report.line("  Hoặc chạy lại đúng một mô hình:")
    report.line(
        f"  {sys.executable} {Path(__file__).resolve()} --models {model_name} "
        f"--seeds {seed} --tag {tag} --force"
    )

    report.save(results_txt)

    # config / environment / data summary / run record
    write_json(run_dir / "config.json", vars(args))
    write_json(run_dir / "environment.json", env)
    write_json(run_dir / "data_summary.json", data_summary)
    write_json(run_dir / "epoch_log.json", epoch_log)

    test_metrics = results.get("test", {}).get("global", {})
    test_det = results.get("test", {}).get("aspect_detection", {})
    record = {
        "run_id": run_id,
        "tag": tag,
        "model": model_name,
        "model_kind": MODEL_REGISTRY[model_name]["kind"],
        "seed": seed,
        "status": "ok",
        "timestamp": now_iso(),
        "command": command,
        "run_dir": str(run_dir),
        "results_txt": str(results_txt),
        "train_rows": results.get("train", {}).get("n_reviews"),
        "dev_rows": results.get("dev", {}).get("n_reviews"),
        "test_rows": results.get("test", {}).get("n_reviews"),
        "n_features": model.n_features,
        "fit_seconds": fit_seconds,
        "accuracy_test": test_metrics.get("accuracy"),
        "balanced_accuracy_test": test_metrics.get("balanced_accuracy"),
        "micro_f1_test": test_metrics.get("micro_f1"),
        "macro_f1_test": test_metrics.get("macro_f1"),
        "weighted_f1_test": test_metrics.get("weighted_f1"),
        "cohen_kappa_test": test_metrics.get("cohen_kappa"),
        "cohen_kappa_quadratic_test": test_metrics.get("cohen_kappa_quadratic"),
        "mcc_test": test_metrics.get("mcc"),
        "roc_auc_ovr_macro_test": test_metrics.get("roc_auc_ovr_macro"),
        "pr_auc_macro_test": test_metrics.get("pr_auc_macro"),
        "log_loss_test": test_metrics.get("log_loss"),
        "ece_10bin_test": test_metrics.get("ece_10bin"),
        "sentiment_macro_f1_test": test_metrics.get("sentiment_macro_f1"),
        "acd_f1_test": test_det.get("acd_f1"),
        "exact_match_test": test_det.get("exact_match"),
        "aspect_set_iou_mean_test": test_det.get("aspect_set_iou_mean"),
        "aspect_map_macro_test": test_det.get("aspect_map_macro"),
        "cmc_recall_at_5_test": (test_det.get("cmc", {}).get("recall_at_k", {}) or {}).get("@5"),
        "epochs": len(epoch_log) if epoch_log else 0,
        "best_epoch": max(epoch_log, key=lambda r: r.get("dev_macro_f1", r["train_macro_f1"]))["epoch"] if epoch_log and "dev_macro_f1" in epoch_log[0] else None,
        "figures": len(FIGURE_RECORDS),
        "figure_manifest": str(run_dir / "figure_manifest.csv") if FIGURE_RECORDS else None,
    }
    write_json(run_dir / "run_record.json", record)

    # Console summary
    print(
        f"  Test: acc={fmt(test_metrics.get('accuracy'))} "
        f"macroF1={fmt(test_metrics.get('macro_f1'))} "
        f"microF1={fmt(test_metrics.get('micro_f1'))} "
        f"ACD-F1={fmt(test_det.get('acd_f1'))} "
        f"EM={fmt(test_det.get('exact_match'))}"
    )
    print(f"  Saved: {results_txt}")

    return record


# ---------------------------------------------------------------------------
# Global manifest / summary
# ---------------------------------------------------------------------------
def rebuild_manifest(output_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(output_dir.rglob("run_record.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.sort_values(["tag", "model", "seed"]).reset_index(drop=True)
    csv_path = output_dir / "manifest.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_json(output_dir / "manifest.json", records)
    return df


def build_summary(output_dir: Path, manifest: pd.DataFrame) -> None:
    if manifest.empty:
        return
    metric_cols = [
        "accuracy_test", "balanced_accuracy_test", "micro_f1_test", "macro_f1_test",
        "weighted_f1_test", "cohen_kappa_test", "cohen_kappa_quadratic_test", "mcc_test",
        "roc_auc_ovr_macro_test", "pr_auc_macro_test", "log_loss_test", "ece_10bin_test",
        "sentiment_macro_f1_test", "acd_f1_test", "exact_match_test",
        "aspect_set_iou_mean_test", "aspect_map_macro_test", "cmc_recall_at_5_test",
        "fit_seconds",
    ]
    rows = []
    for (tag, model), group in manifest.groupby(["tag", "model"], dropna=False):
        row = {
            "tag": tag,
            "model": model,
            "model_kind": group["model_kind"].iloc[0],
            "n_runs": len(group),
            "seeds": ",".join(str(int(s)) for s in sorted(group["seed"].dropna().unique())),
            "train_rows": group["train_rows"].iloc[0],
            "test_rows": group["test_rows"].iloc[0],
            "n_features": group["n_features"].iloc[0],
            "epochs": group["epochs"].iloc[0],
            "best_epoch_mean": group["best_epoch"].dropna().mean() if group["best_epoch"].notna().any() else None,
        }
        for col in metric_cols:
            if col not in group.columns:
                row[f"{col}_mean"] = None
                row[f"{col}_std"] = None
                continue
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(vals) == 0:
                row[f"{col}_mean"] = None
                row[f"{col}_std"] = None
            elif len(vals) == 1:
                row[f"{col}_mean"] = float(vals.iloc[0])
                row[f"{col}_std"] = None
            else:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["tag", "macro_f1_test_mean"], ascending=[True, False])
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "summary.json", rows)

    # Markdown summary
    lines = [
        "# Tổng hợp thực nghiệm mô hình đơn giản — UIT-ViSFD ABSA",
        "",
        f"Sinh lúc: {now_iso()}",
        "",
        "| Tag | Mô hình | Loại | Runs | Accuracy | Macro F1 | Micro F1 | ACD F1 | Exact Match | Kappa | MCC | ROC-AUC | mAP | IoU |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        def ms(col):
            m, s = r.get(f"{col}_mean"), r.get(f"{col}_std")
            if m is None or pd.isna(m):
                return "N/A"
            return f"{m * 100:.2f} ± {s * 100:.2f}" if s is not None and not pd.isna(s) else f"{m * 100:.2f}"
        lines.append(
            f"| {r['tag']} | {r['model']} | {r['model_kind']} | {r['n_runs']} | "
            f"{ms('accuracy_test')} | {ms('macro_f1_test')} | {ms('micro_f1_test')} | "
            f"{ms('acd_f1_test')} | {ms('exact_match_test')} | {ms('cohen_kappa_test')} | "
            f"{ms('mcc_test')} | {ms('roc_auc_ovr_macro_test')} | {ms('aspect_map_macro_test')} | "
            f"{ms('aspect_set_iou_mean_test')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_simple_experiments.py",
        description=(
            "Chạy thực nghiệm các mô hình đơn giản cho UIT-ViSFD ABSA.\n"
            "Kết quả được lưu đầy đủ vào RESULTS.txt + JSON/CSV + hình PNG."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python run_simple_experiments.py --list\n"
            "  python run_simple_experiments.py --models majority logistic_regression --seeds 42\n"
            "  python run_simple_experiments.py --models default --seeds 42 123 2024\n"
            "  python run_simple_experiments.py --models all --tag full_baselines --force\n\n"
            "Mô hình có đường cong epoch: sgd_log, mlp.\n"
            "Các độ đo ẩn tin (PSNR/SSIM/BER) không áp dụng cho ABSA; xem RESULTS.txt mục 12."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    data = parser.add_argument_group("Dữ liệu")
    data.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Thư mục chứa 3 file CSV đã chuẩn bị.")
    data.add_argument("--train-file", default="Train_prepared.csv", help="Tên file Train (mặc định: %(default)s).")
    data.add_argument("--dev-file", default="Dev_prepared.csv", help="Tên file Dev (mặc định: %(default)s).")
    data.add_argument("--test-file", default="Test_prepared.csv", help="Tên file Test (mặc định: %(default)s).")
    data.add_argument("--text-col", default="comment_segmented", help="Cột văn bản (mặc định: %(default)s).")
    data.add_argument("--label-col", default="label", help="Cột nhãn (mặc định: %(default)s).")
    data.add_argument("--limit-train", type=int, default=0, help="Chỉ dùng N dòng Train đầu (0 = tất cả).")
    data.add_argument("--limit-dev", type=int, default=0, help="Chỉ dùng N dòng Dev đầu (0 = tất cả).")
    data.add_argument("--limit-test", type=int, default=0, help="Chỉ dùng N dòng Test đầu (0 = tất cả).")

    model = parser.add_argument_group("Mô hình và huấn luyện")
    model.add_argument("--models", nargs="+", default=["default"], help=f"Danh sách mô hình hoặc 'default'/'all'. Có: {', '.join(ALL_MODELS)}")
    model.add_argument("--list", action="store_true", help="In danh sách mô hình rồi thoát.")
    model.add_argument("--seeds", nargs="+", type=int, default=[42], help="Một hoặc nhiều seed (mặc định: 42).")
    model.add_argument("--epochs", type=int, default=20, help="Số epoch cho sgd_log/mlp (mặc định: %(default)s).")
    model.add_argument("--batch-size", type=int, default=64, help="Batch size cho mlp (mặc định: %(default)s).")
    model.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate cho mlp (mặc định: %(default)s).")
    model.add_argument("--hidden-layers", nargs="+", type=int, default=[256, 128], help="Hidden layers cho mlp (mặc định: 256 128).")
    model.add_argument("--class-weight", choices=["none", "balanced", "balanced_subsample"], default="balanced", help="Cân bằng lớp (mặc định: %(default)s).")
    model.add_argument("--C", type=float, default=1.0, help="Hệ số C cho LogisticRegression/LinearSVC (mặc định: %(default)s).")
    model.add_argument("--alpha", type=float, default=1.0, help="Alpha cho MultinomialNB/ComplementNB/BernoulliNB (mặc định: %(default)s).")
    model.add_argument("--ridge-alpha", type=float, default=1.0, help="Alpha cho RidgeClassifier (mặc định: %(default)s).")
    model.add_argument("--sgd-alpha", type=float, default=1e-4, help="Alpha cho SGDClassifier (mặc định: %(default)s).")
    model.add_argument("--mlp-alpha", type=float, default=1e-4, help="Alpha cho MLPClassifier (mặc định: %(default)s).")
    model.add_argument("--max-iter", type=int, default=1000, help="Số vòng lặp tối đa cho LR/SVC (mặc định: %(default)s).")
    model.add_argument("--solver", default="lbfgs", choices=["lbfgs", "saga", "liblinear", "newton-cg", "newton-cholesky"], help="Solver cho LogisticRegression (mặc định: %(default)s).")
    model.add_argument("--n-estimators", type=int, default=200, help="Số cây cho RF/ET (mặc định: %(default)s).")
    model.add_argument("--max-depth", type=int, default=None, help="Độ sâu tối đa cho tree/RF/ET (mặc định: không giới hạn).")
    model.add_argument("--min-samples-leaf", type=int, default=1, help="min_samples_leaf cho tree/RF/ET (mặc định: %(default)s).")
    model.add_argument("--knn-neighbors", type=int, default=15, help="Số láng giềng cho knn (mặc định: %(default)s).")
    model.add_argument("--jobs", type=int, default=-1, help="Số job song song cho RF/ET/kNN (mặc định: -1).")

    feat = parser.add_argument_group("Đặc trưng TF-IDF")
    feat.add_argument("--max-features", type=int, default=30000, help="Số đặc trưng word tối đa (mặc định: %(default)s).")
    feat.add_argument("--min-df", type=float, default=2, help="min_df (mặc định: %(default)s).")
    feat.add_argument("--max-df", type=float, default=0.95, help="max_df (mặc định: %(default)s).")
    feat.add_argument("--ngram-min", type=int, default=1, help="n-gram nhỏ nhất (mặc định: %(default)s).")
    feat.add_argument("--ngram-max", type=int, default=2, help="n-gram lớn nhất (mặc định: %(default)s).")
    feat.add_argument("--char-ngram-max", type=int, default=0, help="char_wb n-gram lớn nhất; 0 = tắt (mặc định: %(default)s).")
    feat.add_argument("--sublinear-tf", action=argparse.BooleanOptionalAction, default=True, help="Dùng sublinear_tf (mặc định: bật).")
    feat.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True, help="Chuyển chữ thường (mặc định: bật).")

    out = parser.add_argument_group("Đầu ra và đánh giá")
    out.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Thư mục kết quả (mặc định: %(default)s).")
    out.add_argument("--tag", default="simple_models", help="Nhãn nhóm thực nghiệm (mặc định: %(default)s).")
    out.add_argument("--force", action="store_true", help="Ghi đè kết quả đã có.")
    out.add_argument("--eval-train", action=argparse.BooleanOptionalAction, default=True, help="Đánh giá trên Train (mặc định: bật).")
    out.add_argument("--eval-dev", action=argparse.BooleanOptionalAction, default=True, help="Đánh giá trên Dev (mặc định: bật).")
    out.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True, help="Lưu predictions CSV (mặc định: bật).")
    out.add_argument("--save-probabilities", action=argparse.BooleanOptionalAction, default=True, help="Lưu probabilities NPZ (mặc định: bật).")
    out.add_argument("--save-model", action="store_true", help="Lưu vectorizer + 10 head bằng joblib (tốn dung lượng).")
    out.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True, help="Vẽ hình PNG (mặc định: bật).")
    out.add_argument("--figures-dpi", type=int, default=FIGURE_DPI, help="DPI hình (mặc định: %(default)s).")
    out.add_argument("--quiet", action="store_true", help="Giảm log ra màn hình.")
    out.add_argument("--dry-run", action="store_true", help="Chỉ in cấu hình, không huấn luyện.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    global FIGURE_DPI
    FIGURE_DPI = args.figures_dpi

    if args.list:
        print("Các mô hình có sẵn:\n")
        print(describe_models())
        print("\nMặc định (--models default): " + ", ".join(DEFAULT_MODELS))
        print("Tất cả (--models all): " + ", ".join(ALL_MODELS))
        return 0

    models = resolve_models(args.models)
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps({
            "models": models,
            "seeds": args.seeds,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "tag": args.tag,
            "hyperparameters": {k: v for k, v in vars(args).items() if k not in {"models", "seeds"}},
        }, ensure_ascii=False, indent=2))
        return 0

    print("=" * 72)
    print("UIT-ViSFD ABSA — CLI experiment runner")
    print("=" * 72)
    print(f"Models : {', '.join(models)}")
    print(f"Seeds  : {args.seeds}")
    print(f"Data   : {data_dir}")
    print(f"Output : {output_dir}")
    print(f"Tag    : {args.tag}")
    print("=" * 72)

    splits: Dict[str, Tuple[pd.DataFrame, List[str], np.ndarray]] = {}
    for name, fname, limit in (
        ("train", args.train_file, args.limit_train),
        ("dev", args.dev_file, args.limit_dev),
        ("test", args.test_file, args.limit_test),
    ):
        path = data_dir / fname
        if name == "dev" and not path.exists():
            print(f"[warn] Dev file not found, skipping Dev: {path}")
            continue
        df, texts, labels = read_split(path, args.text_col, args.label_col, limit=limit)
        splits[name] = (df, texts, labels)
        print(f"Loaded {name}: {len(df)} rows -> {path.name}")

    # Fit the vectorizer once on Train; transform Dev/Test with the same matrix.
    vectorizer = build_vectorizer(args)
    X: Dict[str, Any] = {}
    X["train"] = vectorizer.fit_transform(splits["train"][1])
    for name in ("dev", "test"):
        if name in splits:
            X[name] = vectorizer.transform(splits[name][1])
    print(f"TF-IDF features: {X['train'].shape[1]}")

    records = []
    failures = []
    for model_name in models:
        for seed in args.seeds:
            try:
                rec = run_single(model_name, seed, args, splits, vectorizer, X, output_dir)
                records.append(rec)
            except Exception as exc:
                failures.append((model_name, seed, str(exc)))
                print(f"[ERROR] {model_name} seed {seed}: {exc}")
                traceback.print_exc()

    manifest = rebuild_manifest(output_dir)
    build_summary(output_dir, manifest)

    print("\n" + "=" * 72)
    print("HOÀN TẤT")
    print("=" * 72)
    print(f"Runs OK   : {sum(1 for r in records if r.get('status') == 'ok')}")
    print(f"Skipped   : {sum(1 for r in records if r.get('status') == 'skipped')}")
    print(f"Failed    : {len(failures)}")
    print(f"Manifest  : {output_dir / 'manifest.csv'}")
    print(f"Summary   : {output_dir / 'summary.csv'}")
    print(f"Summary MD: {output_dir / 'summary.md'}")
    for model_name, seed, msg in failures:
        print(f"  - {model_name} seed {seed}: {msg}")

    # Per-run quick table
    if not manifest.empty:
        cols = ["model", "seed", "accuracy_test", "macro_f1_test", "micro_f1_test", "acd_f1_test", "exact_match_test"]
        available = [c for c in cols if c in manifest.columns]
        print("\nKết quả Test (một dòng / run):")
        print(manifest[available].to_string(index=False))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
