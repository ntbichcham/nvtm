"""
experiment_metrics.py
=====================
Comprehensive evaluation metrics for the UIT-ViSFD ABSA experiments.

The project predicts 10 aspect heads at once; each head is a 4-class problem
(0=None, 1=Negative, 2=Neutral, 3=Positive).  This module therefore reports:

  * sample-level classification metrics (accuracy, P/R/F1 micro/macro/weighted,
    balanced accuracy, Cohen's kappa, quadratic kappa, MCC, top-k accuracy)
  * probability metrics (log loss, multi-class Brier score, ECE)
  * ROC-AUC (OVR macro/micro/weighted) and PR-AUC / average precision
  * per-class and per-aspect breakdowns
  * confusion matrices (global 4x4 and one 4x4 per aspect)
  * aspect-level set metrics (exact match, Hamming, set P/R/F1, IoU/Jaccard)
  * aspect-detection ranking metrics (mAP, CMC@k, mAP@k)

Design notes
------------
CMC, IoU and mAP are not native ABSA metrics.  They are adapted here in a
transparent way and the definition is written into every RESULTS.txt:
  * IoU      -> Jaccard similarity between the predicted and true aspect sets
                of a review (10 aspects, binary mention).
  * CMC@k    -> fraction of true mentioned aspects ranked in the top-k aspects
                by the model's mention score (1 - P(None)).
  * mAP      -> average precision of binary aspect detection; mAP@k is the
                ranking AP@k inside each review.
No image/steganography metric is computed because this task has no cover
image or embedded payload.  See metric_applicability.csv.

Author: Nhom 2 - CT255
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.metrics")

LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["None", "Negative", "Neutral", "Positive"]
SENTIMENT_LABELS = [1, 2, 3]
SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(value: float) -> Optional[float]:
    """Convert numpy scalars to plain Python floats; NaN/Inf -> None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _mean(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    return float(np.mean(values))


def _std(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    return float(np.std(values, ddof=0))


def normalize_probabilities(proba: np.ndarray) -> np.ndarray:
    """Ensure each row sums to 1; rows with zero mass become uniform."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError("proba must be 2-D")
    row_sums = proba.sum(axis=1, keepdims=True)
    uniform = np.full_like(proba, 1.0 / proba.shape[1])
    return np.divide(proba, row_sums, out=uniform, where=row_sums > 0)


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> Optional[float]:
    """Expected Calibration Error (confidence vs. accuracy, equal-width bins)."""
    if proba is None or len(y_true) == 0:
        return None
    conf = np.max(proba, axis=1)
    correct = (y_true == y_pred).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - conf[mask].mean())
    return _f(ece)


# ---------------------------------------------------------------------------
# Single-output (4-class) metrics
# ---------------------------------------------------------------------------
def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    proba: Optional[np.ndarray] = None,
    labels: Sequence[int] = LABELS,
    label_names: Sequence[str] = LABEL_NAMES,
) -> Dict:
    """Full 4-class metric bundle for one output (global or one aspect)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(labels)
    out: Dict = {
        "n_samples": int(len(y_true)),
        "accuracy": _f(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _f(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": _f(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "cohen_kappa_quadratic": _f(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        ),
        "mcc": _f(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    for avg in ("micro", "macro", "weighted"):
        out[f"{avg}_precision"] = _f(
            precision_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )
        out[f"{avg}_recall"] = _f(
            recall_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )
        out[f"{avg}_f1"] = _f(
            f1_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )

    # Per-class P/R/F1 + support + one-vs-rest AUC/AP
    p_cls = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r_cls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f_cls = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class: Dict[str, Dict] = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": _f(p_cls[i]),
            "recall": _f(r_cls[i]),
            "f1": _f(f_cls[i]),
            "support": int((y_true == labels[i]).sum()),
            "roc_auc": None,
            "average_precision": None,
        }

    # Sentiment-only macro metrics over 1..3 (ASC protocol)
    mask = np.isin(y_true, SENTIMENT_LABELS) | np.isin(y_pred, SENTIMENT_LABELS)
    if mask.sum() > 0:
        out["sentiment_macro_precision"] = _f(
            precision_score(
                y_true[mask], y_pred[mask], labels=SENTIMENT_LABELS,
                average="macro", zero_division=0
            )
        )
        out["sentiment_macro_recall"] = _f(
            recall_score(
                y_true[mask], y_pred[mask], labels=SENTIMENT_LABELS,
                average="macro", zero_division=0
            )
        )
        out["sentiment_macro_f1"] = _f(
            f1_score(
                y_true[mask], y_pred[mask], labels=SENTIMENT_LABELS,
                average="macro", zero_division=0
            )
        )
    else:
        out["sentiment_macro_precision"] = None
        out["sentiment_macro_recall"] = None
        out["sentiment_macro_f1"] = None

    # Aspect Category Detection: mention (1..3) vs None (0)
    bin_true = (y_true > 0).astype(int)
    bin_pred = (y_pred > 0).astype(int)
    out["acd_precision"] = _f(precision_score(bin_true, bin_pred, zero_division=0))
    out["acd_recall"] = _f(recall_score(bin_true, bin_pred, zero_division=0))
    out["acd_f1"] = _f(f1_score(bin_true, bin_pred, zero_division=0))

    if proba is None:
        out["per_class"] = per_class
        return out

    proba = normalize_probabilities(np.asarray(proba, dtype=np.float64))
    out["log_loss"] = _f(log_loss(y_true, proba, labels=labels))
    onehot = np.zeros_like(proba)
    for i, label in enumerate(labels):
        onehot[y_true == label, i] = 1.0
    out["brier_score_multiclass"] = _f(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    order = np.argsort(-proba, axis=1)
    for k in (2, 3):
        if proba.shape[1] >= k:
            hits = [y_true[i] in order[i, :k] for i in range(len(y_true))]
            out[f"top{k}_accuracy"] = _f(np.mean(hits))

    present = [l for l in labels if np.any(y_true == l)]
    if len(present) >= 2:
        try:
            out["roc_auc_ovr_macro"] = _f(
                roc_auc_score(
                    y_true, proba[:, present], multi_class="ovr",
                    average="macro", labels=present
                )
            )
            out["roc_auc_ovr_weighted"] = _f(
                roc_auc_score(
                    y_true, proba[:, present], multi_class="ovr",
                    average="weighted", labels=present
                )
            )
        except ValueError:
            out["roc_auc_ovr_macro"] = None
            out["roc_auc_ovr_weighted"] = None
        try:
            out["roc_auc_ovr_micro"] = _f(
                roc_auc_score(
                    y_true, proba[:, present], multi_class="ovr",
                    average="micro", labels=present
                )
            )
        except ValueError:
            out["roc_auc_ovr_micro"] = None

        # one-vs-rest AP / AUC for each class
        for i, name in enumerate(label_names):
            y_bin = (y_true == labels[i]).astype(int)
            if y_bin.min() == y_bin.max():
                continue
            try:
                per_class[name]["roc_auc"] = _f(roc_auc_score(y_bin, proba[:, i]))
            except ValueError:
                pass
            try:
                per_class[name]["average_precision"] = _f(
                    average_precision_score(y_bin, proba[:, i])
                )
            except ValueError:
                pass

        # macro / micro / weighted PR-AUC
        aps = []
        for i in range(len(labels)):
            y_bin = (y_true == labels[i]).astype(int)
            if y_bin.min() == y_bin.max():
                continue
            aps.append(average_precision_score(y_bin, proba[:, i]))
        out["pr_auc_macro"] = _f(np.mean(aps)) if aps else None
        try:
            out["pr_auc_micro"] = _f(
                average_precision_score(
                    np.eye(len(labels))[y_true][:, present].ravel(),
                    proba[:, present].ravel(),
                )
            )
        except ValueError:
            out["pr_auc_micro"] = None

    out["ece_10bin"] = expected_calibration_error(y_true, y_pred, proba)
    out["per_class"] = per_class
    return out


# ---------------------------------------------------------------------------
# Multi-output (10 aspects) metrics
# ---------------------------------------------------------------------------
def multilabel_aspect_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    proba: Optional[np.ndarray] = None,
    aspect_names: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Aspect-level metrics across all 10 heads.

    Y_true, Y_pred : (N, 10) int arrays in {0,1,2,3}
    proba          : (N, 10, 4) probabilities, optional
    """
    Y_true = np.asarray(Y_true, dtype=int)
    Y_pred = np.asarray(Y_pred, dtype=int)
    n, n_aspects = Y_true.shape
    aspect_names = list(aspect_names or [f"A{i}" for i in range(n_aspects)])

    bin_true = (Y_true > 0).astype(int)
    bin_pred = (Y_pred > 0).astype(int)

    out: Dict = {
        "n_reviews": int(n),
        "n_aspects": int(n_aspects),
        "exact_match": _f(np.mean(np.all(Y_true == Y_pred, axis=1))),
        "hamming_loss_4class": _f(np.mean(Y_true != Y_pred)),
        "hamming_loss_mention": _f(np.mean(bin_true != bin_pred)),
        "subset_accuracy": _f(np.mean(np.all(bin_true == bin_pred, axis=1))),
    }

    for avg in ("micro", "macro", "weighted"):
        out[f"mention_{avg}_precision"] = _f(
            precision_score(bin_true, bin_pred, average=avg, zero_division=0)
        )
        out[f"mention_{avg}_recall"] = _f(
            recall_score(bin_true, bin_pred, average=avg, zero_division=0)
        )
        out[f"mention_{avg}_f1"] = _f(
            f1_score(bin_true, bin_pred, average=avg, zero_division=0)
        )

    out["acd_precision"] = out["mention_micro_precision"]
    out["acd_recall"] = out["mention_micro_recall"]
    out["acd_f1"] = out["mention_micro_f1"]

    # Set-level precision / recall / F1 / IoU per review
    set_p, set_r, set_f1, ious = [], [], [], []
    for t, p in zip(bin_true, bin_pred):
        inter = int(np.sum(t & p))
        union = int(np.sum(t | p))
        t_sum = int(np.sum(t))
        p_sum = int(np.sum(p))
        ious.append(1.0 if union == 0 else inter / union)
        set_p.append(1.0 if p_sum == 0 and t_sum == 0 else (inter / p_sum if p_sum else 0.0))
        set_r.append(1.0 if t_sum == 0 else inter / t_sum)
        denom = (set_p[-1] + set_r[-1])
        set_f1.append(0.0 if denom == 0 else 2 * set_p[-1] * set_r[-1] / denom)

    out["aspect_set_iou_mean"] = _f(np.mean(ious))
    out["aspect_set_iou_std"] = _f(np.std(ious))
    out["aspect_set_precision_mean"] = _f(np.mean(set_p))
    out["aspect_set_recall_mean"] = _f(np.mean(set_r))
    out["aspect_set_f1_mean"] = _f(np.mean(set_f1))
    out["aspect_set_f1_macro"] = _f(np.mean(set_f1))

    if proba is None:
        return out

    proba = np.asarray(proba, dtype=np.float64)
    mention_score = 1.0 - proba[:, :, 0]  # P(non-None)

    # Binary aspect detection AP (macro over aspects with both classes present)
    aps = []
    ap_per_aspect = {}
    for a in range(n_aspects):
        if bin_true[:, a].min() == bin_true[:, a].max():
            ap_per_aspect[aspect_names[a]] = None
            continue
        ap = average_precision_score(bin_true[:, a], mention_score[:, a])
        aps.append(ap)
        ap_per_aspect[aspect_names[a]] = _f(ap)
    out["aspect_map_macro"] = _f(np.mean(aps)) if aps else None
    out["aspect_ap_per_aspect"] = ap_per_aspect

    # Micro AP over all review-aspect pairs
    try:
        out["aspect_map_micro"] = _f(
            average_precision_score(bin_true.ravel(), mention_score.ravel())
        )
    except ValueError:
        out["aspect_map_micro"] = None

    # CMC@k and mAP@k over the ranking of aspects inside each review
    cmc_recall, cmc_hit, cmc_all = {}, {}, {}
    for k in range(1, n_aspects + 1):
        rec, hit, all_hit = [], [], []
        for t, s in zip(bin_true, mention_score):
            true_idx = np.where(t == 1)[0]
            if len(true_idx) == 0:
                continue
            topk = np.argsort(-s)[:k]
            inter = len(set(true_idx.tolist()) & set(topk.tolist()))
            rec.append(inter / len(true_idx))
            hit.append(1.0 if inter > 0 else 0.0)
            all_hit.append(1.0 if inter == len(true_idx) else 0.0)
        cmc_recall[f"@{k}"] = _f(np.mean(rec)) if rec else None
        cmc_hit[f"@{k}"] = _f(np.mean(hit)) if hit else None
        cmc_all[f"@{k}"] = _f(np.mean(all_hit)) if all_hit else None
    out["cmc"] = {
        "recall_at_k": cmc_recall,
        "hit_at_k": cmc_hit,
        "all_aspects_at_k": cmc_all,
    }

    map_at_k = {}
    for k in (1, 3, 5, 10):
        vals = []
        for t, s in zip(bin_true, mention_score):
            true_idx = set(np.where(t == 1)[0].tolist())
            if not true_idx:
                continue
            order = np.argsort(-s)[:k]
            hits = 0
            precisions = []
            for i, idx in enumerate(order, start=1):
                if idx in true_idx:
                    hits += 1
                precisions.append(hits / i)
            vals.append(sum(precisions) / min(len(true_idx), k))
        map_at_k[f"@{k}"] = _f(np.mean(vals)) if vals else None
    out["aspect_map_at_k"] = map_at_k

    return out


def per_aspect_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    proba: Optional[np.ndarray],
    aspect_names: Sequence[str],
) -> Dict[str, Dict]:
    """Run the full 4-class metric bundle independently for each aspect."""
    out = {}
    for a, name in enumerate(aspect_names):
        p = None if proba is None else proba[:, a, :]
        out[name] = classification_metrics(Y_true[:, a], Y_pred[:, a], p)
    return out


def build_confusion_matrix_rows(
    y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str] = LABEL_NAMES
) -> List[List]:
    """Confusion matrix as list-of-lists with a header row for CSV export."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    rows = [["true\\pred"] + list(label_names)]
    for i, name in enumerate(label_names):
        rows.append([name] + [int(v) for v in cm[i]])
    return rows


def aggregate_runs(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Mean/std/count for multi-seed aggregation."""
    clean = [v for v in values if v is not None]
    return {
        "mean": _f(np.mean(clean)) if clean else None,
        "std": _f(np.std(clean, ddof=0)) if clean else None,
        "n": len(clean),
    }
