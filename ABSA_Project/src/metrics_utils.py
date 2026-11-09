"""
metrics_utils.py
Shared metric computation so that PhoBERT and the BiLSTM baseline are scored
by exactly the same code (macro/micro/weighted P-R-F1, sentiment F1, ACD, EM,
and per-aspect breakdown).
Author: Nhom 2 - Thanh vien 3
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from dataset_loader import ASPECT_LIST


def compute_full_metrics(all_preds, all_labels):
    """
    all_preds, all_labels: integer arrays of shape (N, 10) with values in {0,1,2,3}.
    Returns a JSON-serializable dict.
    """
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    flat_p = all_preds.flatten()
    flat_t = all_labels.flatten()

    exact_match = float((all_preds == all_labels).all(axis=1).mean())

    metrics = {
        'exact_match': exact_match,
        'micro_precision': float(precision_score(flat_t, flat_p, average='micro', zero_division=0)),
        'micro_recall': float(recall_score(flat_t, flat_p, average='micro', zero_division=0)),
        'micro_f1': float(f1_score(flat_t, flat_p, average='micro', zero_division=0)),
        'macro_precision': float(precision_score(flat_t, flat_p, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(flat_t, flat_p, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(flat_t, flat_p, average='macro', zero_division=0)),
        'weighted_precision': float(precision_score(flat_t, flat_p, average='weighted', zero_division=0)),
        'weighted_recall': float(recall_score(flat_t, flat_p, average='weighted', zero_division=0)),
        'weighted_f1': float(f1_score(flat_t, flat_p, average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(flat_t, flat_p)),
    }

    # Per polarity class (None / Negative / Neutral / Positive)
    p_cls = precision_score(flat_t, flat_p, labels=[0, 1, 2, 3], average=None, zero_division=0)
    r_cls = recall_score(flat_t, flat_p, labels=[0, 1, 2, 3], average=None, zero_division=0)
    f_cls = f1_score(flat_t, flat_p, labels=[0, 1, 2, 3], average=None, zero_division=0)
    metrics['per_class'] = {
        name: {'precision': float(p_cls[i]), 'recall': float(r_cls[i]), 'f1': float(f_cls[i]),
               'support': int((flat_t == i).sum())}
        for i, name in enumerate(['None', 'Negative', 'Neutral', 'Positive'])
    }

    # Sentiment-only macro metrics over classes 1..3
    mask = (flat_t > 0) | (flat_p > 0)
    if mask.sum() > 0:
        metrics['sentiment_macro_precision'] = float(
            precision_score(flat_t[mask], flat_p[mask], labels=[1, 2, 3], average='macro', zero_division=0))
        metrics['sentiment_macro_recall'] = float(
            recall_score(flat_t[mask], flat_p[mask], labels=[1, 2, 3], average='macro', zero_division=0))
        metrics['sentiment_macro_f1'] = float(
            f1_score(flat_t[mask], flat_p[mask], labels=[1, 2, 3], average='macro', zero_division=0))
    else:
        metrics['sentiment_macro_precision'] = 0.0
        metrics['sentiment_macro_recall'] = 0.0
        metrics['sentiment_macro_f1'] = 0.0

    # Aspect Category Detection (binary mention vs None)
    bin_t = (flat_t > 0).astype(int)
    bin_p = (flat_p > 0).astype(int)
    metrics['acd_precision'] = float(precision_score(bin_t, bin_p, zero_division=0))
    metrics['acd_recall'] = float(recall_score(bin_t, bin_p, zero_division=0))
    metrics['acd_f1'] = float(f1_score(bin_t, bin_p, zero_division=0))

    # Per-aspect breakdown
    per_aspect = {}
    for i, aspect in enumerate(ASPECT_LIST):
        yt, yp = all_labels[:, i], all_preds[:, i]
        per_aspect[aspect] = {
            'accuracy': float(accuracy_score(yt, yp)),
            'precision_macro': float(precision_score(yt, yp, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(yt, yp, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(yt, yp, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(yt, yp, average='weighted', zero_division=0)),
            'acd_f1': float(f1_score((yt > 0).astype(int), (yp > 0).astype(int), zero_division=0)),
            'support': {name: int((yt == k).sum())
                        for k, name in enumerate(['None', 'Negative', 'Neutral', 'Positive'])}
        }
    metrics['per_aspect'] = per_aspect
    return metrics
