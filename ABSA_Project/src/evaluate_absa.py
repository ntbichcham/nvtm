"""
evaluate_absa.py
Unified prediction export and evaluation suite for Multi-Head ABSA.

This module provides:
  * one prediction schema (CSV + JSONL) shared by PhoBERT and BiLSTM;
  * Precision / Recall / F1 at cell level (micro, macro, weighted);
  * Aspect Category Detection (ACD) macro and micro F1 over the 10 aspects;
  * Aspect Sentiment Classification (ASC) macro and micro F1 over classes 1-3;
  * Aspect-Polarity Pair F1 (micro and macro);
  * Per-aspect table, Exact Match, classification report and confusion matrix;
  * error cases and a regression check against a previous metrics JSON.

Label convention is imported from `dataset_loader`, the same module used by the
training pipeline, so the evaluation cannot drift from the training labels.

Author: Nhom 2 - Thanh vien 2
"""

import os
import sys
import json
import time
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from dataset_loader import (
    ASPECT_LIST,
    IDX2POLARITY,
    POLARITY2IDX,
    NUM_CLASSES,
    parse_label_string,
)
from metrics_utils import compute_full_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLARITY_NAMES = [IDX2POLARITY[i] for i in range(NUM_CLASSES)]
SENTIMENT_CLASSES = [1, 2, 3]

# Zero-division policy used by every metric in this module:
#   * cell-level P/R/F1 follow scikit-learn's zero_division=0;
#   * ACD macro skips aspects that have neither a true nor a predicted mention
#     (the F1 is undefined there). If *no* aspect is defined, the macro score
#     is 1.0 because "detecting nothing" is the correct answer for that batch;
#   * pair macro averages over the union of observed (aspect, polarity) pairs.


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def _safe_div(numerator, denominator, default=0.0):
    return float(numerator) / float(denominator) if denominator else float(default)


def _prf_from_counts(tp, fp, fn):
    """Precision, recall and F1 from integer counts (zero_division=0)."""
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _as_arrays(all_preds, all_labels):
    preds = np.asarray(all_preds, dtype=np.int64)
    labels = np.asarray(all_labels, dtype=np.int64)
    if preds.shape != labels.shape:
        raise ValueError(f"shape mismatch: preds {preds.shape} vs labels {labels.shape}")
    if preds.ndim != 2 or preds.shape[1] != len(ASPECT_LIST):
        raise ValueError(f"expected (N, {len(ASPECT_LIST)}) arrays, got {preds.shape}")
    return preds, labels


# ---------------------------------------------------------------------------
# New metrics
# ---------------------------------------------------------------------------
def aspect_detection_metrics(all_preds, all_labels):
    """
    ACD treats every cell as binary: 1 if the aspect is mentioned (class > 0).

    Returns
    -------
    micro : dict   precision / recall / f1 over all (sample, aspect) cells
    macro : dict   mean of per-aspect P/R/F1 over *defined* aspects
    defined_aspects : int
        Number of aspects with at least one positive label or prediction.
        If 0, micro follows zero_division=0 (0.0) while macro is 1.0.
    """
    preds, labels = _as_arrays(all_preds, all_labels)
    true_bin = labels > 0
    pred_bin = preds > 0

    tp = int(np.logical_and(true_bin, pred_bin).sum())
    fp = int(np.logical_and(~true_bin, pred_bin).sum())
    fn = int(np.logical_and(true_bin, ~pred_bin).sum())
    micro = _prf_from_counts(tp, fp, fn)

    per_aspect = []
    for idx in range(len(ASPECT_LIST)):
        t = true_bin[:, idx]
        p = pred_bin[:, idx]
        tp_a = int(np.logical_and(t, p).sum())
        fp_a = int(np.logical_and(~t, p).sum())
        fn_a = int(np.logical_and(t, ~p).sum())
        if (tp_a + fp_a + fn_a) > 0:
            per_aspect.append(_prf_from_counts(tp_a, fp_a, fn_a))

    if per_aspect:
        macro = {
            key: float(np.mean([row[key] for row in per_aspect]))
            for key in ("precision", "recall", "f1")
        }
        defined = len(per_aspect)
    else:
        macro = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        defined = 0

    return micro, macro, defined


def sentiment_metrics(all_preds, all_labels):
    """
    ASC metrics on cells where the aspect is present in the truth or prediction.

    Macro averages over the three polarity classes (Negative, Neutral, Positive),
    so a class with no support still contributes F1 = 0 (same convention as the
    existing `compute_full_metrics.sentiment_macro_f1`).

    Returns
    -------
    micro, macro : dict with precision / recall / f1
    support : dict with the number of evaluated cells per class
    """
    preds, labels = _as_arrays(all_preds, all_labels)
    mask = (labels > 0) | (preds > 0)
    y_true = labels[mask]
    y_pred = preds[mask]

    if y_true.size == 0:
        empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return dict(empty), dict(empty), {"Negative": 0, "Neutral": 0, "Positive": 0}

    micro = {
        "precision": float(precision_score(y_true, y_pred, labels=SENTIMENT_CLASSES,
                                           average="micro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=SENTIMENT_CLASSES,
                                     average="micro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=SENTIMENT_CLASSES,
                             average="micro", zero_division=0)),
    }
    macro = {
        "precision": float(precision_score(y_true, y_pred, labels=SENTIMENT_CLASSES,
                                           average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=SENTIMENT_CLASSES,
                                     average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=SENTIMENT_CLASSES,
                             average="macro", zero_division=0)),
    }
    support = {
        name: int((y_true == cls).sum())
        for name, cls in zip(("Negative", "Neutral", "Positive"), SENTIMENT_CLASSES)
    }
    return micro, macro, support


def pair_f1(all_preds, all_labels):
    """
    Aspect-Polarity Pair F1.

    A pair is the tuple (aspect, polarity) with polarity in {Negative, Neutral,
    Positive}. Because each aspect has exactly one label per comment, the count
    of a pair equals the number of comments expressing it. Counts are compared
    as multisets, which makes micro-F1 equivalent to the standard SemEval-style
    pair evaluation. Macro-F1 averages over the union of observed pairs.
    """
    preds, labels = _as_arrays(all_preds, all_labels)

    true_pairs = Counter()
    pred_pairs = Counter()
    for idx, aspect in enumerate(ASPECT_LIST):
        for polarity in SENTIMENT_CLASSES:
            true_count = int((labels[:, idx] == polarity).sum())
            pred_count = int((preds[:, idx] == polarity).sum())
            if true_count:
                true_pairs[(aspect, polarity)] = true_count
            if pred_count:
                pred_pairs[(aspect, polarity)] = pred_count

    # Macro only averages over pairs observed in the truth or the prediction;
    # pairs absent from both sides are undefined and are skipped.
    keys = sorted(set(true_pairs) | set(pred_pairs))
    tp = sum(min(true_pairs[k], pred_pairs[k]) for k in keys)
    fp = sum(max(0, pred_pairs[k] - true_pairs[k]) for k in keys)
    fn = sum(max(0, true_pairs[k] - pred_pairs[k]) for k in keys)

    micro = _prf_from_counts(tp, fp, fn)
    per_pair = [
        _prf_from_counts(
            min(true_pairs[k], pred_pairs[k]),
            max(0, pred_pairs[k] - true_pairs[k]),
            max(0, true_pairs[k] - pred_pairs[k]),
        )
        for k in keys
    ]
    if per_pair:
        macro = {
            metric: float(np.mean([row[metric] for row in per_pair]))
            for metric in ("precision", "recall", "f1")
        }
    else:
        macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return micro, macro, {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                          "true_pairs": int(sum(true_pairs.values())),
                          "pred_pairs": int(sum(pred_pairs.values())),
                          "observed_pairs": len(keys)}


def per_aspect_table(all_preds, all_labels):
    """
    Per-aspect metrics with the same columns as the historical
    `detailed_metrics_test.csv`, plus pair-F1 columns.
    """
    preds, labels = _as_arrays(all_preds, all_labels)
    rows = []
    for idx, aspect in enumerate(ASPECT_LIST):
        y_true = labels[:, idx]
        y_pred = preds[:, idx]

        true_bin = (y_true > 0).astype(int)
        pred_bin = (y_pred > 0).astype(int)

        mask = (y_true > 0) | (y_pred > 0)
        if mask.sum() > 0:
            asc_f1 = float(f1_score(y_true[mask], y_pred[mask],
                                    labels=SENTIMENT_CLASSES,
                                    average="macro", zero_division=0))
        else:
            asc_f1 = 0.0

        true_pairs = Counter(int(v) for v in y_true if v > 0)
        pred_pairs = Counter(int(v) for v in y_pred if v > 0)
        tp = sum(min(true_pairs[k], pred_pairs[k]) for k in set(true_pairs) | set(pred_pairs))
        fp = sum(max(0, pred_pairs[k] - true_pairs[k]) for k in set(true_pairs) | set(pred_pairs))
        fn = sum(max(0, true_pairs[k] - pred_pairs[k]) for k in set(true_pairs) | set(pred_pairs))
        pair = _prf_from_counts(tp, fp, fn)

        rows.append({
            "Aspect": aspect,
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision_Macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "Recall_Macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "F1_Macro_4Class": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "F1_Weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "ACD_Precision": float(precision_score(true_bin, pred_bin, zero_division=0)),
            "ACD_Recall": float(recall_score(true_bin, pred_bin, zero_division=0)),
            "ACD_F1": float(f1_score(true_bin, pred_bin, zero_division=0)),
            "ASC_Sentiment_F1": asc_f1,
            "Pair_Precision": pair["precision"],
            "Pair_Recall": pair["recall"],
            "Pair_F1": pair["f1"],
            "Support_None": int((y_true == 0).sum()),
            "Support_Negative": int((y_true == 1).sum()),
            "Support_Neutral": int((y_true == 2).sum()),
            "Support_Positive": int((y_true == 3).sum()),
            "Total_Samples": int(len(y_true)),
        })
    return pd.DataFrame(rows)


def aggregate_confusion_matrix(all_preds, all_labels):
    """4x4 confusion matrix pooled over the 10 aspects."""
    preds, labels = _as_arrays(all_preds, all_labels)
    cm = confusion_matrix(labels.flatten(), preds.flatten(),
                          labels=list(range(NUM_CLASSES)))
    return cm


def classification_report_text(all_preds, all_labels, digits=4):
    """Global + per-aspect sklearn classification reports as plain text."""
    preds, labels = _as_arrays(all_preds, all_labels)
    lines = []
    lines.append("=" * 78)
    lines.append("GLOBAL CLASSIFICATION REPORT (all 10 aspects, 4 polarity classes)")
    lines.append("=" * 78)
    lines.append(classification_report(
        labels.flatten(), preds.flatten(),
        labels=list(range(NUM_CLASSES)), target_names=POLARITY_NAMES,
        digits=digits, zero_division=0,
    ))
    for idx, aspect in enumerate(ASPECT_LIST):
        lines.append("")
        lines.append("-" * 78)
        lines.append(f"ASPECT: {aspect}")
        lines.append("-" * 78)
        lines.append(classification_report(
            labels[:, idx], preds[:, idx],
            labels=list(range(NUM_CLASSES)), target_names=POLARITY_NAMES,
            digits=digits, zero_division=0,
        ))
    return "\n".join(lines)


def compute_absa_metrics(all_preds, all_labels):
    """
    Full metric dictionary. Core P/R/F1 values come from
    `metrics_utils.compute_full_metrics`, so they are bit-for-bit identical to
    the historical results. New metrics are added under explicit keys.
    """
    preds, labels = _as_arrays(all_preds, all_labels)
    metrics = compute_full_metrics(preds, labels)

    acd_micro, acd_macro, acd_defined = aspect_detection_metrics(preds, labels)
    sent_micro, sent_macro, sent_support = sentiment_metrics(preds, labels)
    pair_micro, pair_macro, pair_support = pair_f1(preds, labels)

    metrics["acd_micro_precision"] = acd_micro["precision"]
    metrics["acd_micro_recall"] = acd_micro["recall"]
    metrics["acd_micro_f1"] = acd_micro["f1"]
    metrics["acd_macro_precision"] = acd_macro["precision"]
    metrics["acd_macro_recall"] = acd_macro["recall"]
    metrics["acd_macro_f1"] = acd_macro["f1"]
    metrics["acd_defined_aspects"] = acd_defined

    metrics["sentiment_micro_precision"] = sent_micro["precision"]
    metrics["sentiment_micro_recall"] = sent_micro["recall"]
    metrics["sentiment_micro_f1"] = sent_micro["f1"]
    metrics["sentiment_macro_precision"] = sent_macro["precision"]
    metrics["sentiment_macro_recall"] = sent_macro["recall"]
    metrics["sentiment_macro_f1"] = sent_macro["f1"]
    metrics["sentiment_support"] = sent_support

    metrics["pair_micro_precision"] = pair_micro["precision"]
    metrics["pair_micro_recall"] = pair_micro["recall"]
    metrics["pair_micro_f1"] = pair_micro["f1"]
    metrics["pair_macro_precision"] = pair_macro["precision"]
    metrics["pair_macro_recall"] = pair_macro["recall"]
    metrics["pair_macro_f1"] = pair_macro["f1"]
    metrics["pair_support"] = pair_support

    metrics["n_samples"] = int(labels.shape[0])
    metrics["n_label_cells"] = int(labels.size)
    metrics["aspects"] = list(ASPECT_LIST)
    metrics["polarity_classes"] = POLARITY_NAMES
    metrics["label_convention"] = "dataset_loader (0=None, 1=Negative, 2=Neutral, 3=Positive)"
    metrics["zero_division_policy"] = (
        "cell-level metrics: 0 when denominator is 0; "
        "ACD macro: undefined aspects skipped, 1.0 if none defined; "
        "sentiment/pair macro: average over fixed classes/observed pairs with 0 when undefined"
    )
    return metrics


# ---------------------------------------------------------------------------
# Prediction export
# ---------------------------------------------------------------------------
def build_predictions_frame(df, all_preds, all_labels, all_probs):
    """One row per sample; 3 columns per aspect (true, pred, confidence)."""
    preds, labels = _as_arrays(all_preds, all_labels)
    data = {
        "sample_index": np.arange(len(df)),
        "source_index": df["index"].values if "index" in df.columns else np.arange(len(df)),
        "comment": df["comment"].astype(str).values,
    }
    for idx, aspect in enumerate(ASPECT_LIST):
        data[f"true_{aspect}"] = [IDX2POLARITY[int(v)] for v in labels[:, idx]]
        data[f"pred_{aspect}"] = [IDX2POLARITY[int(v)] for v in preds[:, idx]]
        if all_probs is not None:
            probs = np.asarray(all_probs)
            if probs.ndim == 3:
                data[f"prob_{aspect}"] = probs[np.arange(len(preds)), idx, preds[:, idx]]
            else:
                data[f"prob_{aspect}"] = probs[:, idx]
    return pd.DataFrame(data)


def write_predictions_jsonl(path, df, all_preds, all_labels, all_probs):
    preds, labels = _as_arrays(all_preds, all_labels)
    probs = None if all_probs is None else np.asarray(all_probs)
    with open(path, "w", encoding="utf-8") as handle:
        for row in range(len(df)):
            record = {
                "sample_index": int(row),
                "source_index": int(df["index"].iloc[row]) if "index" in df.columns else int(row),
                "comment": str(df["comment"].iloc[row]),
                "true": {a: IDX2POLARITY[int(labels[row, i])]
                         for i, a in enumerate(ASPECT_LIST)},
                "pred": {a: IDX2POLARITY[int(preds[row, i])]
                         for i, a in enumerate(ASPECT_LIST)},
            }
            if probs is not None and probs.ndim == 3:
                record["prob"] = {
                    a: {POLARITY_NAMES[c]: float(probs[row, i, c])
                        for c in range(NUM_CLASSES)}
                    for i, a in enumerate(ASPECT_LIST)
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_error_cases(df, all_preds, all_labels, all_probs):
    preds, labels = _as_arrays(all_preds, all_labels)
    probs = None if all_probs is None else np.asarray(all_probs)
    records = []
    for row in range(len(labels)):
        for idx, aspect in enumerate(ASPECT_LIST):
            y_true = int(labels[row, idx])
            y_pred = int(preds[row, idx])
            if y_true == y_pred:
                continue
            if y_true == 0:
                error_type = "False Positive (aspect over-detection)"
            elif y_pred == 0:
                error_type = "False Negative (aspect missed)"
            else:
                error_type = "Sentiment Misclassification (wrong polarity)"
            records.append({
                "sample_index": row,
                "source_index": int(df["index"].iloc[row]) if "index" in df.columns else row,
                "comment": str(df["comment"].iloc[row]),
                "aspect": aspect,
                "true_label": IDX2POLARITY[y_true],
                "pred_label": IDX2POLARITY[y_pred],
                "confidence": (float(probs[row, idx]) if (probs is not None and probs.ndim == 2)
                               else float(probs[row, idx, y_pred]) if probs is not None
                               else np.nan),
                "error_type": error_type,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Prediction loading (for --from-predictions)
# ---------------------------------------------------------------------------
def _polarity_to_idx(name):
    """Map a polarity name to its integer id, including the None class."""
    text = str(name).strip()
    return 0 if text == "None" else POLARITY2IDX[text]


def load_predictions(path):
    """Rebuild (df, preds, labels, probs) from a CSV or JSONL prediction file."""
    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame({
            "comment": [r["comment"] for r in rows],
            "index": [r.get("source_index", i) for i, r in enumerate(rows)],
        })
        preds = np.array([[_polarity_to_idx(r["pred"][a]) for a in ASPECT_LIST] for r in rows])
        labels = np.array([[_polarity_to_idx(r["true"][a]) for a in ASPECT_LIST] for r in rows])
        probs = None
        if rows and "prob" in rows[0]:
            probs = np.array([[[r["prob"][a][name] for name in POLARITY_NAMES]
                               for a in ASPECT_LIST] for r in rows])
        return df, preds, labels, probs

    # keep_default_na=False is required because the literal label "None" would
    # otherwise be parsed as NaN by pandas.
    df = pd.read_csv(path, keep_default_na=False)
    if "source_index" in df.columns:
        df["index"] = df["source_index"]
    preds = np.column_stack([
        df[f"pred_{a}"].map(_polarity_to_idx).to_numpy() for a in ASPECT_LIST
    ])
    labels = np.column_stack([
        df[f"true_{a}"].map(_polarity_to_idx).to_numpy() for a in ASPECT_LIST
    ])
    prob_cols = [f"prob_{a}" for a in ASPECT_LIST]
    probs = df[prob_cols].to_numpy() if all(c in df.columns for c in prob_cols) else None
    return df, preds, labels, probs


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
def _build_loader(model_type, df, max_len, batch_size, tokenizer_name, vocab):
    import torch
    from torch.utils.data import DataLoader

    if model_type == "phobert":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        encodings = tokenizer(
            df["comment_segmented"].astype(str).tolist(),
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        labels = [parse_label_string(v) for v in df["label"].tolist()]

        class _Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(labels)

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(encodings["input_ids"][idx], dtype=torch.long),
                    "attention_mask": torch.tensor(encodings["attention_mask"][idx], dtype=torch.long),
                    "labels": torch.tensor(labels[idx], dtype=torch.long),
                }

        dataset = _Dataset()
    else:
        from baseline_data import BaselineABSADataset

        dataset = BaselineABSADataset(df, vocab, max_len=max_len)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def run_inference(args):
    import torch
    from model_phobert import MultiHeadPhoBERTForABSA
    from model_baseline import BiLSTMMultiHeadForABSA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.split.capitalize()}_prepared.csv"))

    vocab = None
    if args.model_type == "bilstm":
        if not args.vocab:
            raise ValueError("--vocab is required for the BiLSTM model")
        from baseline_data import load_vocab
        vocab = load_vocab(args.vocab)
        model = BiLSTMMultiHeadForABSA(
            vocab_size=len(vocab),
            num_aspects=len(ASPECT_LIST),
            num_classes=NUM_CLASSES,
        )
    else:
        model = MultiHeadPhoBERTForABSA(
            num_aspects=len(ASPECT_LIST),
            num_classes=NUM_CLASSES,
        )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    loader = _build_loader(args.model_type, df, args.max_len, args.batch_size,
                           args.tokenizer, vocab)

    all_preds, all_labels, all_probs = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)["logits"]
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(probs, dim=-1).cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0)

    meta = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "best_epoch": checkpoint.get("epoch"),
        "best_dev_macro_f1": checkpoint.get("best_dev_macro_f1"),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "latency_per_sample_ms": elapsed / max(1, len(df)) * 1000.0,
        "throughput_samples_sec": len(df) / elapsed if elapsed else 0.0,
    }
    return df, preds, labels, probs, meta


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------
def compare_with_reference(metrics, reference_path, tolerance=1e-9):
    """Compare quality metrics; wall-clock measurements are excluded."""
    excluded = {"latency_per_sample_ms", "throughput_samples_sec"}
    with open(reference_path, "r", encoding="utf-8") as handle:
        reference = json.load(handle)

    def flatten(prefix, value, out):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(f"{prefix}.{key}" if prefix else key, item, out)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[prefix] = float(value)

    ref_flat, new_flat = {}, {}
    flatten("", reference, ref_flat)
    flatten("", metrics, new_flat)

    comparisons = []
    skipped_missing = []
    max_delta = 0.0
    for key, ref_value in ref_flat.items():
        if key in excluded:
            continue
        if key not in new_flat:
            # The source may be a predictions file, which does not carry
            # checkpoint metadata such as best_epoch or total_parameters.
            skipped_missing.append(key)
            continue
        delta = abs(new_flat[key] - ref_value)
        max_delta = max(max_delta, delta)
        comparisons.append({
            "metric": key,
            "reference": ref_value,
            "new": new_flat[key],
            "delta": delta,
            "match": bool(delta <= tolerance),
        })

    mismatches = [row for row in comparisons if not row["match"]]
    return {
        "reference": reference_path,
        "tolerance": tolerance,
        "checked_metrics": len(comparisons),
        "skipped_missing": skipped_missing,
        "max_abs_delta": max_delta,
        "all_match": len(mismatches) == 0,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified ABSA evaluation: predictions, metrics, reports.")
    parser.add_argument("--model_type", choices=["phobert", "bilstm"], default="phobert")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint .pt (omit when using --from-predictions)")
    parser.add_argument("--vocab", type=str, default=None,
                        help="Vocabulary JSON for the BiLSTM baseline")
    parser.add_argument("--from-predictions", type=str, default=None,
                        help="Recompute metrics from an existing predictions CSV/JSONL")
    parser.add_argument("--data_dir", type=str,
                        default="ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model")
    parser.add_argument("--output_dir", type=str, default="ABSA_Project/results/evaluation")
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Dev", "Test"])
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tokenizer", type=str, default="vinai/phobert-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--reference-json", type=str, default=None,
                        help="Optional previous metrics JSON for a regression check")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args(argv)


def main(argv=None):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix or f"{args.model_type}_absa_{args.split.lower()}"

    if args.from_predictions:
        df, preds, labels, probs = load_predictions(args.from_predictions)
        meta = {
            "model_type": args.model_type,
            "checkpoint": args.from_predictions,
            "split": args.split,
            "max_len": None,
            "device": "n/a (from predictions)",
            "seed": args.seed,
        }
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --from-predictions is used")
        df, preds, labels, probs, meta = run_inference(args)

    metrics = compute_absa_metrics(preds, labels)
    metrics.update({k: v for k, v in meta.items() if k not in metrics})

    if args.reference_json:
        metrics["regression_check"] = compare_with_reference(
            metrics, args.reference_json, args.tolerance)
        if not metrics["regression_check"]["all_match"]:
            print("[WARN] Regression check found mismatches; see regression_check in the JSON.")
        else:
            print(f"[OK] Regression check passed ({args.reference_json})")

    metrics_path = os.path.join(args.output_dir, f"{prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    per_aspect = per_aspect_table(preds, labels)
    per_aspect_path = os.path.join(args.output_dir, f"{prefix}_per_aspect.csv")
    per_aspect.to_csv(per_aspect_path, index=False, encoding="utf-8-sig")

    report_path = os.path.join(args.output_dir, f"{prefix}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(classification_report_text(preds, labels))

    cm = aggregate_confusion_matrix(preds, labels)
    cm_path = os.path.join(args.output_dir, f"{prefix}_confusion_matrix.csv")
    pd.DataFrame(cm, index=[f"true_{n}" for n in POLARITY_NAMES],
                 columns=[f"pred_{n}" for n in POLARITY_NAMES]).to_csv(
        cm_path, encoding="utf-8-sig")

    predictions_path = os.path.join(args.output_dir, f"{prefix}_predictions.csv")
    build_predictions_frame(df, preds, labels, probs).to_csv(
        predictions_path, index=False, encoding="utf-8-sig")

    jsonl_path = os.path.join(args.output_dir, f"{prefix}_predictions.jsonl")
    write_predictions_jsonl(jsonl_path, df, preds, labels, probs)

    errors = build_error_cases(df, preds, labels, probs)
    errors_path = os.path.join(args.output_dir, f"{prefix}_errors.csv")
    errors.to_csv(errors_path, index=False, encoding="utf-8-sig")

    print(f"=== {args.model_type.upper()} | {args.split} | max_len={args.max_len} ===")
    for key in ("micro_f1", "macro_f1", "sentiment_macro_f1", "sentiment_micro_f1",
                "acd_f1", "acd_macro_f1", "pair_micro_f1", "pair_macro_f1",
                "exact_match"):
        print(f"  {key:24s} {metrics.get(key):.6f}")
    print(f"  samples={metrics['n_samples']}  label_cells={metrics['n_label_cells']}")
    print(f"  predictions: {predictions_path}")
    print(f"  metrics:     {metrics_path}")
    if args.reference_json:
        check = metrics["regression_check"]
        print(f"  regression:  max|delta|={check['max_abs_delta']:.3e} "
              f"all_match={check['all_match']}")

    return metrics


if __name__ == "__main__":
    main()
