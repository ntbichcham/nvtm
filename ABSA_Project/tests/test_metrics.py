"""
test_metrics.py
Hand-checked unit tests for the ABSA metric suite.

Every expected value below is computed by hand from a 10-aspect, 4-class label
vector (0=None, 1=Negative, 2=Neutral, 3=Positive). The tests also import the
label order from `dataset_loader`, which is the same convention used to train
PhoBERT and the BiLSTM baseline.

Run either with pytest:
    pytest ABSA_Project/tests/test_metrics.py -v

or directly (no pytest required):
    python ABSA_Project/tests/test_metrics.py

Author: Nhom 2 - Thanh vien 2
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset_loader import (  # noqa: E402
    ASPECT_LIST,
    ASPECT2IDX,
    POLARITY2IDX,
    IDX2POLARITY,
    parse_label_string,
)
from evaluate_absa import (  # noqa: E402
    aspect_detection_metrics,
    sentiment_metrics,
    pair_f1,
    compute_absa_metrics,
    per_aspect_table,
    aggregate_confusion_matrix,
    build_predictions_frame,
    write_predictions_jsonl,
    load_predictions,
)
from metrics_utils import compute_full_metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_case(true_aspects, pred_aspects, n_samples=1):
    """Build (labels, preds) arrays of shape (n, 10) from {ASPECT: polarity}."""
    labels = np.zeros((n_samples, len(ASPECT_LIST)), dtype=np.int64)
    preds = np.zeros((n_samples, len(ASPECT_LIST)), dtype=np.int64)
    for row in range(n_samples):
        for aspect, polarity in true_aspects.items():
            labels[row, ASPECT2IDX[aspect]] = POLARITY2IDX[polarity]
        for aspect, polarity in pred_aspects.items():
            preds[row, ASPECT2IDX[aspect]] = POLARITY2IDX[polarity]
    return preds, labels


def assert_close(actual, expected, name, tol=1e-12):
    if not np.isclose(actual, expected, atol=tol, rtol=0):
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


# ---------------------------------------------------------------------------
# 1. Perfect prediction
# ---------------------------------------------------------------------------
def test_perfect_prediction():
    preds, labels = make_case(
        {"CAMERA": "Positive"}, {"CAMERA": "Positive"}
    )
    m = compute_absa_metrics(preds, labels)

    assert_close(m["exact_match"], 1.0, "exact_match")
    assert_close(m["micro_f1"], 1.0, "micro_f1")
    # macro-F1 follows scikit-learn and averages over the classes present in
    # y_true U y_pred: here only {None, Positive}, both with F1=1.
    assert_close(m["macro_f1"], 1.0, "macro_f1")
    assert_close(m["acd_micro_f1"], 1.0, "acd_micro_f1")
    assert_close(m["acd_macro_f1"], 1.0, "acd_macro_f1")
    # Macro over Negative/Neutral/Positive always divides by 3.
    assert_close(m["sentiment_macro_f1"], 1 / 3, "sentiment_macro_f1")
    assert_close(m["sentiment_micro_f1"], 1.0, "sentiment_micro_f1")
    assert_close(m["pair_micro_f1"], 1.0, "pair_micro_f1")
    assert_close(m["pair_macro_f1"], 1.0, "pair_macro_f1")


# ---------------------------------------------------------------------------
# 2. Missed aspect
# ---------------------------------------------------------------------------
def test_missed_aspect():
    preds, labels = make_case(
        {"BATTERY": "Positive"}, {}
    )
    m = compute_absa_metrics(preds, labels)

    # 9 correct None cells, 1 missed Positive: TP=9, FP=1, FN=1
    assert_close(m["micro_precision"], 0.9, "micro_precision")
    assert_close(m["micro_recall"], 0.9, "micro_recall")
    assert_close(m["micro_f1"], 0.9, "micro_f1")
    # None F1 = 18/19; sklearn averages macro over the classes present
    # in y_true U y_pred, which here are {None, Positive} -> divide by 2.
    assert_close(m["macro_f1"], (18 / 19) / 2, "macro_f1")
    assert_close(m["exact_match"], 0.0, "exact_match")
    assert_close(m["acd_micro_f1"], 0.0, "acd_micro_f1")
    assert_close(m["acd_macro_f1"], 0.0, "acd_macro_f1")
    assert_close(m["sentiment_macro_f1"], 0.0, "sentiment_macro_f1")
    assert_close(m["sentiment_micro_f1"], 0.0, "sentiment_micro_f1")
    assert_close(m["pair_micro_f1"], 0.0, "pair_micro_f1")


# ---------------------------------------------------------------------------
# 3. Wrong sentiment (aspect still detected)
# ---------------------------------------------------------------------------
def test_wrong_sentiment():
    preds, labels = make_case(
        {"PRICE": "Negative"}, {"PRICE": "Neutral"}
    )
    m = compute_absa_metrics(preds, labels)

    # TP=9 (None), FP=1 (Neutral), FN=1 (Negative)
    assert_close(m["micro_f1"], 0.9, "micro_f1")
    # None F1=1, Negative=0, Neutral=0; macro over the three present classes.
    assert_close(m["macro_f1"], 1 / 3, "macro_f1")
    assert_close(m["exact_match"], 0.0, "exact_match")
    # The aspect is detected, only its polarity is wrong.
    assert_close(m["acd_micro_f1"], 1.0, "acd_micro_f1")
    assert_close(m["acd_macro_f1"], 1.0, "acd_macro_f1")
    assert_close(m["sentiment_macro_f1"], 0.0, "sentiment_macro_f1")
    assert_close(m["sentiment_micro_f1"], 0.0, "sentiment_micro_f1")
    assert_close(m["pair_micro_f1"], 0.0, "pair_micro_f1")


# ---------------------------------------------------------------------------
# 4. Sample with no aspect at all
# ---------------------------------------------------------------------------
def test_no_aspect_sample():
    preds, labels = make_case({}, {})
    m = compute_absa_metrics(preds, labels)

    assert_close(m["exact_match"], 1.0, "exact_match")
    assert_close(m["micro_f1"], 1.0, "micro_f1")
    # Only class None is present, so sklearn macro over present classes is 1.0
    # (the strict 4-class macro is reported separately in the per-class table).
    assert_close(m["macro_f1"], 1.0, "macro_f1")
    # Documented convention: cell-level ACD follows zero_division=0 (0.0),
    # while ACD macro over undefined aspects is vacuously 1.0.
    assert_close(m["acd_micro_f1"], 0.0, "acd_micro_f1 (zero_division=0)")
    assert_close(m["acd_macro_f1"], 1.0, "acd_macro_f1 (undefined -> 1.0)")
    assert_close(m["sentiment_macro_f1"], 0.0, "sentiment_macro_f1")
    assert_close(m["pair_micro_f1"], 0.0, "pair_micro_f1")
    assert m["acd_defined_aspects"] == 0


# ---------------------------------------------------------------------------
# 5. Multiple aspects, one polarity error
# ---------------------------------------------------------------------------
def test_multiple_aspects():
    preds, labels = make_case(
        {"CAMERA": "Positive", "BATTERY": "Negative"},
        {"CAMERA": "Positive", "BATTERY": "Positive"},
    )
    m = compute_absa_metrics(preds, labels)

    # TP=9, FP=1, FN=1
    assert_close(m["micro_f1"], 0.9, "micro_f1")
    # Present classes {None, Negative, Positive}: (1 + 0 + 2/3) / 3
    assert_close(m["macro_f1"], (1 + 0 + 2 / 3) / 3, "macro_f1")
    assert_close(m["exact_match"], 0.0, "exact_match")
    assert_close(m["acd_micro_f1"], 1.0, "acd_micro_f1")
    assert_close(m["acd_macro_f1"], 1.0, "acd_macro_f1")
    assert_close(m["sentiment_micro_f1"], 0.5, "sentiment_micro_f1")
    assert_close(m["sentiment_macro_f1"], (2 / 3) / 3, "sentiment_macro_f1")
    # True pairs: 1 correct + 1 wrong; predicted: 1 correct + 1 wrong
    assert_close(m["pair_micro_f1"], 0.5, "pair_micro_f1")
    # Macro over observed pairs: CAMERA-Pos=1, BATTERY-Neg=0, BATTERY-Pos=0
    assert_close(m["pair_macro_f1"], 1 / 3, "pair_macro_f1")

    table = per_aspect_table(preds, labels).set_index("Aspect")
    assert_close(table.loc["CAMERA", "Pair_F1"], 1.0, "CAMERA pair F1")
    assert_close(table.loc["BATTERY", "Pair_F1"], 0.0, "BATTERY pair F1")
    assert_close(table.loc["CAMERA", "ACD_F1"], 1.0, "CAMERA ACD F1")
    assert_close(table.loc["BATTERY", "ACD_F1"], 1.0, "BATTERY ACD F1")


# ---------------------------------------------------------------------------
# 6. Confusion matrix is a partition of all label cells
# ---------------------------------------------------------------------------
def test_confusion_matrix_sums_to_cells():
    preds = np.array([
        [3, 1, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 1, 0, 0, 0, 0],
    ])
    labels = np.array([
        [3, 1, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    ])
    cm = aggregate_confusion_matrix(preds, labels)
    assert cm.shape == (4, 4)
    assert int(cm.sum()) == labels.size == 20


# ---------------------------------------------------------------------------
# 7. Legacy metrics and the new suite agree on non-degenerate data
# ---------------------------------------------------------------------------
def test_legacy_compute_full_metrics_agrees():
    preds, labels = make_case(
        {"CAMERA": "Positive", "BATTERY": "Negative", "PRICE": "Neutral"},
        {"CAMERA": "Positive", "BATTERY": "Positive", "PRICE": "Neutral"},
    )
    legacy = compute_full_metrics(preds, labels)
    new = compute_absa_metrics(preds, labels)
    for key in ("micro_precision", "micro_recall", "micro_f1",
                "macro_precision", "macro_recall", "macro_f1",
                "weighted_f1", "exact_match", "sentiment_macro_f1",
                "acd_precision", "acd_recall", "acd_f1"):
        assert_close(new[key], legacy[key], key)


# ---------------------------------------------------------------------------
# 8. Label convention matches the training pipeline
# ---------------------------------------------------------------------------
def test_label_convention_matches_training_code():
    assert ASPECT_LIST == [
        "BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL",
        "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE",
    ]
    assert IDX2POLARITY == {0: "None", 1: "Negative", 2: "Neutral", 3: "Positive"}
    vector = parse_label_string("{CAMERA#Positive};{BATTERY#Negative};")
    assert vector[ASPECT2IDX["CAMERA"]] == 3
    assert vector[ASPECT2IDX["BATTERY"]] == 1
    assert vector.sum() == 3 + 1


# ---------------------------------------------------------------------------
# 9. Prediction files round-trip through the loader
# ---------------------------------------------------------------------------
def test_prediction_export_roundtrip():
    preds, labels = make_case(
        {"CAMERA": "Positive", "BATTERY": "Negative"},
        {"CAMERA": "Positive", "BATTERY": "Positive"},
    )
    probs = np.zeros((1, len(ASPECT_LIST), 4), dtype=np.float32)
    probs[:, :, 0] = 1.0
    for idx in range(len(ASPECT_LIST)):
        probs[0, idx, preds[0, idx]] = 0.9
        probs[0, idx, 0] = 0.1 if preds[0, idx] != 0 else 1.0

    df = pd.DataFrame({
        "index": [17],
        "comment": ["pin tốt nhưng camera lỗi"],
    })

    with tempfile.TemporaryDirectory() as tmp:
        frame = build_predictions_frame(df, preds, labels, probs)
        csv_path = os.path.join(tmp, "pred.csv")
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

        jsonl_path = os.path.join(tmp, "pred.jsonl")
        write_predictions_jsonl(jsonl_path, df, preds, labels, probs)

        for path in (csv_path, jsonl_path):
            loaded_df, loaded_preds, loaded_labels, loaded_probs = load_predictions(path)
            assert loaded_preds.shape == (1, len(ASPECT_LIST))
            np.testing.assert_array_equal(loaded_preds, preds)
            np.testing.assert_array_equal(loaded_labels, labels)
            assert int(loaded_df["index"].iloc[0]) == 17
            if loaded_probs is not None:
                expected = (1, len(ASPECT_LIST), 4) if path.endswith(".jsonl") \
                    else (1, len(ASPECT_LIST))
                assert loaded_probs.shape == expected


# ---------------------------------------------------------------------------
# Standalone runner (works without pytest)
# ---------------------------------------------------------------------------
def _run_standalone():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {exc}")
            raise
        else:
            passed += 1
            print(f"[PASS] {name}")
    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_standalone()
