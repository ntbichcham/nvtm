"""
preprocess.py
Reproducible preprocessing for UIT-ViSFD (ABSA + PhoBERT).

Pipeline
--------
1. Read the original split CSVs (never modified).
2. Word-segment `comment`. The default backend `auto` reuses the existing
   `*_segmented.csv` files produced by VnCoreNLP, because they are part of the
   released artifacts and the comparison must use exactly the same tokenisation.
   `vncorenlp` forces a fresh segmentation (requires Java + py_vncorenlp).
3. Compute the PhoBERT token length of every segmented comment with
   `add_special_tokens=True` (the stored `token_length` therefore includes
   <s> and </s>).
4. Parse `date_time` into `date_parsed` (ISO) and `date_status`
   (`Absolute_Date` / `Relative_Time`).
5. Write `*_prepared.csv`, token statistics, segmentation report and handoff
   validation reports into the reproduction folder.

The output directory defaults to the `output.reproduced_dir` value in
configs/preprocessing.yaml, so the official files under
`data/processed/UIT-ViSFD` are never overwritten.

Author: Nhom 2 - Thanh vien 1
"""

import os
import sys
import argparse
import shutil

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "ABSA_Project", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from check_dataset import (  # noqa: E402
    load_config,
    parse_date,
    resolve,
)

PREPARED_COLUMNS = [
    "index", "comment", "n_star", "date_time", "label",
    "comment_segmented", "token_length", "date_parsed", "date_status",
    "dataset_split",
]

TOKEN_STAT_COLUMNS = [
    "Dataset", "Total_Comments", "Min_Tokens", "Max_Tokens", "Mean_Tokens",
    "Median_Tokens", "P90", "P95", "P99",
    "Within_128", "Within_128_Percent", "Exceed_128", "Exceed_128_Percent",
    "Within_256", "Within_256_Percent", "Exceed_256", "Exceed_256_Percent",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def load_teencode_rules(path):
    if not os.path.exists(path):
        return []
    rules = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    pairs = []
    for _, row in rules.iterrows():
        pattern = str(row["pattern"]).strip()
        replacement = str(row["replacement"]).strip()
        if pattern:
            pairs.append((pattern, replacement))
    return pairs


def apply_teencode(text, rules):
    """Whole-token, case-insensitive replacement on a segmented comment."""
    if not rules or not isinstance(text, str):
        return text
    mapping = {pattern.lower(): replacement for pattern, replacement in rules}
    tokens = []
    for token in text.split():
        tokens.append(mapping.get(token.lower(), token))
    return " ".join(tokens)


def segment_with_vncorenlp(texts, config):
    try:
        import py_vncorenlp  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The 'vncorenlp' backend needs py_vncorenlp and Java. "
            "Use segmentation.backend: auto/reuse to reuse *_segmented.csv."
        ) from exc

    model_dir = resolve(config["segmentation"]["vncorenlp_dir"])
    os.makedirs(model_dir, exist_ok=True)
    segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=model_dir)
    outputs = []
    for text in texts:
        sentences = segmenter.word_segment(str(text)) if str(text).strip() else []
        outputs.append(" ".join(sentences).strip())
    return outputs


def phobert_token_lengths(texts, tokenizer_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    encodings = tokenizer(
        [str(t) for t in texts],
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return [len(ids) for ids in encodings["input_ids"]]


def build_token_statistics(frame, split):
    values = frame["token_length"].astype(int)
    total = int(len(values))
    row = {
        "Dataset": split,
        "Total_Comments": total,
        "Min_Tokens": int(values.min()) if total else 0,
        "Max_Tokens": int(values.max()) if total else 0,
        "Mean_Tokens": round(float(values.mean()), 2) if total else 0.0,
        "Median_Tokens": round(float(values.median()), 2) if total else 0.0,
        "P90": round(float(np.percentile(values, 90)), 2) if total else 0.0,
        "P95": round(float(np.percentile(values, 95)), 2) if total else 0.0,
        "P99": round(float(np.percentile(values, 99)), 2) if total else 0.0,
        "Within_128": int((values <= 128).sum()),
        "Within_128_Percent": round(float((values <= 128).mean() * 100), 2) if total else 0.0,
        "Exceed_128": int((values > 128).sum()),
        "Exceed_128_Percent": round(float((values > 128).mean() * 100), 2) if total else 0.0,
        "Within_256": int((values <= 256).sum()),
        "Within_256_Percent": round(float((values <= 256).mean() * 100), 2) if total else 0.0,
        "Exceed_256": int((values > 256).sum()),
        "Exceed_256_Percent": round(float((values > 256).mean() * 100), 2) if total else 0.0,
    }
    return row


def validation_row(split, original, prepared):
    """Row for handoff_validation_report.csv."""
    original_rows = len(original)
    prepared_rows = len(prepared)

    comment_ok = (
        prepared_rows == original_rows
        and (prepared["comment"].astype(str).values == original["comment"].astype(str).values).all()
    )
    label_ok = (
        prepared_rows == original_rows
        and (prepared["label"].astype(str).values == original["label"].astype(str).values).all()
    )
    index_ok = (
        prepared_rows == original_rows
        and (prepared["index"].values == original["index"].values).all()
    )
    segmented_exists = "comment_segmented" in prepared.columns
    empty_segmented = int(
        (prepared["comment_segmented"].astype(str).str.strip() == "").sum()
    ) if segmented_exists else 0
    token_exists = "token_length" in prepared.columns
    missing_token = int(prepared["token_length"].isna().sum()) if token_exists else -1
    date_cols_exist = all(c in prepared.columns for c in ("date_parsed", "date_status"))
    missing_columns = [c for c in PREPARED_COLUMNS if c not in prepared.columns]
    passed = all([
        prepared_rows == original_rows,
        comment_ok,
        label_ok,
        index_ok,
        segmented_exists,
        empty_segmented == 0,
        token_exists,
        missing_token == 0,
        date_cols_exist,
        not missing_columns,
    ])
    return {
        "Dataset": split,
        "Original_Rows": original_rows,
        "Prepared_Rows": prepared_rows,
        "Row_Count_Match": prepared_rows == original_rows,
        "Original_Comment_Preserved": bool(comment_ok),
        "Original_Label_Preserved": bool(label_ok),
        "Index_Order_Preserved": bool(index_ok),
        "Segmented_Column_Exists": bool(segmented_exists),
        "Empty_Segmented_Text": empty_segmented,
        "Token_Column_Exists": bool(token_exists),
        "Missing_Token_Length": missing_token,
        "Date_Columns_Exist": bool(date_cols_exist),
        "Missing_Columns": ";".join(missing_columns) if missing_columns else "",
        "Validation_Passed": bool(passed),
    }


def additional_validation_row(split, original, prepared):
    rating_ok = (prepared["n_star"].values == original["n_star"].values).all()
    date_ok = (prepared["date_time"].astype(str).values
               == original["date_time"].astype(str).values).all()
    split_ok = (prepared["dataset_split"] == split).all()
    token_ok = prepared["token_length"].notna().all() and \
        (prepared["token_length"].astype(int) >= 1).all()
    return {
        "Dataset": split,
        "Rating_Preserved": bool(rating_ok),
        "Date_Time_Preserved": bool(date_ok),
        "Dataset_Split_Correct": bool(split_ok),
        "Token_Length_Valid": bool(token_ok),
        "All_Checks_Passed": bool(rating_ok and date_ok and split_ok and token_ok),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Reproduce UIT-ViSFD preprocessing")
    parser.add_argument("--config", default=os.path.join(
        ROOT, "ABSA_Project", "configs", "preprocessing.yaml"))
    parser.add_argument("--out-dir", default=None,
                        help="Override output.reproduced_dir")
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args(argv)

    config = load_config(resolve(args.config))
    cols = config["columns"]
    splits = args.splits or config["splits"]
    out_dir = resolve(args.out_dir or config["output"]["reproduced_dir"])
    prepared_dir = os.path.join(out_dir, "prepared_for_model")
    os.makedirs(prepared_dir, exist_ok=True)

    tokenizer_name = config["tokenizer"]["name"]
    teencode_cfg = config.get("teencode", {}) or {}
    teencode_enabled = bool(teencode_cfg.get("enabled", False))
    rules = load_teencode_rules(resolve(teencode_cfg.get("rules_file",
                                                         "configs/teencode_rules.csv"))) \
        if teencode_enabled else []
    if teencode_enabled:
        print(f"Teencode enabled with {len(rules)} rules")

    backend = (config.get("segmentation", {}) or {}).get("backend", "auto")
    reuse_dir = resolve((config.get("segmentation", {}) or {}).get(
        "reuse_dir", "ABSA_Project/data/processed/UIT-ViSFD"))

    prepared_frames = {}
    token_rows = []
    validation_rows = []
    additional_rows = []
    segmentation_rows = []
    teencode_audit = []

    for split in splits:
        original_path = os.path.join(
            resolve(config["data"]["original_dir"]), f"{split}.csv")
        original = read_csv(original_path)
        print(f"Loaded {split}: {len(original):,} rows from {original_path}")

        reuse_path = os.path.join(reuse_dir, f"{split}_segmented.csv")
        used_backend = "reuse"
        if backend in ("auto", "reuse") and os.path.exists(reuse_path):
            segmented = read_csv(reuse_path)
            if len(segmented) != len(original):
                raise ValueError(
                    f"{reuse_path} has {len(segmented)} rows, expected {len(original)}")
            segmented_text = segmented[cols["segmented"]].astype(str).tolist()
        elif backend == "reuse":
            raise SystemExit(
                f"segmentation.backend=reuse but {reuse_path} does not exist")
        else:
            used_backend = "vncorenlp"
            segmented_text = segment_with_vncorenlp(
                original[cols["text"]].tolist(), config)

        if teencode_enabled and rules:
            before = segmented_text
            segmented_text = [apply_teencode(t, rules) for t in segmented_text]
            for idx, (old, new) in enumerate(zip(before, segmented_text)):
                if old != new:
                    teencode_audit.append({
                        "Dataset": split,
                        "Index": int(original["index"].iloc[idx]),
                        "Before": old,
                        "After": new,
                    })

        # Persist the segmented file used by this run.
        segmented_out = pd.DataFrame({
            "index": original["index"],
            "comment": original[cols["text"]],
            "n_star": original[cols["rating"]],
            "date_time": original[cols["date"]],
            "label": original[cols["label"]],
            "comment_segmented": segmented_text,
        })
        segmented_out.to_csv(os.path.join(out_dir, f"{split}_segmented.csv"),
                             index=False, encoding="utf-8-sig")

        if used_backend == "reuse" and os.path.exists(reuse_path) and not teencode_enabled:
            # Byte-identical copy keeps the released artifacts reproducible.
            try:
                shutil.copyfile(reuse_path, os.path.join(out_dir, f"{split}_segmented.csv"))
            except OSError:
                pass

        lengths = phobert_token_lengths(segmented_text, tokenizer_name)
        parsed_dates, statuses = [], []
        for value in original[cols["date"]]:
            parsed, status = parse_date(value)
            parsed_dates.append(parsed)
            statuses.append(status)

        prepared = pd.DataFrame({
            "index": original["index"],
            "comment": original[cols["text"]],
            "n_star": original[cols["rating"]],
            "date_time": original[cols["date"]],
            "label": original[cols["label"]],
            "comment_segmented": segmented_text,
            "token_length": lengths,
            "date_parsed": parsed_dates,
            "date_status": statuses,
            "dataset_split": split,
        })[PREPARED_COLUMNS]

        prepared_path = os.path.join(prepared_dir, f"{split}_prepared.csv")
        prepared.to_csv(prepared_path, index=False, encoding="utf-8-sig")
        prepared_frames[split] = prepared

        for index_value, length in zip(prepared["index"], prepared["token_length"]):
            token_rows.append({
                "Dataset": split,
                "Index": int(index_value),
                "Token_Length": int(length),
            })

        empty_segmented = int((prepared["comment_segmented"].astype(str).str.strip() == "").sum())
        segmentation_rows.append({
            "Dataset": f"{split}.csv",
            "Original_Rows": len(original),
            "Processed_Rows": len(prepared),
            "Empty_Segmented_Text": empty_segmented,
            "Output_File": os.path.join(out_dir, f"{split}_segmented.csv"),
        })

        validation_rows.append(validation_row(split, original, prepared))
        additional_rows.append(additional_validation_row(split, original, prepared))
        print(f"  prepared {split}: {len(prepared):,} rows -> {prepared_path} "
              f"(backend={used_backend})")

    # Token statistics and details (split order preserved).
    token_details = pd.DataFrame(token_rows, columns=["Dataset", "Index", "Token_Length"])
    token_details.to_csv(os.path.join(out_dir, "phobert_token_details.csv"),
                         index=False, encoding="utf-8-sig")

    stats_rows = [build_token_statistics(prepared_frames[s], s) for s in splits]
    token_stats = pd.DataFrame(stats_rows, columns=TOKEN_STAT_COLUMNS)
    token_stats.to_csv(os.path.join(out_dir, "phobert_token_statistics.csv"),
                       index=False, encoding="utf-8-sig")

    pd.DataFrame(segmentation_rows).to_csv(
        os.path.join(out_dir, "segmentation_report.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame(validation_rows).to_csv(
        os.path.join(out_dir, "handoff_validation_report.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame(additional_rows).to_csv(
        os.path.join(out_dir, "handoff_additional_validation.csv"),
        index=False, encoding="utf-8-sig")

    # invalid_labels.csv is intentionally empty but must exist with a BOM.
    with open(os.path.join(out_dir, "invalid_labels.csv"), "w",
              encoding="utf-8-sig", newline="") as handle:
        handle.write("\n")

    if teencode_audit:
        pd.DataFrame(teencode_audit).to_csv(
            resolve(teencode_cfg["audit_file"]), index=False, encoding="utf-8-sig")

    print(f"\nPreprocessing written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
