"""
summarize_runs.py
Builds the final comparison table from independent runs.

Protocol
--------
* Each independent run contributes exactly ONE Test result: the metric set
  produced by the best checkpoint (selected on Dev Macro F1).
* Nothing is averaged across epochs inside a run.
* The table reports mean +/- sample standard deviation (ddof=1) across seeds.

Inputs
------
* New runs   : results/experiments_2026/<run>/test_results_<run>.json
* Legacy runs: results/evaluation/test_results_*.json (5-epoch and 10-epoch
  results that must not be overwritten).

Outputs (--out_dir)
-------------------
comparison_table.csv
comparison_table.md
runs_long.csv
summary.json

Author: Nhom 2 - Thanh vien 3
"""

import os
import re
import sys
import json
import math
import argparse
import statistics
from collections import OrderedDict
from datetime import datetime

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


LEGACY_FILES = {
    "test_results_phobert_official.json": ("phobert_5ep", 42),
    "test_results_official_seed123.json": ("phobert_5ep", 123),
    "test_results_official_seed2024.json": ("phobert_5ep", 2024),
    "test_results_official_ep10_seed42.json": ("phobert_10ep", 42),
    "test_results_bilstm_official.json": ("bilstm_random_ref", 42),
    "test_results_bilstm_seed123.json": ("bilstm_random_ref", 123),
    "test_results_bilstm_seed2024.json": ("bilstm_random_ref", 2024),
}

GROUP_LABELS = {
    "bilstm_ft": "Bi-LSTM + fastText",
    "phobert_5ep": "Multi-Head PhoBERT 5 epochs",
    "phobert_10ep": "Multi-Head PhoBERT 10 epochs",
    "phobert_es20": "Multi-Head PhoBERT ES 20/3 (phụ)",
    "bilstm_random_ref": "BiLSTM random embedding (tham khảo)",
}

GROUP_ORDER = [
    "bilstm_ft",
    "phobert_5ep",
    "phobert_10ep",
    "phobert_es20",
    "bilstm_random_ref",
]

REQUIRED_GROUPS = ["bilstm_ft", "phobert_5ep", "phobert_10ep"]
REQUIRED_RUNS = 3


def _metric(payload, *keys):
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and value is not None:
            value = float(value)
            if not math.isnan(value):
                return value
    return None


def _seed_from_name(name):
    match = re.search(r"seed(\d+)", name)
    return int(match.group(1)) if match else None


def _last_epoch(log_dir, name, payload):
    explicit = payload.get("last_epoch")
    if isinstance(explicit, int):
        return explicit

    candidates = [
        os.path.join(log_dir, f"train_log_{name}.csv"),
        os.path.join(os.path.dirname(log_dir), f"train_log_{name}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return int(len(pd.read_csv(path)))
            except Exception:
                return None
    return None


def _new_group(name):
    if name.startswith("phobert_10ep_"):
        return "phobert_10ep"
    if name.startswith("phobert_es20_"):
        return "phobert_es20"
    if name.startswith("bilstm_ft_"):
        return "bilstm_ft"
    return None


def _row_from_payload(group, seed, name, payload, log_dir):
    micro = _metric(payload, "test_micro_f1", "micro_f1")
    macro = _metric(
        payload, "test_macro_f1_4class", "test_macro_f1", "macro_f1"
    )
    asc = _metric(payload, "test_sentiment_macro_f1", "sentiment_macro_f1")
    acd = _metric(payload, "test_acd_f1", "acd_f1")
    em = _metric(payload, "test_exact_match", "exact_match")
    pair = _metric(payload, "test_pair_macro_f1", "pair_macro_f1")
    best_epoch = payload.get("best_epoch")
    last_epoch = _last_epoch(log_dir, name, payload)
    gap = (
        int(last_epoch) - int(best_epoch)
        if last_epoch is not None and best_epoch is not None
        else None
    )

    return {
        "group": group,
        "label": GROUP_LABELS.get(group, group),
        "experiment": name,
        "seed": seed,
        "n_runs": 1,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "gap_epochs": gap,
        "micro_f1": micro,
        "macro_f1": macro,
        "asc_f1": asc,
        "acd_f1": acd,
        "pair_macro_f1": pair,
        "exact_match": em,
        "source": payload.get("_source_path"),
    }


def collect_runs(root_dir, legacy_dir):
    rows = []
    seen = set()

    # 1) New runs (experiments_2026)
    if root_dir and os.path.isdir(root_dir):
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if not (filename.startswith("test_results_") and filename.endswith(".json")):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                name = payload.get("experiment_name") or filename[len("test_results_") : -len(".json")]
                group = _new_group(name)
                if group is None:
                    continue
                seed = payload.get("seed")
                if seed is None:
                    seed = _seed_from_name(name)
                key = (group, seed)
                if key in seen:
                    print(f"[WARN] duplicate run ignored: {key} ({path})")
                    continue
                seen.add(key)
                payload["_source_path"] = path
                rows.append(_row_from_payload(group, seed, name, payload, dirpath))

    # 2) Legacy runs (never modified)
    if legacy_dir and os.path.isdir(legacy_dir):
        for filename, (group, seed) in LEGACY_FILES.items():
            path = os.path.join(legacy_dir, filename)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            name = payload.get("experiment_name") or filename[len("test_results_") : -len(".json")]
            key = (group, seed)
            if key in seen:
                print(f"[WARN] legacy duplicate ignored: {key} ({path})")
                continue
            seen.add(key)
            payload["_source_path"] = path
            rows.append(_row_from_payload(group, seed, name, payload, legacy_dir))

    return rows


def _mean_std(values):
    cleaned = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        cleaned.append(value)
    values = cleaned
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std


def _display(mean, std, scale=100.0):
    if mean is None:
        return "—"
    if std is None:
        return f"{mean * scale:.4f}"
    return f"{mean * scale:.4f} ± {std * scale:.4f}"


def build_table(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No runs found. Check --root/--legacy_dir.")

    group_rows = []
    for group in GROUP_ORDER:
        sub = df[df["group"] == group]
        if sub.empty:
            continue

        stats = {
            "group": group,
            "label": GROUP_LABELS.get(group, group),
            "n_runs": int(len(sub)),
            "seeds": ", ".join(
                str(int(s)) for s in sorted(sub["seed"].dropna().unique())
            ),
        }
        for metric in ["micro_f1", "macro_f1", "asc_f1", "acd_f1", "pair_macro_f1", "exact_match"]:
            mean, std = _mean_std(sub[metric].tolist())
            stats[f"{metric}_mean"] = mean
            stats[f"{metric}_std"] = std
            stats[f"{metric}_display"] = _display(mean, std)

        for metric in ["best_epoch", "last_epoch", "gap_epochs"]:
            mean, std = _mean_std(sub[metric].tolist())
            stats[f"{metric}_mean"] = mean
            stats[f"{metric}_std"] = std

        group_rows.append(stats)

    return df, pd.DataFrame(group_rows)


def write_markdown(table, path):
    columns = [
        ("label", "Mô hình"),
        ("n_runs", "Số run"),
        ("micro_f1_display", "Micro F1 (%)"),
        ("macro_f1_display", "Macro F1 (%)"),
        ("asc_f1_display", "ASC Macro F1 (%)"),
        ("acd_f1_display", "ACD F1 (%)"),
        ("exact_match_display", "Exact Match (%)"),
        ("best_epoch_mean", "Best epoch TB"),
    ]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in table.iterrows():
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                cells.append("—")
            elif key == "n_runs":
                cells.append(str(int(value)))
            elif key == "best_epoch_mean":
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Summarise independent ABSA runs (mean +/- std across seeds)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="ABSA_Project/results/experiments_2026",
        help="Directory containing the new run folders",
    )
    parser.add_argument(
        "--legacy_dir",
        type=str,
        default="ABSA_Project/results/evaluation",
        help="Directory containing the existing 5/10-epoch result JSONs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ABSA_Project/results/experiments_2026/summary",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if a required group has fewer than 3 runs",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = collect_runs(args.root, args.legacy_dir)
    runs_df, table = build_table(rows)

    warnings = []
    for group in REQUIRED_GROUPS:
        sub = runs_df[runs_df["group"] == group]
        if sub.empty:
            warnings.append(f"Thiếu nhóm bắt buộc: {group}")
        elif len(sub) < REQUIRED_RUNS:
            warnings.append(
                f"{group} chỉ có {len(sub)}/{REQUIRED_RUNS} run; "
                "chưa đủ để lấy trung bình giữa các lần chạy độc lập."
            )

    for _, row in table.iterrows():
        if row["group"] in REQUIRED_GROUPS:
            continue  # already reported above, avoid duplicate warnings
        if int(row["n_runs"]) < REQUIRED_RUNS:
            warnings.append(
                f"{row['label']} chỉ có {int(row['n_runs'])} run (khuyến nghị {REQUIRED_RUNS})."
            )

    runs_path = os.path.join(args.out_dir, "runs_long.csv")
    table_path = os.path.join(args.out_dir, "comparison_table.csv")
    markdown_path = os.path.join(args.out_dir, "comparison_table.md")
    summary_path = os.path.join(args.out_dir, "summary.json")

    runs_df.to_csv(runs_path, index=False, encoding="utf-8-sig")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    markdown_text = write_markdown(table, markdown_path)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": (
            "One Test result per independent run (best Dev checkpoint). "
            "Mean and sample std (ddof=1) across seeds. No epoch averaging."
        ),
        "required_runs_per_group": REQUIRED_RUNS,
        "warnings": warnings,
        "groups": table.to_dict(orient="records"),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n=== Bảng so sánh (mean ± std qua các seed) ===")
    print(markdown_text)
    print(f"Saved: {table_path}")
    print(f"Saved: {markdown_path}")
    print(f"Saved: {runs_path}")
    print(f"Saved: {summary_path}")

    if warnings:
        print("\n=== CẢNH BÁO ===")
        for warning in warnings:
            print(f"  - {warning}")
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
