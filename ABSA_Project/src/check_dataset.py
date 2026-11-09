"""
check_dataset.py
Reproducible data-quality audit for the UIT-ViSFD splits.

Reads the *original* CSVs (never modifies them) and regenerates:
  * data_quality_report.csv
  * label_quality_report.csv
  * text_length_statistics.csv
  * aspect_distribution.csv
  * sentiment_distribution.csv
  * aspect_cooccurrence_count.csv / aspect_cooccurrence_jaccard.csv
  * cross_split_duplicate_{summary,details}.csv
  * cross_split_label_{comparison,conflicts}.csv
  * cross_split_duplicate_label_audit.csv
  * cross_split_annotation_diagnostic.csv
  * date_time_format_{details,summary}.csv
  * invalid_absolute_dates.csv
  * monthly_comment_distribution.csv / yearly_comment_distribution.csv
  * long_comment_audit.csv (uses the token_length column if available)

Author: Nhom 2 - Thanh vien 1
"""

import os
import re
import sys
import json
import argparse
import unicodedata
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# ---------------------------------------------------------------------------
# Label parsing (shares the link order with dataset_loader)
# ---------------------------------------------------------------------------
def load_label_convention():
    """Import the canonical order from the training code, with a safe fallback."""
    src = os.path.join(ROOT, "ABSA_Project", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from dataset_loader import ASPECT_LIST, IDX2POLARITY  # noqa: WPS433
    except Exception:
        ASPECT_LIST = [
            "BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL",
            "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE",
        ]
        IDX2POLARITY = {0: "None", 1: "Negative", 2: "Neutral", 3: "Positive"}
    return list(ASPECT_LIST), dict(IDX2POLARITY)


ASPECT_LIST, IDX2POLARITY = load_label_convention()
VALID_POLARITIES = [IDX2POLARITY[i] for i in (1, 2, 3)]
LABEL_TOKEN_RE = re.compile(r"\{([^{}]+)\}")
VALID_PAIR_RE = re.compile(r"^([A-Za-z0-9_&]+)#(Negative|Neutral|Positive)$")


def parse_entries(label):
    """Return (valid_pairs, others_count, invalid_count) for one label string."""
    pairs, others, invalid = [], 0, 0
    if not isinstance(label, str):
        return pairs, others, invalid
    for token in LABEL_TOKEN_RE.findall(label):
        token = token.strip()
        if token == "OTHERS" or token.startswith("OTHERS#"):
            others += 1
            continue
        match = VALID_PAIR_RE.match(token)
        if match and match.group(1) in ASPECT_LIST:
            pairs.append((match.group(1), match.group(2)))
        else:
            invalid += 1
    return pairs, others, invalid


def canonical_label(label):
    """Alphabetical aspect tokens, with OTHERS sorted in as a normal token."""
    pairs, others, _ = parse_entries(label)
    values = {aspect: polarity for aspect, polarity in pairs}
    tokens = [f"{aspect}#{values[aspect]}" for aspect in ASPECT_LIST if aspect in values]
    if others:
        tokens.append("OTHERS")
    return ";".join(sorted(tokens))


def ordered_label(label):
    """Aspect-order tokens (dataset order), with OTHERS appended at the end."""
    pairs, others, _ = parse_entries(label)
    values = {aspect: polarity for aspect, polarity in pairs}
    tokens = [f"{aspect}#{values[aspect]}" for aspect in ASPECT_LIST if aspect in values]
    if others:
        tokens.append("OTHERS")
    return ";".join(tokens)


def label_vector(label):
    vector = {aspect: "None" for aspect in ASPECT_LIST}
    pairs, _, _ = parse_entries(label)
    for aspect, polarity in pairs:
        vector[aspect] = polarity
    return vector


# ---------------------------------------------------------------------------
# Text / date helpers
# ---------------------------------------------------------------------------
def normalize_text(value):
    text = unicodedata.normalize("NFC", str(value))
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


RELATIVE_RE = re.compile(
    r"^\s*\d+\s*(giây|phút|giờ|ngày|tuần|tháng|năm)\s+trước\s*$", re.IGNORECASE
)
ABSOLUTE_DMY_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
ABSOLUTE_YMD_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")


def classify_date(value):
    if pd.isna(value):
        return "Missing"
    text = unicodedata.normalize("NFC", str(value)).strip()
    if not text:
        return "Empty"
    if ABSOLUTE_DMY_RE.match(text):
        return "Absolute_Date_DD_MM_YYYY"
    if ABSOLUTE_YMD_RE.match(text):
        return "Absolute_Date_YYYY_MM_DD"
    if RELATIVE_RE.match(text):
        return "Relative_Time"
    return "Other"


def parse_date(value):
    """Return (ISO date string or None, status)."""
    kind = classify_date(value)
    if kind == "Absolute_Date_DD_MM_YYYY":
        try:
            return datetime.strptime(str(value).strip(), "%d/%m/%Y").strftime("%Y-%m-%d"), "Absolute_Date"
        except ValueError:
            return None, "Invalid_Absolute_Date"
    if kind == "Absolute_Date_YYYY_MM_DD":
        try:
            return datetime.strptime(str(value).strip(), "%Y-%m-%d").strftime("%Y-%m-%d"), "Absolute_Date"
        except ValueError:
            return None, "Invalid_Absolute_Date"
    if kind == "Relative_Time":
        return None, "Relative_Time"
    if kind in ("Missing", "Empty"):
        return None, kind
    return None, "Unrecognized"


# ---------------------------------------------------------------------------
# Core reports
# ---------------------------------------------------------------------------
def data_quality_report(frames, cols):
    rows = []
    for split, df in frames.items():
        comment = df[cols["text"]]
        label = df[cols["label"]]
        rating = df[cols["rating"]]
        date = df[cols["date"]]
        rating_numeric = pd.to_numeric(rating, errors="coerce")
        rows.append({
            "Dataset": split,
            "Total_Rows": len(df),
            "Total_Columns": df.shape[1],
            "Missing_Comment": int(comment.isna().sum()),
            "Blank_Comment": int(((comment.astype(str).str.strip() == "") & comment.notna()).sum()),
            "Missing_Label": int(label.isna().sum()),
            "Blank_Label": int(((label.astype(str).str.strip() == "") & label.notna()).sum()),
            "Missing_Rating": int(rating.isna().sum()),
            "Missing_Date": int(date.isna().sum()),
            "Duplicate_Rows": int(df.duplicated().sum()),
            "Duplicate_Comments": int(df.duplicated(subset=[cols["text"]]).sum()),
            "Invalid_Ratings": int((~rating_numeric.isin([1, 2, 3, 4, 5])
                                   & rating.notna()).sum()),
            "Invalid_Rating_Format": int((rating.notna() & rating_numeric.isna()).sum()),
        })
    return pd.DataFrame(rows)


def label_quality_report(frames, cols):
    rows = []
    for split, df in frames.items():
        no_valid = others = duplicates = invalid = 0
        for label in df[cols["label"]]:
            pairs, other_count, invalid_count = parse_entries(label)
            others += other_count
            invalid += invalid_count
            if not pairs:
                no_valid += 1
            if len({aspect for aspect, _ in pairs}) != len(pairs):
                duplicates += 1
        rows.append({
            "Dataset": split,
            "Total_Comments": len(df),
            "Comments_With_No_Valid_Aspect": no_valid,
            "Others_Count": others,
            "Duplicate_Aspect_Count": duplicates,
            "Invalid_Label_Count": invalid,
        })
    return pd.DataFrame(rows)


def text_length_statistics(df, cols):
    comment = df[cols["text"]].astype(str)
    measures = {
        "Character_Length": comment.str.len(),
        "Whitespace_Unit_Count": comment.str.split().str.len(),
    }
    rows = []
    for name, series in measures.items():
        rows.append({
            "Measure": name,
            "Count": int(series.count()),
            "Minimum": int(series.min()),
            "Maximum": int(series.max()),
            "Mean": round(float(series.mean()), 2),
            "Median": round(float(series.median()), 2),
            "Standard_Deviation": round(float(series.std()), 2),
            "Percentile_90": float(np.percentile(series, 90)),
            "Percentile_95": float(np.percentile(series, 95)),
            "Percentile_99": float(np.percentile(series, 99)),
        })
    return pd.DataFrame(rows)


def aspect_tables(frames, cols):
    distribution, sentiment = [], []
    for split, df in frames.items():
        counts = {aspect: Counter() for aspect in ASPECT_LIST}
        for label in df[cols["label"]]:
            for aspect, polarity in parse_entries(label)[0]:
                counts[aspect][polarity] += 1
        total = len(df)
        for aspect in ASPECT_LIST:
            pos = counts[aspect]["Positive"]
            neu = counts[aspect]["Neutral"]
            neg = counts[aspect]["Negative"]
            mentions = pos + neu + neg
            distribution.append({
                "Dataset": split,
                "Aspect": aspect,
                "Comment_Count": mentions,
                "Positive": pos,
                "Neutral": neu,
                "Negative": neg,
                "Total_Labels": mentions,
                "Percentage_Of_Comments": round(mentions / total * 100, 2) if total else 0.0,
            })
            for polarity, value in (("Positive", pos), ("Neutral", neu), ("Negative", neg)):
                sentiment.append({
                    "Dataset": split, "Aspect": aspect,
                    "Sentiment": polarity, "Count": value,
                })
    return pd.DataFrame(distribution), pd.DataFrame(sentiment)


def cooccurrence_tables(df, cols):
    counts = np.zeros((len(ASPECT_LIST), len(ASPECT_LIST)), dtype=np.int64)
    index = {aspect: i for i, aspect in enumerate(ASPECT_LIST)}
    for label in df[cols["label"]]:
        present = sorted({index[aspect] for aspect, _ in parse_entries(label)[0]})
        for i in present:
            for j in present:
                counts[i, j] += 1
    count_df = pd.DataFrame(counts, index=ASPECT_LIST, columns=ASPECT_LIST)

    jaccard = np.zeros_like(counts, dtype=float)
    for i in range(len(ASPECT_LIST)):
        for j in range(len(ASPECT_LIST)):
            union = counts[i, i] + counts[j, j] - counts[i, j]
            jaccard[i, j] = counts[i, j] / union if union else 0.0
    jaccard_df = pd.DataFrame(jaccard, index=ASPECT_LIST, columns=ASPECT_LIST)
    return count_df, jaccard_df


def cooccurrence_pairs(count_df, jaccard_df):
    """Flatten the upper triangle of the co-occurrence matrices, most frequent first."""
    rows = []
    for i, aspect_a in enumerate(ASPECT_LIST):
        for j, aspect_b in enumerate(ASPECT_LIST):
            if j <= i:
                continue
            rows.append({
                "Aspect_A": aspect_a,
                "Aspect_B": aspect_b,
                "Cooccurrence_Count": int(count_df.iloc[i, j]),
                "Jaccard_Similarity": float(jaccard_df.iloc[i, j]),
            })
    pairs = pd.DataFrame(rows)
    return pairs.sort_values("Cooccurrence_Count", ascending=False,
                             kind="mergesort").reset_index(drop=True)


def cross_split_tables(frames, cols):
    records = []
    for split in ["Dev", "Test", "Train"]:
        df = frames[split]
        for row, label in zip(df.itertuples(index=False), df[cols["label"]]):
            index_value = getattr(row, "index", None)
            records.append({
                "index": index_value,
                "comment": getattr(row, cols["text"]),
                "Dataset": split,
                "Normalized_Comment": normalize_text(getattr(row, cols["text"])),
                "label": label,
                "Canonical_Label": canonical_label(label),
            })
    long_df = pd.DataFrame(records)
    long_df = long_df.sort_values(["Normalized_Comment", "Dataset"],
                                  kind="mergesort").reset_index(drop=True)
    groups = long_df.groupby("Normalized_Comment", sort=True)
    duplicated = groups.filter(lambda g: g["Dataset"].nunique() > 1).copy()

    summary_rows, comparison_rows, diagnostic_rows = [], [], []
    ordered_groups = duplicated.groupby("Normalized_Comment", sort=True)
    for normalized, group in ordered_groups:
        datasets = sorted(group["Dataset"].unique())
        canonical_values = list(dict.fromkeys(group["Canonical_Label"]))
        summary_rows.append({
            "Normalized_Comment": normalized,
            "Dataset_List": ", ".join(datasets),
            "Number_Of_Datasets": len(datasets),
            "Number_Of_Occurrences": len(group),
        })

        labels_by_dataset = {}
        for dataset in datasets:
            subset = group[group["Dataset"] == dataset]
            labels_by_dataset[dataset] = list(dict.fromkeys(subset["Canonical_Label"]))
        flat_labels = [value for values in labels_by_dataset.values() for value in values]
        comparison_rows.append({
            "Normalized_Comment": normalized,
            "Dataset_List": ", ".join(datasets),
            "Number_Of_Datasets": len(datasets),
            "Number_Of_Different_Labels": len(set(flat_labels)),
            "Original_Labels": " || ".join(sorted(set(flat_labels))),
        })

        # Diagnostics compare one representative label per dataset.
        vectors = {}
        raw_labels = {}
        indices = {}
        for dataset in datasets:
            subset = group[group["Dataset"] == dataset]
            first = subset.iloc[0]
            vectors[dataset] = label_vector(first["label"])
            raw_labels[dataset] = ordered_label(first["label"])
            indices[dataset] = first["index"]

        polarity_conflicts = []
        coverage = []
        others_mismatch = False
        for aspect in ASPECT_LIST:
            present = {d: vectors[d][aspect] != "None" for d in datasets}
            values = {vectors[d][aspect] for d in datasets}
            # A polarity conflict requires the aspect to be annotated in every
            # dataset; extra/missing aspects are reported as coverage gaps.
            if all(present.values()) and len(values) > 1:
                ordered = sorted((v for v in values if v != "None"),
                                 key=lambda name: ["Negative", "Neutral", "Positive"].index(name)
                                 if name in ["Negative", "Neutral", "Positive"] else 99)
                polarity_conflicts.append(f"{aspect}: {', '.join(ordered)}")
            if len(set(present.values())) > 1:
                coverage.append(aspect)
        others_flags = {d: ("OTHERS" in raw_labels[d].split(";")) for d in datasets}
        others_mismatch = len(set(others_flags.values())) > 1

        conflict_types = []
        if polarity_conflicts:
            conflict_types.append("Mâu thuẫn cực tính cảm xúc")
        if coverage:
            conflict_types.append("Khác biệt phạm vi gán nhãn khía cạnh")
        if others_mismatch:
            conflict_types.append("Khác biệt nhãn OTHERS")
        if not conflict_types:
            conflict_types.append("Không phát hiện khác biệt")

        details = " || ".join(
            f"{d} (Index={indices[d]}): {raw_labels[d].replace(';', '; ')}" for d in datasets
        )
        diagnostic_rows.append({
            "Normalized_Comment": normalized,
            "Dataset_List": ", ".join(datasets),
            "Conflict_Type": "; ".join(conflict_types),
            "Polarity_Conflicts": "; ".join(polarity_conflicts),
            "Aspect_Coverage_Differences": ", ".join(coverage),
            "OTHERS_Mismatch": others_mismatch,
            "Label_Details": details,
        })

    summary = pd.DataFrame(summary_rows)
    comparison = pd.DataFrame(comparison_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    conflicts = comparison[comparison["Number_Of_Different_Labels"] > 1].copy() \
        if not comparison.empty else comparison

    details_df = duplicated[["index", "comment", "Dataset", "Normalized_Comment"]] \
        .sort_values(["Normalized_Comment", "Dataset"]).reset_index(drop=True)
    audit_df = duplicated[["index", "comment", "Dataset", "Normalized_Comment",
                           "label", "Canonical_Label"]].copy()
    audit_df["Index_Key"] = audit_df["index"].astype(str)
    audit_df = audit_df[["index", "comment", "Dataset", "Normalized_Comment",
                         "Index_Key", "label", "Canonical_Label"]]
    return summary, details_df, comparison, conflicts, audit_df, diagnostics


def date_tables(frames, cols):
    detail_rows, summary_rows, invalid_rows = [], [], []
    monthly = Counter()
    yearly = Counter()
    for split, df in frames.items():
        for index_value, value in zip(df.get("index", df.index), df[cols["date"]]):
            kind = classify_date(value)
            detail_rows.append({
                "index": index_value,
                "date_time": value,
                "Dataset": split,
                "Date_Format_Type": kind,
            })
            parsed, status = parse_date(value)
            if status == "Invalid_Absolute_Date":
                invalid_rows.append({
                    "index": index_value,
                    "comment": df.loc[df.get("index", df.index) == index_value,
                                      cols["text"]].iloc[0] if "index" in df.columns else "",
                    "n_star": np.nan,
                    "date_time": value,
                    "label": "",
                    "Dataset": split,
                    "date_parsed": None,
                    "date_status": status,
                })
            if parsed:
                monthly[parsed[:7]] += 1
                yearly[parsed[:4]] += 1

    details = pd.DataFrame(detail_rows)
    for split, df in frames.items():
        counts = details[details["Dataset"] == split]["Date_Format_Type"].value_counts()
        total = len(df)
        for kind, count in counts.items():
            summary_rows.append({
                "Dataset": split,
                "Date_Format_Type": kind,
                "Count": int(count),
                "Percentage": round(count / total * 100, 2) if total else 0.0,
            })

    monthly_df = pd.DataFrame(sorted(monthly.items()),
                              columns=["Year_Month", "Comment_Count"])
    yearly_df = pd.DataFrame(sorted(yearly.items()),
                             columns=["Year", "Comment_Count"])
    invalid_df = pd.DataFrame(invalid_rows)
    if invalid_df.empty:
        invalid_df = pd.DataFrame(columns=["index", "comment", "n_star", "date_time",
                                           "label", "Dataset", "date_parsed", "date_status"])
    return details, pd.DataFrame(summary_rows), monthly_df, yearly_df, invalid_df


def long_comment_audit(frames, prepared_dir, cols):
    rows = []
    for split, original in frames.items():
        path = os.path.join(prepared_dir, f"{split}_prepared.csv")
        if not os.path.exists(path):
            continue
        prepared = pd.read_csv(path)
        if "token_length" not in prepared.columns:
            continue
        long_rows = prepared[prepared["token_length"] > 128]
        if "index" in long_rows.columns:
            long_rows = long_rows.sort_values("index")
        for position, row in long_rows.iterrows():
            rows.append({
                "Dataset": split,
                "Index": row.get("index", position),
                "Token_Length": int(row["token_length"]),
                "Comment": row.get(cols["text"], ""),
                "Label": row.get(cols["label"], ""),
                "Exceed_256": bool(row["token_length"] > 256),
            })
    result = pd.DataFrame(rows)
    # Keep the split order from the config (Train, Dev, Test), then index order,
    # so the audit lines up with the original report.
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="UIT-ViSFD data-quality audit")
    parser.add_argument("--config", default=os.path.join(ROOT, "ABSA_Project", "configs", "preprocessing.yaml"))
    parser.add_argument("--out-dir", default=None,
                        help="Override the statistics output directory")
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args(argv)

    config = load_config(resolve(args.config))
    cols = config["columns"]
    splits = args.splits or config["splits"]
    out_dir = resolve(args.out_dir or config["data"]["stats_dir"])
    os.makedirs(out_dir, exist_ok=True)

    frames = {}
    for split in splits:
        path = os.path.join(resolve(config["data"]["original_dir"]), f"{split}.csv")
        frames[split] = pd.read_csv(path, encoding="utf-8-sig")
        print(f"Loaded {split}: {len(frames[split]):,} rows from {path}")

    def save(frame, name):
        path = os.path.join(out_dir, name)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  -> {name} ({len(frame)} rows)")

    save(data_quality_report(frames, cols), "data_quality_report.csv")
    save(label_quality_report(frames, cols), "label_quality_report.csv")
    save(text_length_statistics(frames.get("Train", next(iter(frames.values()))), cols),
         "text_length_statistics.csv")

    distribution, sentiment = aspect_tables(frames, cols)
    save(distribution, "aspect_distribution.csv")
    save(sentiment, "sentiment_distribution.csv")

    cooccurrence, jaccard = cooccurrence_tables(
        frames.get("Train", next(iter(frames.values()))), cols)
    cooccurrence.to_csv(os.path.join(out_dir, "aspect_cooccurrence_count.csv"),
                        encoding="utf-8-sig")
    jaccard.to_csv(os.path.join(out_dir, "aspect_cooccurrence_jaccard.csv"),
                   encoding="utf-8-sig")
    pairs = cooccurrence_pairs(cooccurrence, jaccard)
    save(pairs, "aspect_cooccurrence_pairs.csv")
    print("  -> aspect_cooccurrence_count.csv")
    print("  -> aspect_cooccurrence_jaccard.csv")

    summary, details, comparison, conflicts, audit, diagnostics = cross_split_tables(frames, cols)
    save(summary, "cross_split_duplicate_summary.csv")
    save(details, "cross_split_duplicate_details.csv")
    save(comparison, "cross_split_label_comparison.csv")
    save(conflicts, "cross_split_label_conflicts.csv")
    save(audit, "cross_split_duplicate_label_audit.csv")
    save(diagnostics, "cross_split_annotation_diagnostic.csv")

    date_details, date_summary, monthly, yearly, invalid = date_tables(frames, cols)
    save(date_details, "date_time_format_details.csv")
    save(date_summary, "date_time_format_summary.csv")
    save(monthly, "monthly_comment_distribution.csv")
    save(yearly, "yearly_comment_distribution.csv")
    save(invalid, "invalid_absolute_dates.csv")

    audit_long = long_comment_audit(frames, resolve(config["data"]["prepared_dir"]), cols)
    if not audit_long.empty:
        save(audit_long, "long_comment_audit.csv")

    print(f"\nAll reports written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
