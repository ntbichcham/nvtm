"""
build_literature_table.py
=========================
Writes ``ABSA_Project/data/literature_baselines.csv`` — the curated table of
numbers reported in related studies plus the group's own reproduced results.

Every row is validated against the header (exact column count) before the file
is written, so a mis-aligned comma can never silently corrupt the table.

Comparability flags
-------------------
reported                  number copied from the original paper (different split
                          protocol possible)
reproduced                number produced by this group's own runs with the
                          same Train/Dev/Test splits and metrics_utils
not_directly_comparable   different dataset / benchmark; listed for context only

Usage
-----
python build_literature_table.py
python build_literature_table.py --check   # validate the existing CSV only

Author: Nhom 2 - CT255
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Sequence

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "literature_baselines.csv"

HEADER = [
    "study_id", "study", "year", "venue", "dataset", "task", "model",
    "model_family", "acd_precision", "acd_recall", "acd_f1",
    "asc_precision", "asc_recall", "asc_f1",
    "macro_f1", "macro_f1_scope", "accuracy", "micro_f1", "weighted_f1",
    "exact_match", "comparison_type", "source", "notes",
]

SA2SL_DOI = "10.1007/978-3-030-82147-0_53"
THIN_SRC = "Table 6 in So_Sanh_Chi_Tiet_Da_Chieu_Voi_Cac_Bai_Bao_Goc.docx"
VISO_SRC = "Table in So_Sanh_Chi_Tiet_Da_Chieu_Voi_Cac_Bai_Bao_Goc.docx"
GROUP_SRC = "results/experiments_2026 + bang_so_sanh_mo_hinh_va_early_stopping.csv"


def sa2sl(model: str, family: str, p: float, r: float, f1: float,
          sp: float, sr: float, sf1: float, note: str) -> List[object]:
    return [
        f"sa2sl_{model.lower().replace(' ', '_').replace('-', '')}",
        "SA2SL (Phan et al.)", 2021, "KSEM 2021", "UIT-ViSFD", "ACD+ASC",
        model, family, p, r, f1, sp, sr, sf1, "", "", "", "", "", "",
        "reported", SA2SL_DOI, note,
    ]


def thin(model: str, family: str, p: float, r: float, f1: float, note: str) -> List[object]:
    return [
        f"thin_{model.lower().replace(' ', '').replace('-', '')}",
        "Thin et al. (Aspect-Category)", 2023, "VNU Journal of Science",
        "UIT-ViSFD smartphone", "ASC (macro P/R/F1)", model, family,
        "", "", "", p, r, f1, "", "", "", "", "", "",
        "reported", THIN_SRC, note,
    ]


def visobert(model: str, acc: float, wf1: float, mf1: float, note: str) -> List[object]:
    return [
        f"visobert_{model.lower().replace(' ', '').replace('-', '')}",
        "ViSoBERT (Nguyen et al.)", 2023, "EMNLP 2023",
        "Vietnamese social-media benchmark (not UIT-ViSFD)", "ASC", model,
        "Transformer", "", "", "", "", "", "", mf1,
        "own benchmark macro F1 (3-class social media)", acc, "", wf1, "",
        "not_directly_comparable", VISO_SRC, note,
    ]


def group(model: str, family: str, acd_p: float, acd_r: float, acd_f1: float,
          asc_p: float, asc_r: float, asc_f1: float, macro_f1: float,
          micro_f1: float, em: float, note: str) -> List[object]:
    return [
        "group_" + model.lower().replace(' ', '_').replace('-', '').replace('+', ''),
        "Nhóm 2 (tái lập)", 2026, "internal", "UIT-ViSFD", "ACD+ASC+4class",
        model, family, acd_p, acd_r, acd_f1, asc_p, asc_r, asc_f1,
        macro_f1, "4-class incl None (flattened 10 heads)",
        micro_f1, micro_f1, "", em, "reproduced", GROUP_SRC, note,
    ]


def build_rows() -> List[List[object]]:
    rows: List[List[object]] = []
    rows += [
        sa2sl("Naive Bayes", "ML", 72.18, 59.53, 64.65, 49.07, 30.43, 37.56,
              "Baseline ML trong bài SA2SL; ACD = aspect detection, ASC = aspect sentiment classification."),
        sa2sl("SVM", "ML", 36.45, 51.34, 42.63, 16.09, 23.35, 19.69,
              "SVM tuyến tính; kết quả thấp do đặc trưng thưa."),
        sa2sl("Random Forest", "ML", 45.72, 50.15, 47.83, 17.11, 24.56, 20.17,
              "Random Forest baseline."),
        sa2sl("CNN", "DL", 77.25, 63.49, 69.70, 33.34, 22.92, 27.16,
              "CNN baseline."),
        sa2sl("LSTM", "DL", 82.61, 78.05, 80.27, 56.51, 48.39, 52.13,
              "LSTM baseline."),
        sa2sl("Bi-LSTM (original)", "DL", 87.55, 83.22, 84.48, 65.82, 60.53, 63.06,
              "Mô hình đề xuất của bài báo gốc SA2SL."),
    ]
    rows += [
        thin("SVM", "ML", 78.20, 55.40, 64.84, "Macro P/R/F1 cho bài toán ASC."),
        thin("CNN", "DL", 75.40, 58.12, 65.65, "Macro P/R/F1."),
        thin("BiLSTM-CNN", "DL", 65.82, 60.53, 63.06, "Macro P/R/F1."),
        thin("PhoBERT", "Transformer", 82.72, 67.66, 74.44, "PhoBERT encoder single-task."),
    ]
    rows += [
        visobert("PhoBERT-base", 75.72, 75.52, 75.52,
                 "Benchmark khác (dữ liệu social media), KHÔNG so sánh trực tiếp với UIT-ViSFD."),
        visobert("PhoBERT-large", 76.52, 76.36, 76.22,
                 "Benchmark khác, KHÔNG so sánh trực tiếp."),
        visobert("ViSoBERT", 77.83, 77.75, 77.75,
                 "Mô hình ViSoBERT; benchmark khác, KHÔNG so sánh trực tiếp."),
    ]
    rows += [
        group("Bi-LSTM + fastText", "DL", 87.89, 85.21, 86.53, 70.42, 67.91, 69.14,
              75.37, 88.24, 33.32,
              "Trung bình 3 seed; micro F1 = accuracy; exact match 33.32."),
        group("Multi-Head PhoBERT 5 epochs", "Transformer", 90.15, 88.86, 89.50,
              73.84, 71.62, 72.71, 78.37, 90.50, 41.91,
              "Trung bình 3 seed; exact match 41.91."),
        group("Multi-Head PhoBERT 10 epochs", "Transformer", 91.24, 89.78, 90.50,
              77.10, 75.22, 76.15, 81.07, 91.58, 46.18,
              "Trung bình 3 seed; exact match 46.18."),
        group("Multi-Head PhoBERT ES 20/3", "Transformer", 91.45, 89.98, 90.71,
              77.85, 76.08, 76.95, 81.68, 91.79, 46.79,
              "Trung bình 3 seed. Checkpoint website best_phobert_absa.pt là checkpoint PhoBERT epoch 5 riêng, không phải con số mean 3 seed này."),
    ]
    return rows


def validate(rows: Sequence[Sequence[object]]) -> None:
    problems = []
    for i, row in enumerate(rows, start=2):
        if len(row) != len(HEADER):
            problems.append(f"  line {i}: {len(row)} columns (expected {len(HEADER)}) -> {row[:4]}")
    if problems:
        raise SystemExit("CSV validation failed:\n" + "\n".join(problems))


def write_csv(path: Path, rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def check_existing(path: Path) -> int:
    if not path.exists():
        print(f"[missing] {path}")
        return 1
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header, data = rows[0], rows[1:]
    bad = [i for i, r in enumerate(data, start=2) if len(r) != len(header)]
    print(f"{path}: {len(data)} rows, {len(header)} columns")
    if bad:
        print(f"  BAD ROWS: {bad}")
        return 1
    print("  OK — every row matches the header")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sinh va kiem tra bang so lieu tai lieu tham khao.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true", help="Chi kiem tra file hien co.")
    args = parser.parse_args(argv)
    path = Path(args.out).resolve()
    if args.check:
        return check_existing(path)
    rows = build_rows()
    validate(rows)
    write_csv(path, rows)
    print(f"Wrote {len(rows)} rows x {len(HEADER)} columns -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
