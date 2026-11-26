"""
run_5seed_2026.py

Batch bổ sung 2 seed (7, 777) để đủ 5 seed cho báo cáo:
  * phobert_10ep_seed7 / _seed777   -> cấu hình chính thức 10 epoch, ES tắt
  * phobert_es20_seed7 / _seed777   -> ES max 20 epoch, patience 3
  * bilstm_ft_seed7 / _seed777      -> BiLSTM + fastText, max 20 epoch, patience 3
  * bilstm_rand_seed7 / _seed777    -> BiLSTM random embedding (train_baseline.py)

Nguyên tắc:
  * Không ghi đè kết quả cũ: mọi thứ nằm trong experiments_5seed/.
  * Bỏ qua run đã có test_results_<name>.json.
  * Mỗi run thêm một dòng vào registry_5seed.csv.

Usage
-----
python ABSA_Project/src/run_5seed_2026.py --list
python ABSA_Project/src/run_5seed_2026.py --all
python ABSA_Project/src/run_5seed_2026.py --only phobert_10ep_seed7
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_experiments_2026 as r26  # noqa: E402
from train_phobert import train as train_phobert  # noqa: E402
from train_bilstm_fasttext import train as train_bilstm_ft  # noqa: E402
import train_baseline  # noqa: E402

NEW_MODELS = "ABSA_Project/models/experiments_5seed"
NEW_RESULTS = "ABSA_Project/results/experiments_5seed"
REGISTRY = os.path.join(NEW_RESULTS, "registry_5seed.csv")
DATA_DIR = r26.DATA_DIR
EMBEDDING_PATH = r26.EMBEDDING_PATH
VOCAB_PATH = r26.VOCAB_PATH

SEEDS = (7, 777)


def _experiments() -> list[dict]:
    exps = []
    for seed in SEEDS:
        exps.append({
            "name": f"phobert_10ep_seed{seed}",
            "kind": "phobert",
            "seed": seed,
            "epochs": 10,
            "patience": 10,
            "description": f"PhoBERT 10 epoch, seed {seed} (fixed budget, ES disabled)",
        })
    for seed in SEEDS:
        exps.append({
            "name": f"phobert_5ep_seed{seed}",
            "kind": "phobert",
            "seed": seed,
            "epochs": 5,
            "patience": 3,
            "description": f"PhoBERT 5 epoch (checkpoint phát triển), seed {seed}",
        })
    for seed in SEEDS:
        exps.append({
            "name": f"phobert_es20_seed{seed}",
            "kind": "phobert",
            "seed": seed,
            "epochs": 20,
            "patience": 3,
            "description": f"PhoBERT ES max 20 epoch, patience 3, seed {seed}",
        })
    for seed in SEEDS:
        exps.append({
            "name": f"bilstm_ft_seed{seed}",
            "kind": "bilstm_ft",
            "seed": seed,
            "epochs": 20,
            "patience": 3,
            "description": f"BiLSTM + fastText, max 20 epoch, patience 3, seed {seed}",
        })
    for seed in SEEDS:
        exps.append({
            "name": f"bilstm_rand_seed{seed}",
            "kind": "bilstm_rand",
            "seed": seed,
            "epochs": 20,
            "patience": 4,
            "description": f"BiLSTM random embedding, max 20 epoch, patience 4, seed {seed}",
        })
    return exps


EXPERIMENTS = _experiments()
BY_NAME = {e["name"]: e for e in EXPERIMENTS}


def _paths(exp):
    return (
        os.path.join(NEW_MODELS, exp["name"]),
        os.path.join(NEW_RESULTS, exp["name"]),
    )


def _result_path(exp):
    _, log_dir = _paths(exp)
    return os.path.join(log_dir, f"test_results_{exp['name']}.json")


def _count_epochs(exp):
    _, log_dir = _paths(exp)
    path = os.path.join(log_dir, f"train_log_{exp['name']}.csv")
    if not os.path.exists(path):
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def _build_args(exp):
    output_dir, log_dir = _paths(exp)
    if exp["kind"] == "phobert":
        values = copy.deepcopy(r26.PHOBERT_COMMON)
        values.update({
            "experiment_name": exp["name"],
            "output_dir": output_dir,
            "log_dir": log_dir,
            "seed": exp["seed"],
            "epochs": exp["epochs"],
            "patience": exp["patience"],
        })
        return argparse.Namespace(**values)

    if exp["kind"] == "bilstm_ft":
        values = copy.deepcopy(r26.BILSTM_FT_COMMON)
        values.update({
            "experiment_name": exp["name"],
            "output_dir": output_dir,
            "log_dir": log_dir,
            "seed": exp["seed"],
            "epochs": exp["epochs"],
            "patience": exp["patience"],
            "embedding_path": EMBEDDING_PATH,
            "vocab_path": VOCAB_PATH,
        })
        return argparse.Namespace(**values)

    # bilstm_rand: defaults của train_baseline.py
    return argparse.Namespace(
        data_dir=DATA_DIR,
        output_dir=output_dir,
        log_dir=log_dir,
        experiment_name=exp["name"],
        max_len=128,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=exp["epochs"],
        dropout=0.3,
        emb_dim=256,
        hidden_dim=256,
        num_layers=2,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        patience=exp["patience"],
        seed=exp["seed"],
        vocab_min_freq=2,
        vocab_size=64000,
        fp16=True,
        no_cuda=False,
        num_workers=0,
    )


def append_registry(exp, args, result, wall_time_s):
    row = {
        "Experiment": exp["name"],
        "Kind": exp["kind"],
        "Description": exp["description"],
        "Seed": exp["seed"],
        "Epochs_Planned": exp["epochs"],
        "Patience": exp["patience"],
        "Last_Epoch": result.get("last_epoch") or _count_epochs(exp),
        "Best_Epoch": result.get("best_epoch"),
        "Stopped_Early": result.get("stopped_early"),
        "Best_Dev_Macro_F1": result.get("best_dev_macro_f1"),
        "Test_Micro_F1": result.get("test_micro_f1"),
        "Test_Macro_F1": result.get("test_macro_f1_4class", result.get("test_macro_f1")),
        "Test_Sentiment_F1": result.get("test_sentiment_macro_f1"),
        "Test_ACD_F1": result.get("test_acd_f1"),
        "Test_Exact_Match": result.get("test_exact_match"),
        "Wall_Time_s": wall_time_s,
        "Output_Dir": args.output_dir,
        "Log_Dir": args.log_dir,
    }
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    header = not os.path.exists(REGISTRY)
    df.to_csv(REGISTRY, mode="a", header=header, index=False, encoding="utf-8-sig")


def run_one(exp):
    name = exp["name"]
    result_path = _result_path(exp)
    if os.path.exists(result_path):
        print(f"[SKIP] {name}: {result_path} exists")
        return None

    if exp["kind"] == "bilstm_ft":
        missing = [p for p in (EMBEDDING_PATH, VOCAB_PATH) if not os.path.exists(p)]
        if missing:
            raise SystemExit("Missing fastText artifacts:\n  " + "\n  ".join(missing))

    args = _build_args(exp)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n===== RUN {name} =====", flush=True)
    print(f"       seed={exp['seed']} epochs={exp['epochs']} "
          f"patience={exp['patience']}", flush=True)

    started = time.time()
    if exp["kind"] == "phobert":
        result = train_phobert(args)
    elif exp["kind"] == "bilstm_ft":
        result = train_bilstm_ft(args)
    else:
        result = train_baseline.train(args)
    wall = time.time() - started

    if result.get("last_epoch") is None:
        result["last_epoch"] = _count_epochs(exp)
    append_registry(exp, args, result, wall)
    print(f"===== DONE {name} in {wall/60:.1f} min | "
          f"best_epoch={result.get('best_epoch')} "
          f"macro4={result.get('test_macro_f1_4class', result.get('test_macro_f1'))} =====",
          flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'Experiment':<26} {'Kind':<12} {'Seed':>6} {'Epochs':>7} {'Done':>6}")
        for e in EXPERIMENTS:
            print(f"{e['name']:<26} {e['kind']:<12} {e['seed']:>6} "
                  f"{e['epochs']:>7} {str(os.path.exists(_result_path(e))):>6}")
        return 0

    if args.only:
        unknown = [n for n in args.only if n not in BY_NAME]
        if unknown:
            print("Unknown:", unknown)
            return 2
        selected = [BY_NAME[n] for n in args.only]
    elif args.all:
        selected = EXPERIMENTS
    else:
        print("Use --all or --only <name> ...")
        return 2

    failures = []
    for exp in selected:
        try:
            run_one(exp)
        except Exception as exc:
            failures.append((exp["name"], str(exc)))
            print(f"!!!!! FAILED {exp['name']}: {exc}", flush=True)
            traceback.print_exc()
    if failures:
        print("\n=== FAILURES ===")
        for n, m in failures:
            print(f"  {n}: {m}")
        return 1
    print("\nAll selected experiments finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
