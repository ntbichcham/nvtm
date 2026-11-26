"""
run_experiments_2026.py
Driver for the new experiment batch (results are written to NEW folders only):

  * phobert_10ep_seed123 / phobert_10ep_seed2024
        Multi-Head PhoBERT, exactly 10 epochs (patience = 10 disables ES).
  * phobert_es20_seed42 / _seed123 / _seed2024
        Multi-Head PhoBERT, max 20 epochs, patience 3, monitor Dev Macro F1.
  * bilstm_ft_seed42 / _seed123 / _seed2024
        BiLSTM + fastText embedding, max 20 epochs, patience 3.

Unlike the old `run_experiments.py`, this driver:
  * never deletes checkpoints;
  * skips a run whose `test_results_<name>.json` already exists;
  * appends one row per run to `registry_2026.csv`.

Usage
-----
python ABSA_Project/src/run_experiments_2026.py --list
python ABSA_Project/src/run_experiments_2026.py --only phobert_es20_seed42
python ABSA_Project/src/run_experiments_2026.py --only bilstm_ft_seed42 bilstm_ft_seed123
python ABSA_Project/src/run_experiments_2026.py --all

Author: Nhom 2 - Thanh vien 3
"""

import os
import sys
import copy
import time
import json
import argparse
import traceback
from pathlib import Path

import pandas as pd
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from train_phobert import train as train_phobert
from train_bilstm_fasttext import train as train_bilstm_ft


DATA_DIR = "ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model"
NEW_MODELS = "ABSA_Project/models/experiments_2026"
NEW_RESULTS = "ABSA_Project/results/experiments_2026"
REGISTRY = os.path.join(NEW_RESULTS, "registry_2026.csv")
EMBEDDING_PATH = os.path.join(
    NEW_RESULTS, "fasttext", "fasttext_embeddings.npy"
)
VOCAB_PATH = os.path.join(NEW_RESULTS, "fasttext", "fasttext_vocab.json")

PHOBERT_COMMON = {
    "data_dir": DATA_DIR,
    "pretrained_model": "vinai/phobert-base",
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_len": 160,
    "dropout": 0.1,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "grad_accum": 1,
    "dense_head": False,
    "use_class_weights": True,
    "fp16": True,
    "no_cuda": False,
    "num_workers": 0,
}

BILSTM_FT_COMMON = {
    "data_dir": DATA_DIR,
    "max_len": 128,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "emb_dim": 300,
    "hidden_dim": 256,
    "num_layers": 2,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "vocab_min_freq": 2,
    "vocab_size": 64000,
    "fp16": True,
    "no_cuda": False,
    "num_workers": 0,
    "embedding_path": EMBEDDING_PATH,
    "vocab_path": VOCAB_PATH,
    "freeze_embeddings": False,
}

EXPERIMENTS = [
    {
        "name": "phobert_10ep_seed123",
        "kind": "phobert",
        "seed": 123,
        "epochs": 10,
        "patience": 10,
        "description": "PhoBERT 10 epoch, seed 123 (fixed budget, ES disabled)",
    },
    {
        "name": "phobert_10ep_seed2024",
        "kind": "phobert",
        "seed": 2024,
        "epochs": 10,
        "patience": 10,
        "description": "PhoBERT 10 epoch, seed 2024 (fixed budget, ES disabled)",
    },
    {
        "name": "phobert_es20_seed42",
        "kind": "phobert",
        "seed": 42,
        "epochs": 20,
        "patience": 3,
        "description": "PhoBERT ES max 20 epoch, patience 3, seed 42",
    },
    {
        "name": "phobert_es20_seed123",
        "kind": "phobert",
        "seed": 123,
        "epochs": 20,
        "patience": 3,
        "description": "PhoBERT ES max 20 epoch, patience 3, seed 123",
    },
    {
        "name": "phobert_es20_seed2024",
        "kind": "phobert",
        "seed": 2024,
        "epochs": 20,
        "patience": 3,
        "description": "PhoBERT ES max 20 epoch, patience 3, seed 2024",
    },
    {
        "name": "bilstm_ft_seed42",
        "kind": "bilstm_ft",
        "seed": 42,
        "epochs": 20,
        "patience": 3,
        "description": "BiLSTM + fastText, max 20 epoch, patience 3, seed 42",
    },
    {
        "name": "bilstm_ft_seed123",
        "kind": "bilstm_ft",
        "seed": 123,
        "epochs": 20,
        "patience": 3,
        "description": "BiLSTM + fastText, max 20 epoch, patience 3, seed 123",
    },
    {
        "name": "bilstm_ft_seed2024",
        "kind": "bilstm_ft",
        "seed": 2024,
        "epochs": 20,
        "patience": 3,
        "description": "BiLSTM + fastText, max 20 epoch, patience 3, seed 2024",
    },
]

BY_NAME = {exp["name"]: exp for exp in EXPERIMENTS}


def _paths(exp):
    name = exp["name"]
    return (
        os.path.join(NEW_MODELS, name),
        os.path.join(NEW_RESULTS, name),
    )


def _result_path(exp):
    _, log_dir = _paths(exp)
    return os.path.join(log_dir, f"test_results_{exp['name']}.json")


def _is_done(exp):
    return os.path.exists(_result_path(exp))


def _build_args(exp):
    name = exp["name"]
    output_dir, log_dir = _paths(exp)

    if exp["kind"] == "phobert":
        values = copy.deepcopy(PHOBERT_COMMON)
    else:
        values = copy.deepcopy(BILSTM_FT_COMMON)

    values.update(
        {
            "experiment_name": name,
            "output_dir": output_dir,
            "log_dir": log_dir,
            "seed": exp["seed"],
            "epochs": exp["epochs"],
            "patience": exp["patience"],
        }
    )
    if exp["kind"] == "bilstm_ft":
        # Resolve the shared fastText artifacts at call time so the driver
        # always uses the same embedding/vocabulary pair.
        values["embedding_path"] = EMBEDDING_PATH
        values["vocab_path"] = VOCAB_PATH
    return argparse.Namespace(**values)


def _count_epochs(exp):
    _, log_dir = _paths(exp)
    path = os.path.join(log_dir, f"train_log_{exp['name']}.csv")
    if not os.path.exists(path):
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


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
    df.to_csv(
        REGISTRY,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def run_one(exp, force=False):
    name = exp["name"]
    result_path = _result_path(exp)

    if _is_done(exp) and not force:
        print(f"[SKIP] {name}: {result_path} already exists")
        return None

    if exp["kind"] == "bilstm_ft":
        missing = [
            p for p in (EMBEDDING_PATH, VOCAB_PATH) if not os.path.exists(p)
        ]
        if missing:
            raise SystemExit(
                "Missing fastText embedding files:\n  "
                + "\n  ".join(missing)
                + "\nRun fasttext_embedding.py first."
            )

    args = _build_args(exp)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n===== RUN {name} =====")
    print(f"       {exp['description']}")
    print(f"       seed={exp['seed']} epochs={exp['epochs']} patience={exp['patience']}")
    print(f"       output_dir={args.output_dir}")
    print(f"       log_dir={args.log_dir}")

    started = time.time()
    if exp["kind"] == "phobert":
        result = train_phobert(args)
    else:
        result = train_bilstm_ft(args)
    wall_time = time.time() - started

    if result.get("last_epoch") is None:
        result["last_epoch"] = _count_epochs(exp)

    append_registry(exp, args, result, wall_time)
    print(
        f"===== DONE {name} in {wall_time:.0f}s | "
        f"last_epoch={result.get('last_epoch')} "
        f"best_epoch={result.get('best_epoch')} ====="
    )
    return result


def print_list():
    print(f"{'Experiment':<26} {'Kind':<10} {'Seed':>6} {'Epochs':>7} {'Patience':>9} {'Done':>6}")
    print("-" * 72)
    for exp in EXPERIMENTS:
        print(
            f"{exp['name']:<26} {exp['kind']:<10} {exp['seed']:>6} "
            f"{exp['epochs']:>7} {exp['patience']:>9} {str(_is_done(exp)):>6}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the 2026 comparison experiments (new output folders only)"
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--all", action="store_true", help="Run every experiment (skip completed)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if the result JSON exists"
    )
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    args = parser.parse_args()

    if args.list:
        print_list()
        return 0

    if args.only:
        unknown = [n for n in args.only if n not in BY_NAME]
        if unknown:
            print(f"Unknown experiment(s): {unknown}")
            print("Use --list to see valid names.")
            return 2
        selected = [BY_NAME[n] for n in args.only]
    elif args.all:
        selected = EXPERIMENTS
    else:
        print("Nothing selected. Use --all or --only <name> [<name> ...].")
        print_list()
        return 2

    failures = []
    for exp in selected:
        try:
            run_one(exp, force=args.force)
        except Exception as exc:  # keep the batch alive, log the failure
            failures.append((exp["name"], str(exc)))
            print(f"\n!!!!! FAILED {exp['name']}: {exc}")
            traceback.print_exc()

    if failures:
        print("\n=== FAILURES ===")
        for name, message in failures:
            print(f"  {name}: {message}")
        return 1

    print("\nAll selected experiments finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
