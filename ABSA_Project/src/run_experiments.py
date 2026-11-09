"""
run_experiments.py
Driver for the experiment batches requested for the report:
  * tuning  : the 6 remaining controlled hyper-parameter configurations
              (plus a 4-epoch reference run for an apples-to-apples table)
  * seeds   : seed-variance runs of the standard PhoBERT configuration
  * epochs  : longer training run to check whether epoch 5 was a real optimum
Results are appended to results/evaluation/experiment_registry.csv after every
run so that partial results survive an interrupted batch.
Author: Nhom 2 - Thanh vien 3
"""

import os
import sys
import glob
import time
import json
import copy
import argparse
import pandas as pd

import torch

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from train_phobert import train, parse_args

REGISTRY = 'ABSA_Project/results/evaluation/experiment_registry.csv'

BASE = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'grad_accum': 1,
    'max_len': 160,
    'weight_decay': 0.01,
    'dropout': 0.1,
    'warmup_ratio': 0.1,
    'use_class_weights': True,
    'epochs': 4,
    'patience': 3,
}

TUNING = [
    ('exp0_baseline', 'Baseline', 'Chuẩn 4 epoch (mốc so sánh)', {}),
    ('exp1a_lr_1e5', 'Learning Rate', 'LR = 1e-5', {'learning_rate': 1e-5}),
    ('exp1b_lr_3e5', 'Learning Rate', 'LR = 3e-5', {'learning_rate': 3e-5}),
    ('exp2a_batch_32', 'Batch Size', 'Batch hiệu dụng 32 (16 x grad_accum 2)',
     {'batch_size': 16, 'grad_accum': 2}),
    ('exp2b_maxlen_128', 'Max Length', 'max_len = 128', {'max_len': 128}),
    ('exp3a_reg_dropout02', 'Regularization', 'dropout 0.2, weight_decay 0.05',
     {'dropout': 0.2, 'weight_decay': 0.05}),
    ('exp3b_ablation_no_weights', 'Ablation', 'Không dùng class weights',
     {'use_class_weights': False}),
]


def append_registry(row):
    df = pd.DataFrame([row])
    header = not os.path.exists(REGISTRY)
    df.to_csv(REGISTRY, mode='a', header=header, index=False, encoding='utf-8-sig')


def cleanup_checkpoints(out_dir):
    for f in glob.glob(os.path.join(out_dir, '*.pt')):
        try:
            os.remove(f)
        except OSError:
            pass


def run_one(name, group, description, overrides, extra=None):
    # parse_args() reads sys.argv; isolate it from this driver's own CLI args.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        base = parse_args()
    finally:
        sys.argv = saved_argv
    base.experiment_name = name
    base.output_dir = f'ABSA_Project/models/experiments/{name}'
    base.log_dir = 'ABSA_Project/results/evaluation'
    for k, v in BASE.items():
        setattr(base, k, v)
    for k, v in overrides.items():
        setattr(base, k, v)
    if extra:
        for k, v in extra.items():
            setattr(base, k, v)

    os.makedirs(base.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(f'\n===== RUN {name} | {description} =====', flush=True)
    t0 = time.time()
    res = train(base)
    cleanup_checkpoints(base.output_dir)

    append_registry({
        'Experiment': name,
        'Group': group,
        'Description': description,
        'Seed': base.seed,
        'Learning_Rate': base.learning_rate,
        'Batch_Size': base.batch_size,
        'Grad_Accum': base.grad_accum,
        'Effective_Batch': base.batch_size * base.grad_accum,
        'Max_Len': base.max_len,
        'Dropout': base.dropout,
        'Weight_Decay': base.weight_decay,
        'Class_Weights': base.use_class_weights,
        'Epochs_Planned': base.epochs,
        'Best_Epoch': res['best_epoch'],
        'Best_Dev_Macro_F1': res['best_dev_macro_f1'],
        'Test_Micro_F1': res['test_micro_f1'],
        'Test_Macro_F1': res['test_macro_f1_4class'],
        'Test_Sentiment_F1': res['test_sentiment_macro_f1'],
        'Test_ACD_F1': res['test_acd_f1'],
        'Test_Exact_Match': res['test_exact_match'],
        'Latency_ms': res['latency_per_sample_ms'],
        'Peak_VRAM_MB': res['peak_vram_mb'],
        'Wall_Time_s': time.time() - t0,
    })
    print(f'===== DONE {name} in {time.time() - t0:.0f}s =====', flush=True)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['tuning', 'seeds', 'epochs', 'all'])
    parser.add_argument('--only', nargs='*', default=None)
    args = parser.parse_args()

    def do_tuning():
        for name, group, desc, ov in TUNING:
            if args.only and name not in args.only:
                continue
            try:
                run_one(name, group, desc, ov)
            except Exception as exc:  # keep the batch alive, log the failure
                print(f'!!!!! FAILED {name}: {exc}', flush=True)

    def do_seeds():
        for seed in [123, 2024]:
            try:
                run_one(f'official_seed{seed}', 'Stability',
                        f'Cấu hình chuẩn, seed {seed}', {},
                        extra={'seed': seed, 'epochs': 5, 'patience': 3})
            except Exception as exc:
                print(f'!!!!! FAILED seed {seed}: {exc}', flush=True)

    def do_epochs():
        try:
            run_one('official_ep10_seed42', 'Stability',
                    'Cấu hình chuẩn, 10 epoch, patience 4', {},
                    extra={'seed': 42, 'epochs': 10, 'patience': 4})
        except Exception as exc:
            print(f'!!!!! FAILED ep10: {exc}', flush=True)

    if args.mode == 'tuning':
        do_tuning()
    elif args.mode == 'seeds':
        do_seeds()
    elif args.mode == 'epochs':
        do_epochs()
    else:
        do_tuning()
        do_seeds()
        do_epochs()


if __name__ == '__main__':
    main()
