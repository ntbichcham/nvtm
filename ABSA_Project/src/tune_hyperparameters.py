"""
tune_hyperparameters.py
Controlled hyperparameter tuning and ablation experiments for Multi-Head PhoBERT ABSA.
Executes systematic comparisons where only one group of parameters is varied at a time.
Generates comparative evaluation tables and saves the best model configuration.
Author: Nhom 2 - Thanh vien 3
"""

import os
import sys
import copy
import json
import time
import argparse
import pandas as pd

from train_phobert import train, parse_args


# Definition of controlled experimental configurations
CONTROLLED_EXPERIMENTS = [
    # 1. Baseline Configuration
    {
        'experiment_name': 'exp0_baseline',
        'group': 'Baseline',
        'description': 'PhoBERT-base standard setup',
        'params': {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'max_len': 160,
            'weight_decay': 0.01,
            'dropout': 0.1,
            'warmup_ratio': 0.1,
            'use_class_weights': True,
            'epochs': 4
        }
    },
    # 2. Group 1: Learning Rate Impact (LR lower vs higher)
    {
        'experiment_name': 'exp1a_lr_1e5',
        'group': 'Learning Rate',
        'description': 'Lower learning rate (1e-5)',
        'params': {
            'learning_rate': 1e-5,
            'batch_size': 16,
            'max_len': 160,
            'weight_decay': 0.01,
            'dropout': 0.1,
            'warmup_ratio': 0.1,
            'use_class_weights': True,
            'epochs': 4
        }
    },
    {
        'experiment_name': 'exp1b_lr_3e5',
        'group': 'Learning Rate',
        'description': 'Higher learning rate (3e-5)',
        'params': {
            'learning_rate': 3e-5,
            'batch_size': 16,
            'max_len': 160,
            'weight_decay': 0.01,
            'dropout': 0.1,
            'warmup_ratio': 0.1,
            'use_class_weights': True,
            'epochs': 4
        }
    },
    # 3. Group 2: Batch Size & Max Length
    {
        'experiment_name': 'exp2a_batch_32',
        'group': 'Batch Size & Length',
        'description': 'Larger batch size (32)',
        'params': {
            'learning_rate': 2e-5,
            'batch_size': 32,
            'max_len': 160,
            'weight_decay': 0.01,
            'dropout': 0.1,
            'warmup_ratio': 0.1,
            'use_class_weights': True,
            'epochs': 4
        }
    },
    {
        'experiment_name': 'exp2b_maxlen_128',
        'group': 'Batch Size & Length',
        'description': 'Shorter max sequence length (128)',
        'params': {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'max_len': 128,
            'weight_decay': 0.01,
            'dropout': 0.1,
            'warmup_ratio': 0.1,
            'use_class_weights': True,
            'epochs': 4
        }
    },
    # 4. Group 3: Regularization & Ablation Study
    {
        'experiment_name': 'exp3a_reg_dropout02',
        'group': 'Regularization',
        'description': 'Higher dropout (0.2) & weight decay (0.05)',
        'params': {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'max_len': 160,
            'weight_decay': 0.05,
            'dropout': 0.2,
            'warmup_ratio': 0.1,
            'use_class_weights': True,
            'epochs': 4
        }
    },
    {
        'experiment_name': 'exp3b_ablation_no_weights',
        'group': 'Ablation Study',
        'description': 'Ablation: Without class weights (unweighted CE)',
        'params': {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'max_len': 160,
            'weight_decay': 0.01,
            'dropout': 0.1,
            'warmup_ratio': 0.1,
            'use_class_weights': False,
            'epochs': 4
        }
    }
]


def run_tuning(selected_exps=None, quick_mode=False):
    """Executes the suite of controlled experiments and tabulates results."""
    base_args = parse_args()

    experiments = CONTROLLED_EXPERIMENTS
    if selected_exps:
        experiments = [e for e in experiments if e['experiment_name'] in selected_exps]

    if quick_mode:
        print(">>> Quick mode active: limiting epochs to 2 for rapid verification.")
        for e in experiments:
            e['params']['epochs'] = 2

    results_table = []
    print(f"\n========================================================")
    print(f"Starting Hyperparameter Tuning Suite ({len(experiments)} Experiments)")
    print(f"========================================================\n")

    for i, exp in enumerate(experiments, 1):
        print(f"\n--- [{i}/{len(experiments)}] Running {exp['experiment_name']} ({exp['description']}) ---")
        exp_args = copy.deepcopy(base_args)
        exp_args.experiment_name = exp['experiment_name']

        # Apply specific parameters
        for k, v in exp['params'].items():
            setattr(exp_args, k, v)

        # Train and evaluate
        res = train(exp_args)

        results_table.append({
            'Experiment': exp['experiment_name'],
            'Group': exp['group'],
            'Description': exp['description'],
            'Learning_Rate': exp['params']['learning_rate'],
            'Batch_Size': exp['params']['batch_size'],
            'Max_Len': exp['params']['max_len'],
            'Dropout': exp['params']['dropout'],
            'Weight_Decay': exp['params']['weight_decay'],
            'Class_Weights': exp['params']['use_class_weights'],
            'Epochs': exp['params']['epochs'],
            'Best_Dev_Macro_F1': res['best_dev_macro_f1'],
            'Test_Macro_F1': res['test_macro_f1_4class'],
            'Test_Sentiment_F1': res['test_sentiment_macro_f1'],
            'Test_Exact_Match': res['test_exact_match'],
            'Test_ACD_F1': res['test_acd_f1'],
            'Latency_ms': res['latency_per_sample_ms'],
            'Peak_VRAM_MB': res['peak_vram_mb'],
            'Train_Time_s': res['total_train_time_sec']
        })

    # Save summary table
    df_results = pd.DataFrame(results_table)
    summary_path = os.path.join(base_args.log_dir, 'hyperparameter_comparison.csv')
    df_results.to_csv(summary_path, index=False)
    print(f"\n>>> Hyperparameter comparison saved to: {summary_path}")

    # Identify best configuration
    best_row = df_results.sort_values(by='Best_Dev_Macro_F1', ascending=False).iloc[0]
    print("\n========================================================")
    print(f"WINNING CONFIGURATION (Selected by Dev Macro F1):")
    print(f"  Experiment: {best_row['Experiment']} ({best_row['Description']})")
    print(f"  Dev Macro F1: {best_row['Best_Dev_Macro_F1']:.4f}")
    print(f"  Test Macro F1: {best_row['Test_Macro_F1']:.4f}")
    print(f"  Test Sentiment F1: {best_row['Test_Sentiment_F1']:.4f}")
    print(f"  Inference Latency: {best_row['Latency_ms']:.2f} ms/sample")
    print(f"  Peak VRAM: {best_row['Peak_VRAM_MB']:.0f} MB")
    print("========================================================\n")

    return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run in quick mode (2 epochs per config)')
    parser.add_argument('--exp', type=str, nargs='*', help='Specific experiment names to run')
    cmd_args = parser.parse_args()

    run_tuning(selected_exps=cmd_args.exp, quick_mode=cmd_args.quick)
