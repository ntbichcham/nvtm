"""
train_phobert.py
Training pipeline for Multi-Head PhoBERT on UIT-ViSFD dataset.
Includes: AdamW, LR scheduler with warmup, weighted loss, mixed precision (AMP),
early stopping, full metric logging, and model checkpointing.
Author: Nhom 2 - Thanh vien 3
"""

import os
import sys
import time
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

# Import local modules
from dataset_loader import get_dataloaders, ASPECT_LIST, NUM_CLASSES
from model_phobert import MultiHeadPhoBERTForABSA


def set_seed(seed=42):
    """Sets random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_absa_metrics(all_preds, all_labels):
    """
    Computes ABSA evaluation metrics across 10 aspects.
    all_preds: (N, 10) with values in {0, 1, 2, 3}
    all_labels: (N, 10) with values in {0, 1, 2, 3}
    """
    flat_preds = all_preds.flatten()
    flat_labels = all_labels.flatten()

    # Exact Match Ratio (Subset Accuracy): all 10 heads correct for a sample
    exact_match = (all_preds == all_labels).all(axis=1).mean()

    # Overall 4-class Micro and Macro F1
    macro_f1_4class = f1_score(flat_labels, flat_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(flat_labels, flat_preds, average='micro', zero_division=0)

    # Sentiment F1 (classes 1: Neg, 2: Neu, 3: Pos - excluding None=0)
    mask_sentiment = (flat_labels > 0) | (flat_preds > 0)
    if mask_sentiment.sum() > 0:
        sentiment_macro_f1 = f1_score(
            flat_labels[mask_sentiment],
            flat_preds[mask_sentiment],
            labels=[1, 2, 3],
            average='macro',
            zero_division=0
        )
    else:
        sentiment_macro_f1 = 0.0

    # Aspect Category Detection (ACD) binary F1 (is aspect mentioned or not: 0 vs >0)
    bin_preds = (all_preds > 0).astype(int).flatten()
    bin_labels = (all_labels > 0).astype(int).flatten()
    acd_f1 = f1_score(bin_labels, bin_preds, average='binary', zero_division=0)

    # Per-aspect F1 breakdown
    per_aspect_f1 = {}
    for i, aspect in enumerate(ASPECT_LIST):
        asp_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='macro', zero_division=0)
        per_aspect_f1[aspect] = asp_f1

    return {
        'exact_match': exact_match,
        'macro_f1_4class': macro_f1_4class,
        'sentiment_macro_f1': sentiment_macro_f1,
        'micro_f1': micro_f1,
        'acd_f1': acd_f1,
        'per_aspect_f1': per_aspect_f1
    }


def evaluate(model, dataloader, class_weights, device):
    """Evaluates the model on validation / test split."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels=labels, class_weights=class_weights)
            total_loss += outputs['loss'].item()

            logits = outputs['logits'] # (B, 10, 4)
            preds = torch.argmax(logits, dim=-1) # (B, 10)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_absa_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss
    return metrics, all_preds, all_labels


def train(args):
    """Main training loop."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"=== Starting Training Multi-Head PhoBERT ABSA ===")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from: {args.data_dir}")
    train_loader, dev_loader, test_loader, class_weights, tokenizer = get_dataloaders(
        data_dir=args.data_dir,
        tokenizer_name=args.pretrained_model,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers
    )

    if not args.use_class_weights:
        class_weights = None
        print("Class weighting disabled.")
    else:
        print("Class weighting enabled (smoothed inverse frequency).")

    # 2. Build Model
    model = MultiHeadPhoBERTForABSA(
        pretrained_model_name=args.pretrained_model,
        num_aspects=len(ASPECT_LIST),
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        dense_head=args.dense_head
    ).to(device)

    param_info = model.count_parameters()
    print(f"Total Parameters: {param_info['total_parameters']:,} (Trainable: {param_info['trainable_parameters']:,})")

    # 3. Optimizer & Weight Decay handling
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    accum = max(1, getattr(args, 'grad_accum', 1))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(args.fp16 and device.type == 'cuda'))

    best_dev_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=(args.fp16 and device.type == 'cuda')):
                outputs = model(input_ids, attention_mask, labels=labels, class_weights=class_weights)
                loss = outputs['loss']

            # Scale loss so that accumulation reproduces the larger effective batch
            scaler.scale(loss / accum).backward()
            train_loss += loss.item()
            pbar.set_postfix({'train_loss': f"{loss.item():.4f}"})

            is_last = (step + 1) == len(train_loader)
            if (step + 1) % accum == 0 or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scale_after = scaler.get_scale()
                # Only step scheduler if optimizer actually updated (no inf/nan gradient skipped)
                if scale_before <= scale_after:
                    scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Validation on Dev
        dev_metrics, _, _ = evaluate(model, dev_loader, class_weights, device)

        # GPU Peak Memory
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # Log epoch result
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'dev_loss': dev_metrics['loss'],
            'dev_macro_f1_4class': dev_metrics['macro_f1_4class'],
            'dev_sentiment_macro_f1': dev_metrics['sentiment_macro_f1'],
            'dev_micro_f1': dev_metrics['micro_f1'],
            'dev_exact_match': dev_metrics['exact_match'],
            'dev_acd_f1': dev_metrics['acd_f1'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_sec': epoch_time,
            'peak_vram_mb': vram_mb
        })

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Dev Loss: {dev_metrics['loss']:.4f} | "
            f"Dev Macro F1 (4cl): {dev_metrics['macro_f1_4class']:.4f} | "
            f"Dev Sentiment F1: {dev_metrics['sentiment_macro_f1']:.4f} | "
            f"Dev EM: {dev_metrics['exact_match']:.4f} | "
            f"Time: {epoch_time:.1f}s | "
            f"VRAM: {vram_mb:.0f}MB"
        )

        # Monitor metric: dev_macro_f1_4class
        monitored_metric = dev_metrics['macro_f1_4class']
        if monitored_metric > best_dev_macro_f1:
            best_dev_macro_f1 = monitored_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            best_checkpoint_path = os.path.join(args.output_dir, 'best_phobert_absa.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dev_macro_f1': best_dev_macro_f1,
                'args': vars(args),
                'metrics': dev_metrics
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Macro F1: {best_dev_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} (Patience: {args.patience}).")
                break

    total_train_time = time.time() - start_train_time
    print(f"\nTraining completed in {total_train_time:.1f}s. Best Epoch: {best_epoch} with Dev Macro F1: {best_dev_macro_f1:.4f}")

    # 4. Save Training History & Config
    history_df = pd.DataFrame(history)
    log_csv_path = os.path.join(args.log_dir, f"train_log_{args.experiment_name}.csv")
    history_df.to_csv(log_csv_path, index=False)
    print(f"Training log saved to {log_csv_path}")

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 5. Load best checkpoint and evaluate on Test set
    print("\n=== Evaluating Best Checkpoint on Test Set ===")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Measure test inference latency
    test_start = time.perf_counter()
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, class_weights, device)
    test_elapsed = time.perf_counter() - test_start
    total_test_samples = len(test_loader.dataset)
    latency_per_sample_ms = (test_elapsed / total_test_samples) * 1000
    throughput_samples_sec = total_test_samples / test_elapsed

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Exact Match (EM): {test_metrics['exact_match']:.4f}")
    print(f"Test Macro F1 (4-class): {test_metrics['macro_f1_4class']:.4f}")
    print(f"Test Sentiment Macro F1: {test_metrics['sentiment_macro_f1']:.4f}")
    print(f"Test Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"Test Aspect Detection (ACD) F1: {test_metrics['acd_f1']:.4f}")
    print(f"Inference Latency: {latency_per_sample_ms:.2f} ms/sample | Throughput: {throughput_samples_sec:.1f} samples/sec")

    # Save test metrics
    test_results = {
        'experiment_name': args.experiment_name,
        'best_epoch': best_epoch,
        'best_dev_macro_f1': best_dev_macro_f1,
        'test_loss': test_metrics['loss'],
        'test_exact_match': test_metrics['exact_match'],
        'test_macro_f1_4class': test_metrics['macro_f1_4class'],
        'test_sentiment_macro_f1': test_metrics['sentiment_macro_f1'],
        'test_micro_f1': test_metrics['micro_f1'],
        'test_acd_f1': test_metrics['acd_f1'],
        'latency_per_sample_ms': latency_per_sample_ms,
        'throughput_samples_sec': throughput_samples_sec,
        'total_train_time_sec': total_train_time,
        'peak_vram_mb': vram_mb,
        'per_aspect_test_f1': test_metrics['per_aspect_f1']
    }
    test_results_path = os.path.join(args.log_dir, f"test_results_{args.experiment_name}.json")
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"Test results saved to {test_results_path}")

    return test_results


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Head PhoBERT Training for ABSA")
    parser.add_argument('--data_dir', type=str, default='ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model')
    parser.add_argument('--pretrained_model', type=str, default='vinai/phobert-base')
    parser.add_argument('--output_dir', type=str, default='ABSA_Project/models')
    parser.add_argument('--log_dir', type=str, default='ABSA_Project/results/evaluation')
    parser.add_argument('--experiment_name', type=str, default='phobert_baseline')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=160)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')

    # Architectural & Hardware Options
    parser.add_argument('--dense_head', action='store_true', help='Use 2-layer dense head instead of single linear')
    parser.add_argument('--use_class_weights', action='store_true', default=True, help='Enable smoothed class weighting')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable AMP fp16 mixed precision')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU execution')
    parser.add_argument('--num_workers', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
