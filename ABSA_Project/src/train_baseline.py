"""
train_baseline.py
Trains and evaluates the BiLSTM multi-head baseline on exactly the same
Train/Dev/Test splits, labels, class weights and metric code as PhoBERT.
The model is trained from scratch (no pretrained word embeddings), which is
the standard reference point when arguing for a pretrained Transformer.
Author: Nhom 2 - Thanh vien 3
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from dataset_loader import ASPECT_LIST, NUM_CLASSES
from baseline_data import get_baseline_dataloaders, save_vocab
from model_baseline import BiLSTMMultiHeadForABSA
from metrics_utils import compute_full_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, class_weights, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, labels=labels, class_weights=class_weights)
            total_loss += outputs['loss'].item()
            preds = torch.argmax(outputs['logits'], dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_full_metrics(preds, labels)
    metrics['loss'] = avg_loss
    return metrics, preds, labels


def measure_latency(model, dataloader, device, warmup_batches=5):
    """Inference-only latency (data already encoded), after warmup."""
    model.eval()
    batches = list(dataloader)
    n_warm = min(warmup_batches, len(batches))
    with torch.no_grad():
        for b in batches[:n_warm]:
            model(b['input_ids'].to(device), b['attention_mask'].to(device))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    n_samples = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in batches:
            input_ids = b['input_ids'].to(device)
            mask = b['attention_mask'].to(device)
            model(input_ids, mask)
            n_samples += input_ids.size(0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        'latency_per_sample_ms': elapsed / n_samples * 1000,
        'throughput_samples_sec': n_samples / elapsed,
        'timed_samples': n_samples
    }


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print('=== BiLSTM Multi-Head Baseline (from scratch) ===')
    print(f'Device: {device} | Seed: {args.seed}')
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, class_weights, vocab = get_baseline_dataloaders(
        data_dir=args.data_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        vocab_min_freq=args.vocab_min_freq,
        vocab_size=args.vocab_size,
        seed=args.seed
    )
    vocab_path = os.path.join(args.output_dir, f'vocab_{args.experiment_name}.json')
    save_vocab(vocab, vocab_path)
    print(f'Vocabulary: {len(vocab):,} tokens (fitted on Train, min_freq={args.vocab_min_freq})')

    model = BiLSTMMultiHeadForABSA(
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_aspects=len(ASPECT_LIST),
        num_classes=NUM_CLASSES,
        dropout=args.dropout
    ).to(device)
    param_info = model.count_parameters()
    print(f"Parameters: {param_info['total_parameters']:,}")

    no_decay = ['bias']
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ], lr=args.learning_rate, eps=1e-8)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = args.fp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_dev_f1 = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(input_ids, attention_mask, labels=labels, class_weights=class_weights)
                loss = outputs['loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scale_before <= scaler.get_scale():
                scheduler.step()
            running += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_train_loss = running / len(train_loader)
        epoch_time = time.time() - epoch_start

        dev_metrics, _, _ = evaluate(model, dev_loader, class_weights, device)
        vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'dev_loss': dev_metrics['loss'],
            'dev_macro_f1': dev_metrics['macro_f1'],
            'dev_micro_f1': dev_metrics['micro_f1'],
            'dev_sentiment_macro_f1': dev_metrics['sentiment_macro_f1'],
            'dev_acd_f1': dev_metrics['acd_f1'],
            'dev_exact_match': dev_metrics['exact_match'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_sec': epoch_time,
            'peak_vram_mb': vram
        })
        print(f"Epoch {epoch:02d} | Train {avg_train_loss:.4f} | Dev loss {dev_metrics['loss']:.4f} | "
              f"Dev MacroF1 {dev_metrics['macro_f1']:.4f} | SentF1 {dev_metrics['sentiment_macro_f1']:.4f} | "
              f"EM {dev_metrics['exact_match']:.4f} | {epoch_time:.1f}s | {vram:.0f}MB")

        if dev_metrics['macro_f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'epoch': epoch, 'model_state_dict': best_state,
                        'best_dev_macro_f1': best_dev_f1, 'vocab_size': len(vocab),
                        'args': vars(args)}, os.path.join(args.output_dir, f'best_bilstm_{args.experiment_name}.pt'))
            print(f'  --> new best (Dev Macro F1 {best_dev_f1:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch} (patience {args.patience}).')
                break

    total_train_time = time.time() - start
    print(f'Training finished in {total_train_time:.1f}s. Best epoch {best_epoch} (Dev Macro F1 {best_dev_f1:.4f})')

    history_df = pd.DataFrame(history)
    log_path = os.path.join(args.log_dir, f'train_log_{args.experiment_name}.csv')
    history_df.to_csv(log_path, index=False)

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, class_weights, device)
    latency = measure_latency(model, test_loader, device)

    print('=== Test (BiLSTM baseline) ===')
    for key in ['micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall',
                'macro_f1', 'weighted_f1', 'sentiment_macro_f1', 'acd_f1', 'exact_match']:
        print(f'  {key}: {test_metrics[key]:.4f}')

    result = {
        'experiment_name': args.experiment_name,
        'model': 'BiLSTM (from scratch)',
        'seed': args.seed,
        'best_epoch': best_epoch,
        'best_dev_macro_f1': best_dev_f1,
        'test_loss': test_metrics['loss'],
        'test_micro_precision': test_metrics['micro_precision'],
        'test_micro_recall': test_metrics['micro_recall'],
        'test_micro_f1': test_metrics['micro_f1'],
        'test_macro_precision': test_metrics['macro_precision'],
        'test_macro_recall': test_metrics['macro_recall'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_weighted_f1': test_metrics['weighted_f1'],
        'test_sentiment_macro_f1': test_metrics['sentiment_macro_f1'],
        'test_acd_f1': test_metrics['acd_f1'],
        'test_exact_match': test_metrics['exact_match'],
        'test_per_class': test_metrics['per_class'],
        'per_aspect_test_f1': {k: v['f1_macro'] for k, v in test_metrics['per_aspect'].items()},
        'total_parameters': param_info['total_parameters'],
        'vocab_size': len(vocab),
        'max_len': args.max_len,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'total_train_time_sec': total_train_time,
        'peak_vram_mb': torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
        'latency_per_sample_ms': latency['latency_per_sample_ms'],
        'throughput_samples_sec': latency['throughput_samples_sec']
    }
    json_path = os.path.join(args.log_dir, f'test_results_{args.experiment_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('Saved:', json_path)

    per_aspect_rows = [{'Aspect': k, **v} for k, v in test_metrics['per_aspect'].items()]
    pd.DataFrame(per_aspect_rows).to_csv(
        os.path.join(args.log_dir, f'detailed_metrics_{args.experiment_name}.csv'), index=False)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description='BiLSTM Multi-Head ABSA baseline')
    parser.add_argument('--data_dir', type=str,
                        default='ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model')
    parser.add_argument('--output_dir', type=str, default='ABSA_Project/models')
    parser.add_argument('--log_dir', type=str, default='ABSA_Project/results/evaluation')
    parser.add_argument('--experiment_name', type=str, default='bilstm_official')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vocab_min_freq', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=64000)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
