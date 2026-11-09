"""
evaluate_quality.py
Re-evaluates a trained checkpoint with one shared metric implementation,
so the PhoBERT vs BiLSTM comparison uses identical scoring code and splits.
Supports --model_type phobert | bilstm.
Author: Nhom 2 - Thanh vien 3
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset_loader import ASPECT_LIST, NUM_CLASSES
from baseline_data import BaselineABSADataset, load_vocab
from metrics_utils import compute_full_metrics
from model_phobert import MultiHeadPhoBERTForABSA
from model_baseline import BiLSTMMultiHeadForABSA


class PhoBERTFrameDataset(torch.utils.data.Dataset):
    """Pre-tokenizes once, then serves tensors (avoids tokenizer overhead in every epoch)."""

    def __init__(self, df, tokenizer, max_len, text_col='comment_segmented', label_col='label'):
        from dataset_loader import parse_label_string
        self.encodings = tokenizer(
            df[text_col].astype(str).tolist(),
            truncation=True, max_length=max_len, padding='max_length',
            return_token_type_ids=False, return_attention_mask=True
        )
        self.labels = [parse_label_string(l) for l in df[label_col].tolist()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def build_dataset(model_type, df, max_len, tokenizer_name='vinai/phobert-base', vocab=None):
    if model_type == 'phobert':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        return PhoBERTFrameDataset(df, tokenizer, max_len)
    return BaselineABSADataset(df, vocab, max_len=max_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['phobert', 'bilstm'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vocab', type=str, default=None)
    parser.add_argument('--data_dir', type=str,
                        default='ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model')
    parser.add_argument('--output_dir', type=str, default='ABSA_Project/results/evaluation')
    parser.add_argument('--split', type=str, default='Test')
    parser.add_argument('--max_len', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tokenizer', type=str, default='vinai/phobert-base')
    parser.add_argument('--out_prefix', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(os.path.join(args.data_dir, f'{args.split.capitalize()}_prepared.csv'))

    vocab = None
    if args.model_type == 'bilstm':
        assert args.vocab, '--vocab is required for bilstm'
        vocab = load_vocab(args.vocab)

    ds = build_dataset(args.model_type, df, args.max_len, args.tokenizer, vocab)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    if args.model_type == 'phobert':
        model = MultiHeadPhoBERTForABSA(num_aspects=len(ASPECT_LIST), num_classes=NUM_CLASSES)
    else:
        model = BiLSTMMultiHeadForABSA(vocab_size=len(vocab), num_aspects=len(ASPECT_LIST),
                                       num_classes=NUM_CLASSES)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.to(device).eval()

    all_preds, all_labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(input_ids, mask)['logits']
            all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.append(batch['labels'].numpy())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = compute_full_metrics(preds, labels)
    metrics['latency_per_sample_ms'] = elapsed / len(ds) * 1000
    metrics['throughput_samples_sec'] = len(ds) / elapsed
    metrics['model_type'] = args.model_type
    metrics['checkpoint'] = args.checkpoint
    metrics['max_len'] = args.max_len
    metrics['split'] = args.split
    metrics['best_epoch'] = ck.get('epoch')
    metrics['total_parameters'] = int(sum(p.numel() for p in model.parameters()))

    prefix = args.out_prefix or f'{args.model_type}_quality_{args.split.lower()}'
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f'{prefix}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    rows = [{'Aspect': k, **v} for k, v in metrics['per_aspect'].items()]
    csv_path = os.path.join(args.output_dir, f'{prefix}_per_aspect.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f'=== {args.model_type.upper()} | {args.split} | max_len={args.max_len} ===')
    for k in ['micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall',
              'macro_f1', 'weighted_f1', 'sentiment_macro_f1', 'acd_f1', 'exact_match']:
        print(f'  {k:24s} {metrics[k]:.6f}')
    print(f"  latency {metrics['latency_per_sample_ms']:.2f} ms/sample | "
          f"throughput {metrics['throughput_samples_sec']:.1f}/s")
    print('Saved:', json_path)


if __name__ == '__main__':
    main()
