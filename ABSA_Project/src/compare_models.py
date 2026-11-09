"""
compare_models.py
Builds the PhoBERT vs BiLSTM quality comparison table from the unified
evaluation JSON files produced by evaluate_quality.py.
Author: Nhom 2 - Thanh vien 3
"""

import os
import json
import argparse
import pandas as pd

FIELDS = [
    ('Tham số', 'total_parameters', 'int'),
    ('Micro Precision', 'micro_precision', 'pct'),
    ('Micro Recall', 'micro_recall', 'pct'),
    ('Micro F1', 'micro_f1', 'pct'),
    ('Macro Precision', 'macro_precision', 'pct'),
    ('Macro Recall', 'macro_recall', 'pct'),
    ('Macro F1', 'macro_f1', 'pct'),
    ('Weighted F1', 'weighted_f1', 'pct'),
    ('Sentiment Macro F1', 'sentiment_macro_f1', 'pct'),
    ('ACD F1', 'acd_f1', 'pct'),
    ('Exact Match', 'exact_match', 'pct'),
    ('Độ trễ (ms/mẫu)', 'latency_per_sample_ms', 'float'),
    ('Thông lượng (mẫu/s)', 'throughput_samples_sec', 'float'),
]


def fmt(value, kind):
    if value is None:
        return '—'
    if kind == 'int':
        return f'{int(value):,}'.replace(',', '.')
    if kind == 'pct':
        return f'{value * 100:.2f}'
    return f'{value:.2f}'


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phobert', required=True)
    parser.add_argument('--bilstm', required=True)
    parser.add_argument('--out', default='ABSA_Project/results/evaluation/model_quality_comparison.csv')
    args = parser.parse_args()

    pho = load(args.phobert)
    bil = load(args.bilstm)

    rows = []
    for label, key, kind in FIELDS:
        rows.append({'Chỉ số': label, 'Multi-Head PhoBERT': fmt(pho.get(key), kind),
                     'BiLSTM (từ đầu)': fmt(bil.get(key), kind)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')

    print(df.to_string(index=False))
    print('\nPer-aspect Macro-F1:')
    print(f"{'Aspect':12s} {'PhoBERT':>9s} {'BiLSTM':>9s} {'Delta':>8s}")
    for asp in pho['per_aspect']:
        p = pho['per_aspect'][asp]['f1_macro'] * 100
        b = bil['per_aspect'][asp]['f1_macro'] * 100
        print(f'{asp:12s} {p:9.2f} {b:9.2f} {p - b:+8.2f}')
    print('\nSaved:', args.out)


if __name__ == '__main__':
    main()
