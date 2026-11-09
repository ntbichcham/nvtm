"""
baseline_data.py
Word-level vocabulary and Dataset/DataLoader utilities for the BiLSTM baseline.
The baseline uses whitespace-separated word tokens from `comment_segmented`
(the same pre-segmented text PhoBERT receives), so both models see the same
input content and the same labels/splits.
Author: Nhom 2 - Thanh vien 3
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from dataset_loader import parse_label_string, compute_smoothed_class_weights, ASPECT_LIST

PAD_IDX = 1
UNK_IDX = 0
SPECIAL_TOKENS = ['<unk>', '<pad>']


def tokenize_text(text):
    """Whitespace tokenization of already word-segmented Vietnamese text."""
    if not isinstance(text, str):
        return []
    return text.lower().split()


def build_vocab(texts, min_freq=2, max_size=64000):
    """Builds a word vocabulary. Index 1 is reserved for padding (model padding_idx=1)."""
    counter = Counter()
    for t in texts:
        counter.update(tokenize_text(t))

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tok, freq in counter.most_common():
        if len(vocab) >= max_size:
            break
        if freq < min_freq:
            break
        vocab[tok] = len(vocab)
    return vocab


def encode_text(text, vocab, max_len):
    """Encodes one text into ids + attention mask, truncating/padding to max_len."""
    toks = tokenize_text(text)[:max_len]
    ids = [vocab.get(t, UNK_IDX) for t in toks]
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    if pad > 0:
        ids += [PAD_IDX] * pad
        mask += [0] * pad
    return ids, mask


class BaselineABSADataset(Dataset):
    """Pre-tokenized dataset for the BiLSTM baseline (no tokenizer overhead per epoch)."""

    def __init__(self, df, vocab, max_len=128, text_col='comment_segmented', label_col='label'):
        self.ids = []
        self.masks = []
        self.labels = []
        for text, label in zip(df[text_col].astype(str).tolist(), df[label_col].tolist()):
            ids, mask = encode_text(text, vocab, max_len)
            self.ids.append(ids)
            self.masks.append(mask)
            self.labels.append(parse_label_string(label))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def get_baseline_dataloaders(
    data_dir,
    vocab=None,
    max_len=128,
    batch_size=32,
    num_workers=0,
    vocab_min_freq=2,
    vocab_size=64000,
    seed=42
):
    """
    Builds Train/Dev/Test loaders with a vocabulary fitted on Train only.
    Returns loaders, class weights, vocab and split sizes.
    """
    df_train = pd.read_csv(os.path.join(data_dir, 'Train_prepared.csv'))
    df_dev = pd.read_csv(os.path.join(data_dir, 'Dev_prepared.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'Test_prepared.csv'))

    if vocab is None:
        vocab = build_vocab(df_train['comment_segmented'].astype(str), vocab_min_freq, vocab_size)

    class_weights = compute_smoothed_class_weights(df_train)

    train_ds = BaselineABSADataset(df_train, vocab, max_len=max_len)
    dev_ds = BaselineABSADataset(df_dev, vocab, max_len=max_len)
    test_ds = BaselineABSADataset(df_test, vocab, max_len=max_len)

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, generator=generator)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    return train_loader, dev_loader, test_loader, class_weights, vocab


def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    data_dir = 'ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model'
    tr, dv, te, cw, voc = get_baseline_dataloaders(data_dir, max_len=128, batch_size=4)
    print(f'Vocab size: {len(voc):,}')
    print(f'Train/Dev/Test batches: {len(tr)}/{len(dv)}/{len(te)}')
    batch = next(iter(tr))
    print('input_ids', tuple(batch['input_ids'].shape),
          'mask', tuple(batch['attention_mask'].shape),
          'labels', tuple(batch['labels'].shape))
    print('First label row:', batch['labels'][0].tolist())
