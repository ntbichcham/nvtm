"""
dataset_loader.py
Dataset and DataLoader utilities for UIT-ViSFD Aspect-Based Sentiment Analysis.
Author: Nhom 2 - Thanh vien 3
"""

import re
import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 10 Aspect categories in UIT-ViSFD dataset
ASPECT_LIST = [
    'BATTERY',
    'CAMERA',
    'DESIGN',
    'FEATURES',
    'GENERAL',
    'PERFORMANCE',
    'PRICE',
    'SCREEN',
    'SER&ACC',
    'STORAGE'
]

ASPECT2IDX = {asp: i for i, asp in enumerate(ASPECT_LIST)}
IDX2ASPECT = {i: asp for i, asp in enumerate(ASPECT_LIST)}

# Sentiment polarities: 0: None, 1: Negative, 2: Neutral, 3: Positive
POLARITY2IDX = {
    'Negative': 1,
    'Neutral': 2,
    'Positive': 3
}
IDX2POLARITY = {
    0: 'None',
    1: 'Negative',
    2: 'Neutral',
    3: 'Positive'
}
NUM_CLASSES = 4


def parse_label_string(label_str):
    """
    Parses UIT-ViSFD label format like '{CAMERA#Positive};{BATTERY#Negative};'
    Returns a numpy array of shape (10,) with values in {0, 1, 2, 3}.
    """
    labels = np.zeros(len(ASPECT_LIST), dtype=np.int64)
    if not isinstance(label_str, str) or not label_str.strip():
        return labels

    matches = re.findall(r'\{([A-Za-z0-9_&]+)#([A-Za-z0-9_]+)\}', label_str)
    for aspect, polarity in matches:
        if aspect in ASPECT2IDX and polarity in POLARITY2IDX:
            labels[ASPECT2IDX[aspect]] = POLARITY2IDX[polarity]
    return labels


class ViSFDABSADataset(Dataset):
    """
    PyTorch Dataset for UIT-ViSFD Aspect-Based Sentiment Analysis.
    Compatible with preprocessed segmented data from Thanh vien 1.
    """
    def __init__(self, df, tokenizer, max_len=160, text_col='comment_segmented', label_col='label'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df[text_col].astype(str).tolist()
        self.labels = [parse_label_string(l) for l in df[label_col].tolist()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),        # (max_len,)
            'attention_mask': encoding['attention_mask'].squeeze(0), # (max_len,)
            'labels': torch.tensor(target, dtype=torch.long)       # (10,)
        }


def compute_smoothed_class_weights(df, label_col='label', num_classes=4, power=0.5):
    """
    Computes smoothed class weights for each of the 10 heads to address class imbalance.
    weight = (total_samples / (num_classes * count))^power, then normalized.
    """
    all_labels = np.array([parse_label_string(l) for l in df[label_col].tolist()])
    total_samples = len(all_labels)

    head_weights = []
    for k in range(len(ASPECT_LIST)):
        counts = np.bincount(all_labels[:, k], minlength=num_classes)
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        # Smoothed inverse class frequency
        weights = (total_samples / (num_classes * counts)) ** power
        # Normalize weights so that mean is 1.0
        weights = weights / np.mean(weights)
        head_weights.append(torch.tensor(weights, dtype=torch.float32))

    return head_weights


def get_dataloaders(
    data_dir,
    tokenizer_name='vinai/phobert-base',
    batch_size=16,
    max_len=160,
    num_workers=0
):
    """
    Builds DataLoader objects for Train, Dev, and Test splits.
    Returns:
        train_loader, dev_loader, test_loader, class_weights, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    train_path = os.path.join(data_dir, 'Train_prepared.csv')
    dev_path = os.path.join(data_dir, 'Dev_prepared.csv')
    test_path = os.path.join(data_dir, 'Test_prepared.csv')

    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)
    df_test = pd.read_csv(test_path)

    # Compute class weights on training split
    class_weights = compute_smoothed_class_weights(df_train)

    train_dataset = ViSFDABSADataset(df_train, tokenizer, max_len=max_len)
    dev_dataset = ViSFDABSADataset(df_dev, tokenizer, max_len=max_len)
    test_dataset = ViSFDABSADataset(df_test, tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, dev_loader, test_loader, class_weights, tokenizer


if __name__ == '__main__':
    # Unit test / sanity check
    data_dir = 'ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model'
    print("Testing get_dataloaders...")
    train_loader, dev_loader, test_loader, weights, tok = get_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        max_len=64
    )
    print(f"Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}, Test batches: {len(test_loader)}")
    sample_batch = next(iter(train_loader))
    print("input_ids shape:", sample_batch['input_ids'].shape)
    print("attention_mask shape:", sample_batch['attention_mask'].shape)
    print("labels shape:", sample_batch['labels'].shape)
    print("Sample label row 0:", sample_batch['labels'][0].tolist())
    print("Class weights for head 0 (BATTERY):", weights[0])
    print("DataLoader verification passed successfully!")
