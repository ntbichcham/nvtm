"""
evaluate_and_analyze.py
Comprehensive evaluation and error analysis module for Multi-Head PhoBERT ABSA.
Computes Precision, Recall, F1-score (Micro/Macro/Weighted), Confusion Matrices,
and extracts/classifies error cases (False Positive, False Negative, Sentiment Misclassification).
Author: Nhom 2 - Thanh vien 3
"""

import os
import sys

# Configure UTF-8 stdout for Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn.functional as F

from dataset_loader import get_dataloaders, ASPECT_LIST, NUM_CLASSES, IDX2POLARITY
from model_phobert import MultiHeadPhoBERTForABSA


def run_evaluation_and_error_analysis(
    checkpoint_path='ABSA_Project/models/best_phobert_absa.pt',
    data_dir='ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model',
    output_dir='ABSA_Project/results/evaluation',
    split='Test',
    max_len=128,
    batch_size=16
):
    """
    Evaluates a trained checkpoint on a specified split (Dev or Test),
    computes detailed metrics, exports confusion matrices, and extracts error cases.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Comprehensive Evaluation & Error Analysis ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split} | Max Sequence Length: {max_len}")

    # 1. Load Data
    _, dev_loader, test_loader, class_weights, tokenizer = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        max_len=max_len
    )
    dataloader = dev_loader if split.lower() == 'dev' else test_loader

    # Load original raw/prepared dataframe to retrieve text for error audit
    df_raw = pd.read_csv(os.path.join(data_dir, f"{split.capitalize()}_prepared.csv"))

    # 2. Load Model
    model = MultiHeadPhoBERTForABSA(num_aspects=len(ASPECT_LIST), num_classes=NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] # (B, 10, 4)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0) # (N, 10)
    all_labels = np.concatenate(all_labels, axis=0) # (N, 10)
    all_probs = np.concatenate(all_probs, axis=0) # (N, 10, 4)

    # 3. Compute Comprehensive Metrics
    aspect_metrics = []
    confusion_matrices = {}

    for i, aspect in enumerate(ASPECT_LIST):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]

        # 4-class metrics
        acc = accuracy_score(y_true, y_pred)
        p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_wgt = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Aspect Category Detection (binary: is aspect present >0)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        acd_p = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        acd_r = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        acd_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        # Aspect Sentiment (classes 1, 2, 3 where present)
        mask = (y_true > 0) | (y_pred > 0)
        if mask.sum() > 0:
            asc_f1 = f1_score(y_true[mask], y_pred[mask], labels=[1, 2, 3], average='macro', zero_division=0)
        else:
            asc_f1 = 0.0

        # Confusion Matrix (4x4: None, Neg, Neu, Pos)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        confusion_matrices[aspect] = cm

        aspect_metrics.append({
            'Aspect': aspect,
            'Accuracy': acc,
            'Precision_Macro': p_macro,
            'Recall_Macro': r_macro,
            'F1_Macro_4Class': f1_mac,
            'F1_Weighted': f1_wgt,
            'ACD_Precision': acd_p,
            'ACD_Recall': acd_r,
            'ACD_F1': acd_f1,
            'ASC_Sentiment_F1': asc_f1,
            'Support_None': int((y_true == 0).sum()),
            'Support_Negative': int((y_true == 1).sum()),
            'Support_Neutral': int((y_true == 2).sum()),
            'Support_Positive': int((y_true == 3).sum()),
            'Total_Samples': len(y_true)
        })

    df_metrics = pd.DataFrame(aspect_metrics)
    metrics_path = os.path.join(output_dir, f"detailed_metrics_{split.lower()}.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nDetailed classification report saved to: {metrics_path}")

    # Summary overall metrics
    flat_true = all_labels.flatten()
    flat_pred = all_preds.flatten()
    overall_exact_match = (all_preds == all_labels).all(axis=1).mean()
    overall_macro_f1 = f1_score(flat_true, flat_pred, average='macro', zero_division=0)
    overall_micro_f1 = f1_score(flat_true, flat_pred, average='micro', zero_division=0)
    overall_macro_p = precision_score(flat_true, flat_pred, average='macro', zero_division=0)
    overall_macro_r = recall_score(flat_true, flat_pred, average='macro', zero_division=0)

    print("\n=== Overall Evaluation Summary ===")
    print(f"  Exact Match Ratio (All 10 heads): {overall_exact_match:.4f} ({overall_exact_match*100:.2f}%)")
    print(f"  Overall Macro F1 (4-class):       {overall_macro_f1:.4f}")
    print(f"  Overall Micro F1:                 {overall_micro_f1:.4f}")
    print(f"  Overall Macro Precision:          {overall_macro_p:.4f}")
    print(f"  Overall Macro Recall:             {overall_macro_r:.4f}")

    # 4. Save Confusion Matrices
    cm_rows = []
    classes = ['None', 'Negative', 'Neutral', 'Positive']
    for asp, cm in confusion_matrices.items():
        for true_idx, true_cls in enumerate(classes):
            for pred_idx, pred_cls in enumerate(classes):
                cm_rows.append({
                    'Aspect': asp,
                    'True_Class': true_cls,
                    'Pred_Class': pred_cls,
                    'Count': int(cm[true_idx, pred_idx])
                })
    df_cm = pd.DataFrame(cm_rows)
    cm_path = os.path.join(output_dir, f"confusion_matrices_{split.lower()}.csv")
    df_cm.to_csv(cm_path, index=False)
    print(f"Confusion matrices saved to: {cm_path}")

    # 5. Extract and Categorize Error Cases (Phân tích lỗi)
    error_records = []
    for row_idx in range(len(all_labels)):
        comment_text = df_raw.iloc[row_idx]['comment']
        segmented_text = df_raw.iloc[row_idx].get('comment_segmented', '')
        original_label = df_raw.iloc[row_idx].get('label', '')

        for asp_idx, aspect in enumerate(ASPECT_LIST):
            y_t = all_labels[row_idx, asp_idx]
            y_p = all_preds[row_idx, asp_idx]

            if y_t != y_p:
                true_label_name = IDX2POLARITY[y_t]
                pred_label_name = IDX2POLARITY[y_p]
                confidence = float(all_probs[row_idx, asp_idx, y_p])

                # Error categorization
                if y_t == 0 and y_p > 0:
                    error_type = 'False Positive (Nhận diện khía cạnh thừa)'
                elif y_t > 0 and y_p == 0:
                    error_type = 'False Negative (Bỏ sót khía cạnh)'
                else:
                    error_type = 'Sentiment Misclassification (Sai cực tính cảm xúc)'

                error_records.append({
                    'sample_index': row_idx,
                    'comment': comment_text,
                    'comment_segmented': segmented_text,
                    'aspect': aspect,
                    'true_label': true_label_name,
                    'pred_label': pred_label_name,
                    'confidence': confidence,
                    'error_type': error_type,
                    'full_annotated_label': original_label
                })

    df_errors = pd.DataFrame(error_records)
    errors_path = os.path.join(output_dir, f"error_analysis_{split.lower()}.csv")
    df_errors.to_csv(errors_path, index=False, encoding='utf-8-sig')
    print(f"Error analysis cases saved to: {errors_path} (Total error instances: {len(df_errors)})")

    # Error summary distribution
    error_dist = df_errors['error_type'].value_counts()
    print("\n=== Phân bố các dạng lỗi ===")
    for etype, cnt in error_dist.items():
        print(f"  - {etype}: {cnt} trường hợp ({cnt/len(df_errors)*100:.2f}%)")

    return df_metrics, df_errors, confusion_matrices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='ABSA_Project/models/best_phobert_absa.pt')
    parser.add_argument('--data_dir', type=str, default='ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model')
    parser.add_argument('--split', type=str, default='Test')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    run_evaluation_and_error_analysis(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        max_len=args.max_len,
        batch_size=args.batch_size
    )
