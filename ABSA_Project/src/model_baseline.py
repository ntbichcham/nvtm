"""
model_baseline.py
BiLSTM and BiGRU Multi-Head baseline models for ABSA comparison.
Used for theoretical complexity analysis and empirical benchmarking against PhoBERT.
Author: Nhom 2 - Thanh vien 3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMMultiHeadForABSA(nn.Module):
    """
    Bidirectional LSTM with 10 classification heads for ABSA.
    Architecture:
        Embedding (vocab_size -> emb_dim)
        Spatial Dropout
        Bidirectional LSTM (num_layers, hidden_dim)
        Self-Attention / Max-Pooling over time
        10 Linear Classification Heads (each with 4 classes)
    """
    def __init__(
        self,
        vocab_size=64000,
        emb_dim=256,
        hidden_dim=256,
        num_layers=2,
        num_aspects=10,
        num_classes=4,
        dropout=0.2
    ):
        super(BiLSTMMultiHeadForABSA, self).__init__()
        self.num_aspects = num_aspects
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        rep_dim = hidden_dim * 2 # Bidirectional

        # Attention pooling layer
        self.att_dense = nn.Linear(rep_dim, 1)

        # 10 Classification Heads
        self.heads = nn.ModuleList([
            nn.Linear(rep_dim, num_classes) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask=None, labels=None, class_weights=None):
        # input_ids: (B, seq_len)
        emb = self.embedding(input_ids) # (B, seq_len, emb_dim)
        emb = self.dropout(emb)

        lstm_out, _ = self.lstm(emb) # (B, seq_len, rep_dim)

        # Masked Attention Pooling
        att_scores = self.att_dense(lstm_out).squeeze(-1) # (B, seq_len)
        if attention_mask is not None:
            att_scores = att_scores.masked_fill(
                attention_mask == 0, torch.finfo(att_scores.dtype).min)
        att_weights = F.softmax(att_scores, dim=-1).unsqueeze(-1) # (B, seq_len, 1)
        rep = torch.sum(lstm_out * att_weights, dim=1) # (B, rep_dim)
        rep = self.dropout(rep)

        head_logits = [head(rep) for head in self.heads]
        logits = torch.stack(head_logits, dim=1) # (B, 10, 4)

        loss = None
        if labels is not None:
            total_loss = 0.0
            for k in range(self.num_aspects):
                target = labels[:, k]
                weight = class_weights[k].to(input_ids.device) if class_weights is not None else None
                total_loss += F.cross_entropy(logits[:, k, :], target, weight=weight)
            loss = total_loss

        return {
            'logits': logits,
            'loss': loss
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = sum(p.numel() for p in self.embedding.parameters())
        lstm = sum(p.numel() for p in self.lstm.parameters())
        heads = sum(p.numel() for p in self.heads.parameters())
        return {
            'total_parameters': total,
            'trainable_parameters': trainable,
            'embedding_parameters': emb,
            'lstm_parameters': lstm,
            'heads_parameters': heads
        }


def compare_complexity_formulas():
    """
    Theoretical complexity calculation comparison between Transformer (PhoBERT-base)
    and BiLSTM/BiGRU baseline.
    """
    return {
        "PhoBERT-base": {
            "Layers": 12,
            "Hidden_dim (d)": 768,
            "Heads (A)": 12,
            "FFN_dim (d_ff)": 3072,
            "Vocab (V)": 64000,
            "Time_Complexity_per_layer": "O(N^2 * d + N * d^2)",
            "Self_Attention_Complexity": "O(N^2 * d)",
            "Memory_Complexity": "O(A * N^2 + N * d)",
            "Sequential_Operations": "O(1) - Fully parallel across sequence length N",
            "Max_Path_Length": "O(1)"
        },
        "BiLSTM": {
            "Layers": 2,
            "Hidden_dim (d)": 256,
            "Time_Complexity_per_layer": "O(N * d^2)",
            "Memory_Complexity": "O(N * d)",
            "Sequential_Operations": "O(N) - Strictly sequential unrolling over steps",
            "Max_Path_Length": "O(N)"
        },
        "BiGRU": {
            "Layers": 2,
            "Hidden_dim (d)": 256,
            "Time_Complexity_per_layer": "O(N * d^2)",
            "Memory_Complexity": "O(N * d)",
            "Sequential_Operations": "O(N) - Strictly sequential unrolling over steps",
            "Max_Path_Length": "O(N)"
        }
    }


if __name__ == '__main__':
    model = BiLSTMMultiHeadForABSA()
    params = model.count_parameters()
    print("BiLSTM Baseline Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    comp = compare_complexity_formulas()
    print("\nTheoretical Complexity Breakdown:")
    for m, details in comp.items():
        print(f"[{m}]:")
        for k, v in details.items():
            print(f"    {k}: {v}")
