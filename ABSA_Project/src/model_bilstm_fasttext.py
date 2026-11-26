"""
model_bilstm_fasttext.py
BiLSTM multi-head ABSA model whose embedding layer is initialised from
fastText vectors trained on the Train split.

The architecture, loss and heads are identical to
`model_baseline.BiLSTMMultiHeadForABSA`; the only difference is the embedding
matrix. That keeps the comparison "random embedding vs fastText embedding"
controlled.

Author: Nhom 2 - Thanh vien 3
"""

import numpy as np
import torch
import torch.nn as nn

from model_baseline import BiLSTMMultiHeadForABSA


class BiLSTMFastTextForABSA(BiLSTMMultiHeadForABSA):
    """BiLSTM multi-head ABSA with a fastText-initialised embedding layer."""

    def __init__(self, embedding_matrix, freeze=False, **kwargs):
        matrix = np.asarray(embedding_matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"embedding_matrix must be 2-D, got {matrix.shape}")

        emb_dim = int(matrix.shape[1])
        if "emb_dim" in kwargs and int(kwargs["emb_dim"]) != emb_dim:
            raise ValueError(
                f"emb_dim={kwargs['emb_dim']} does not match embedding matrix "
                f"dimension {emb_dim}"
            )
        kwargs["emb_dim"] = emb_dim

        super().__init__(**kwargs)

        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(matrix.copy()),
            freeze=bool(freeze),
            padding_idx=1,
        )
        self.embedding_frozen = bool(freeze)

    def embedding_summary(self):
        weight = self.embedding.weight.detach()
        return {
            "vocab_size": int(weight.shape[0]),
            "emb_dim": int(weight.shape[1]),
            "frozen": self.embedding_frozen,
            "nonzero_rows": int((weight.abs().sum(dim=1) > 0).sum().item()),
        }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 0.01, size=(100, 300)).astype(np.float32)
    matrix[1] = 0.0
    model = BiLSTMFastTextForABSA(
        matrix,
        freeze=False,
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_aspects=10,
        num_classes=4,
        dropout=0.2,
    )
    ids = torch.randint(0, 100, (2, 16))
    mask = torch.ones_like(ids)
    out = model(ids, mask)
    print("logits:", tuple(out["logits"].shape))
    print(model.count_parameters())
    print(model.embedding_summary())
