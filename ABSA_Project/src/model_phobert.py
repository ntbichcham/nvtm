"""
model_phobert.py
Multi-Head PhoBERT architecture for Vietnamese Aspect-Based Sentiment Analysis.
Author: Nhom 2 - Thanh vien 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class MultiHeadPhoBERTForABSA(nn.Module):
    """
    Multi-Head Architecture based on PhoBERT (RoBERTa backbone) for 10-Aspect ABSA.
    Each aspect has a dedicated classification head with 4 classes:
        0: None (aspect not mentioned)
        1: Negative
        2: Neutral
        3: Positive
    """
    def __init__(
        self,
        pretrained_model_name="vinai/phobert-base",
        num_aspects=10,
        num_classes=4,
        dropout=0.1,
        dense_head=False
    ):
        super(MultiHeadPhoBERTForABSA, self).__init__()
        self.num_aspects = num_aspects
        self.num_classes = num_classes
        self.pretrained_model_name = pretrained_model_name

        # Load PhoBERT backbone
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.phobert = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        hidden_size = self.config.hidden_size  # 768 for PhoBERT-base

        self.dropout = nn.Dropout(dropout)
        self.dense_head = dense_head

        # 10 Classification Heads
        if dense_head:
            # RoBERTa classification head style: Linear -> Tanh -> Dropout -> Linear
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_classes)
                ) for _ in range(num_aspects)
            ])
        else:
            # Standard single linear projection head
            self.heads = nn.ModuleList([
                nn.Linear(hidden_size, num_classes) for _ in range(num_aspects)
            ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the classification heads."""
        for head in self.heads:
            if isinstance(head, nn.Sequential):
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(head, nn.Linear):
                nn.init.xavier_uniform_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        """
        Forward pass.
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, num_aspects) or None
            class_weights: list of 10 tensors with class weights for each head
        Returns:
            dict containing:
                'logits': (batch_size, num_aspects, num_classes)
                'loss': scalar loss (if labels provided)
        """
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract representation of <s> token (index 0)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)

        # Compute logits for each of the 10 heads
        head_logits = []
        for head in self.heads:
            head_logits.append(head(cls_rep))  # each is (batch_size, num_classes)

        # Stack into (batch_size, num_aspects, num_classes)
        logits = torch.stack(head_logits, dim=1)

        loss = None
        if labels is not None:
            total_loss = 0.0
            for k in range(self.num_aspects):
                target = labels[:, k]
                weight = class_weights[k].to(input_ids.device) if class_weights is not None else None
                head_loss = F.cross_entropy(logits[:, k, :], target, weight=weight)
                total_loss += head_loss
            loss = total_loss

        return {
            'logits': logits,
            'loss': loss
        }

    def predict(self, input_ids, attention_mask):
        """
        Predict discrete classes and probabilities.
        Returns:
            preds: (batch_size, num_aspects) with values in {0, 1, 2, 3}
            probs: (batch_size, num_aspects, num_classes)
        """
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def count_parameters(self):
        """Returns total parameters and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.phobert.parameters())
        heads_params = sum(p.numel() for p in self.heads.parameters())
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'heads_parameters': heads_params
        }


if __name__ == '__main__':
    # Unit test / verification
    print("Initializing MultiHeadPhoBERTForABSA...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = MultiHeadPhoBERTForABSA(dropout=0.1, dense_head=False).to(device)
    param_info = model.count_parameters()
    print("Parameter Breakdown:")
    for k, v in param_info.items():
        print(f"  {k}: {v:,}")

    # Dummy batch forward pass
    batch_size = 2
    seq_len = 32
    dummy_input_ids = torch.randint(1, 1000, (batch_size, seq_len)).to(device)
    dummy_mask = torch.ones((batch_size, seq_len)).to(device)
    dummy_labels = torch.randint(0, 4, (batch_size, 10)).to(device)

    out = model(dummy_input_ids, dummy_mask, labels=dummy_labels)
    print("Logits shape:", out['logits'].shape) # Expected: (2, 10, 4)
    print("Loss scalar:", out['loss'].item())

    # Backward pass check
    out['loss'].backward()
    print("Backward pass executed successfully! Gradients computed properly.")
