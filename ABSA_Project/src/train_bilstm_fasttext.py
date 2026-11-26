"""
train_bilstm_fasttext.py
Trains and evaluates a BiLSTM multi-head ABSA model whose embedding layer is
initialised from fastText vectors trained on the Train split.

This script is deliberately self-contained and does NOT modify the existing
`train_baseline.py` / `evaluate_absa.py`. It reuses their data pipeline and
metric definitions so the comparison is apples-to-apples:

* same Train/Dev/Test files, labels, class weights and vocabulary;
* same BiLSTM architecture and loss;
* early stopping on Dev Macro F1 (default: max 20 epochs, patience 3);
* the best checkpoint is restored before the single Test evaluation;
* Test metrics include ACD micro/macro, ASC and Aspect-Polarity Pair F1.

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

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import get_linear_schedule_with_warmup

from dataset_loader import ASPECT_LIST, NUM_CLASSES
from baseline_data import (
    build_vocab,
    get_baseline_dataloaders,
    load_vocab,
    save_vocab,
)
from model_bilstm_fasttext import BiLSTMFastTextForABSA
from metrics_utils import compute_full_metrics
from evaluate_absa import (
    build_error_cases,
    build_predictions_frame,
    compute_absa_metrics,
    per_aspect_table,
    write_predictions_jsonl,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, class_weights, device, collect_probs=False):
    """Returns metrics, predictions, labels and (optionally) softmax probs."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids, attention_mask, labels=labels, class_weights=class_weights
            )
            total_loss += outputs["loss"].item()

            logits = outputs["logits"]
            if collect_probs:
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_full_metrics(preds, labels)
    metrics["loss"] = avg_loss

    probs = np.concatenate(all_probs, axis=0) if collect_probs else None
    return metrics, preds, labels, probs


def measure_latency(model, dataloader, device, warmup_batches=5):
    """Inference-only latency after a short warmup."""
    model.eval()
    batches = list(dataloader)
    n_warm = min(warmup_batches, len(batches))
    with torch.no_grad():
        for batch in batches[:n_warm]:
            model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_samples = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            model(input_ids, mask)
            n_samples += input_ids.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return {
        "latency_per_sample_ms": elapsed / n_samples * 1000,
        "throughput_samples_sec": n_samples / elapsed,
        "timed_samples": n_samples,
    }


def _save_predictions(output_dir, experiment_name, df_test, preds, labels, probs):
    prefix = os.path.join(output_dir, f"predictions_{experiment_name}")
    build_predictions_frame(df_test, preds, labels, probs).to_csv(
        f"{prefix}.csv", index=False, encoding="utf-8-sig"
    )
    write_predictions_jsonl(f"{prefix}.jsonl", df_test, preds, labels, probs)
    build_error_cases(df_test, preds, labels, probs).to_csv(
        os.path.join(output_dir, f"errors_{experiment_name}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return f"{prefix}.csv", f"{prefix}.jsonl"


def train(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print("=== BiLSTM + fastText (multi-head ABSA) ===")
    print(f"Device: {device} | Seed: {args.seed}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_csv = os.path.join(args.data_dir, "Train_prepared.csv")
    test_csv = os.path.join(args.data_dir, "Test_prepared.csv")
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    if args.vocab_path and os.path.exists(args.vocab_path):
        vocab = load_vocab(args.vocab_path)
        print(f"Vocabulary loaded: {args.vocab_path} ({len(vocab):,} tokens)")
    else:
        vocab = build_vocab(
            df_train["comment_segmented"].astype(str),
            min_freq=args.vocab_min_freq,
            max_size=args.vocab_size,
        )
        args.vocab_path = os.path.join(
            args.output_dir, f"vocab_bilstm_{args.experiment_name}.json"
        )
        save_vocab(vocab, args.vocab_path)
        print(f"Vocabulary built on Train: {len(vocab):,} tokens -> {args.vocab_path}")

    if not args.embedding_path or not os.path.exists(args.embedding_path):
        raise SystemExit(
            "Missing fastText embedding matrix. Run fasttext_embedding.py first "
            "and pass --embedding_path. Expected: "
            f"{args.embedding_path}"
        )

    matrix = np.load(args.embedding_path)
    if matrix.ndim != 2:
        raise SystemExit(f"Embedding matrix must be 2-D, got {matrix.shape}")
    if matrix.shape[0] != len(vocab):
        raise SystemExit(
            f"Embedding rows ({matrix.shape[0]}) != vocabulary size ({len(vocab)}). "
            "Regenerate the matrix with the same vocabulary settings."
        )
    args.emb_dim = int(matrix.shape[1])
    print(f"Embedding matrix: {matrix.shape} ({matrix.dtype}) -> {args.embedding_path}")

    train_loader, dev_loader, test_loader, class_weights, vocab = (
        get_baseline_dataloaders(
            data_dir=args.data_dir,
            vocab=vocab,
            max_len=args.max_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            vocab_min_freq=args.vocab_min_freq,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )
    )
    print(
        f"Batches Train/Dev/Test: {len(train_loader)}/{len(dev_loader)}/{len(test_loader)}"
    )

    model = BiLSTMFastTextForABSA(
        matrix,
        freeze=args.freeze_embeddings,
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_aspects=len(ASPECT_LIST),
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    ).to(device)
    param_info = model.count_parameters()
    print(f"Parameters: {param_info['total_parameters']:,}")
    print(f"Embedding : {model.embedding_summary()}")

    no_decay = ["bias"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
        eps=1e-8,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dev_f1 = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history = []
    start = time.time()
    checkpoint_path = os.path.join(
        args.output_dir, f"best_bilstm_{args.experiment_name}.pt"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running = 0.0
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
        )
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    input_ids,
                    attention_mask,
                    labels=labels,
                    class_weights=class_weights,
                )
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scale_before <= scaler.get_scale():
                scheduler.step()

            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running / len(train_loader)
        epoch_time = time.time() - epoch_start

        dev_metrics, _, _, _ = evaluate(
            model, dev_loader, class_weights, device, collect_probs=False
        )
        vram = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "dev_loss": dev_metrics["loss"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_micro_f1": dev_metrics["micro_f1"],
                "dev_sentiment_macro_f1": dev_metrics["sentiment_macro_f1"],
                "dev_acd_f1": dev_metrics["acd_f1"],
                "dev_exact_match": dev_metrics["exact_match"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time,
                "peak_vram_mb": vram,
            }
        )
        print(
            f"Epoch {epoch:02d} | Train {avg_train_loss:.4f} | "
            f"Dev loss {dev_metrics['loss']:.4f} | "
            f"Dev MacroF1 {dev_metrics['macro_f1']:.4f} | "
            f"SentF1 {dev_metrics['sentiment_macro_f1']:.4f} | "
            f"EM {dev_metrics['exact_match']:.4f} | "
            f"{epoch_time:.1f}s | {vram:.0f}MB"
        )

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "best_dev_macro_f1": best_dev_f1,
                    "vocab_size": len(vocab),
                    "emb_dim": args.emb_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "embedding_path": args.embedding_path,
                    "freeze_embeddings": args.freeze_embeddings,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  --> new best (Dev Macro F1 {best_dev_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} (patience {args.patience})."
                )
                break

    last_epoch = len(history)
    stopped_early = last_epoch < args.epochs
    total_train_time = time.time() - start
    print(
        f"Training finished in {total_train_time:.1f}s. "
        f"last_epoch={last_epoch} best_epoch={best_epoch} "
        f"(Dev Macro F1 {best_dev_f1:.4f})"
    )

    history_df = pd.DataFrame(history)
    log_path = os.path.join(
        args.log_dir, f"train_log_{args.experiment_name}.csv"
    )
    history_df.to_csv(log_path, index=False)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint; check the data/loop.")

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, class_weights, device, collect_probs=True
    )
    test_full = compute_absa_metrics(test_preds, test_labels)
    test_full["loss"] = test_metrics["loss"]
    latency = measure_latency(model, test_loader, device)

    print("=== Test (BiLSTM + fastText, best checkpoint) ===")
    for key in [
        "micro_f1",
        "macro_f1",
        "sentiment_macro_f1",
        "acd_f1",
        "acd_macro_f1",
        "pair_micro_f1",
        "pair_macro_f1",
        "exact_match",
    ]:
        print(f"  {key}: {test_full[key]:.4f}")

    predictions_csv, predictions_jsonl = _save_predictions(
        args.log_dir,
        args.experiment_name,
        df_test,
        test_preds,
        test_labels,
        test_probs,
    )
    per_aspect_table(test_preds, test_labels).to_csv(
        os.path.join(args.log_dir, f"detailed_metrics_{args.experiment_name}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "experiment_name": args.experiment_name,
        "model": "BiLSTM + fastText",
        "seed": args.seed,
        "epochs_planned": args.epochs,
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "patience": args.patience,
        "best_dev_macro_f1": best_dev_f1,
        "test_loss": test_full["loss"],
        "test_micro_f1": test_full["micro_f1"],
        "test_macro_f1": test_full["macro_f1"],
        "test_macro_f1_4class": test_full["macro_f1"],
        "test_sentiment_macro_f1": test_full["sentiment_macro_f1"],
        "test_acd_f1": test_full["acd_f1"],
        "test_acd_macro_f1": test_full["acd_macro_f1"],
        "test_pair_micro_f1": test_full["pair_micro_f1"],
        "test_pair_macro_f1": test_full["pair_macro_f1"],
        "test_exact_match": test_full["exact_match"],
        "test_n_samples": test_full["n_samples"],
        "test_n_label_cells": test_full["n_label_cells"],
        "per_aspect_test_f1": {
            k: v["f1_macro"] for k, v in test_full["per_aspect"].items()
        },
        "total_parameters": param_info["total_parameters"],
        "vocab_size": len(vocab),
        "embedding_dim": args.emb_dim,
        "freeze_embeddings": args.freeze_embeddings,
        "embedding_path": args.embedding_path,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "total_train_time_sec": total_train_time,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else 0.0
        ),
        "latency_per_sample_ms": latency["latency_per_sample_ms"],
        "throughput_samples_sec": latency["throughput_samples_sec"],
        "predictions_csv": predictions_csv,
        "predictions_jsonl": predictions_jsonl,
        "test_metrics_full": test_full,
    }

    json_path = os.path.join(
        args.log_dir, f"test_results_{args.experiment_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print("Saved:", json_path)
    print("Saved:", log_path)
    print("Saved:", checkpoint_path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="BiLSTM multi-head ABSA with fastText-initialised embeddings"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ABSA_Project/models/experiments_2026/bilstm_ft",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="ABSA_Project/results/experiments_2026/bilstm_ft",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="bilstm_ft_seed42"
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="ABSA_Project/results/experiments_2026/fasttext/fasttext_embeddings.npy",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="ABSA_Project/results/experiments_2026/fasttext/fasttext_vocab.json",
    )
    parser.add_argument(
        "--freeze_embeddings",
        action="store_true",
        help="Freeze the fastText embedding (default: fine-tune it)",
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_min_freq", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=64000)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
