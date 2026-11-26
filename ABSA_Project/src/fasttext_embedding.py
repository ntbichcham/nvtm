"""
fasttext_embedding.py
Builds a fastText embedding matrix for the BiLSTM baseline.

Protocol
--------
* fastText is trained with the unsupervised skip-gram objective on the
  **Train split only** (no Dev/Test leakage).
* The vocabulary is exactly the same word-level vocabulary used by
  `baseline_data.py` (whitespace tokens of `comment_segmented`, lower-cased,
  min_freq / max_size identical to the BiLSTM baseline).
* The output matrix is aligned with that vocabulary, so it can be passed
  directly to `model_bilstm_fasttext.BiLSTMFastTextForABSA`.

Outputs (in --out_dir)
----------------------
fasttext_embeddings.npy : float32 (vocab_size, dim)
fasttext_vocab.json     : token -> row index
fasttext_meta.json      : training hyper-parameters + coverage statistics

Author: Nhom 2 - Thanh vien 3
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from baseline_data import build_vocab, tokenize_text  # noqa: E402


def _write_corpus(texts, path):
    """Writes one whitespace-tokenised comment per line (fastText input)."""
    with open(path, "w", encoding="utf-8") as handle:
        for text in texts:
            tokens = tokenize_text(text)
            if tokens:
                handle.write(" ".join(tokens) + "\n")


def train_fasttext(
    train_csv,
    out_dir,
    dim=300,
    epochs=20,
    min_count=2,
    window=5,
    min_n=3,
    max_n=6,
    seed=42,
    vocab_min_freq=2,
    vocab_size=64000,
    save_model=False,
):
    """Trains fastText on Train and exports a vocabulary-aligned matrix."""
    try:
        import fasttext
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Missing dependency: fasttext. Install with `pip install fasttext-wheel`."
        ) from exc

    os.makedirs(out_dir, exist_ok=True)

    df_train = pd.read_csv(train_csv)
    if "comment_segmented" not in df_train.columns:
        raise ValueError(f"{train_csv} has no 'comment_segmented' column")

    vocab = build_vocab(
        df_train["comment_segmented"].astype(str),
        min_freq=vocab_min_freq,
        max_size=vocab_size,
    )
    print(f"Train comments : {len(df_train):,}")
    print(f"Vocabulary     : {len(vocab):,} tokens (fitted on Train only)")

    corpus_path = os.path.join(out_dir, "_fasttext_corpus.txt")
    _write_corpus(df_train["comment_segmented"].astype(str).tolist(), corpus_path)
    n_lines = sum(1 for _ in open(corpus_path, "r", encoding="utf-8"))
    print(f"Corpus         : {n_lines:,} lines -> {corpus_path}")

    print(
        "Training fastText skip-gram "
        f"(dim={dim}, epochs={epochs}, ws={window}, minCount={min_count}, "
        f"minn={min_n}, maxn={max_n}, thread=1)"
    )
    # NOTE: the Python binding of fastText does not expose the C++ `-seed`
    # option, so the embedding is trained ONCE and saved as an artifact
    # (`fasttext_embeddings.npy`). All BiLSTM seeds reuse the same matrix,
    # which is exactly what a controlled comparison requires.
    model = fasttext.train_unsupervised(
        corpus_path,
        model="skipgram",
        dim=dim,
        epoch=epochs,
        lr=0.05,
        ws=window,
        minCount=min_count,
        minn=min_n,
        maxn=max_n,
        neg=5,
        wordNgrams=1,
        loss="ns",
        bucket=2000000,
        thread=1,
        verbose=2,
    )

    # Row 0: <unk> = mean of all in-vocabulary vectors.
    model_words = set(model.get_words())
    in_vocab_vectors = []
    for token in vocab:
        if token in ("<pad>", "<unk>"):
            continue
        if token in model_words:
            in_vocab_vectors.append(model.get_word_vector(token))
    if in_vocab_vectors:
        unk_vector = np.mean(np.stack(in_vocab_vectors, axis=0), axis=0)
    else:  # extremely small corpus fallback
        unk_vector = np.zeros(dim, dtype=np.float32)

    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    in_vocab = subword_only = fallback = 0
    for token, index in vocab.items():
        if token == "<pad>":
            matrix[index] = 0.0
            continue
        if token == "<unk>":
            matrix[index] = unk_vector
            continue
        vector = np.asarray(model.get_word_vector(token), dtype=np.float32)
        if vector.shape != (dim,):
            raise RuntimeError(f"Unexpected vector shape for {token!r}: {vector.shape}")
        if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) == 0.0:
            matrix[index] = unk_vector
            fallback += 1
        else:
            matrix[index] = vector
            if token in model_words:
                in_vocab += 1
            else:
                subword_only += 1

    matrix[vocab["<pad>"]] = 0.0
    total = max(1, len(vocab) - 2)  # exclude <unk>, <pad>

    emb_path = os.path.join(out_dir, "fasttext_embeddings.npy")
    vocab_path = os.path.join(out_dir, "fasttext_vocab.json")
    meta_path = os.path.join(out_dir, "fasttext_meta.json")
    np.save(emb_path, matrix)
    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False)

    meta = {
        "train_csv": os.path.abspath(train_csv),
        "dim": dim,
        "epochs": epochs,
        "min_count": min_count,
        "window": window,
        "min_n": min_n,
        "max_n": max_n,
        "seed": seed,
        "vocab_min_freq": vocab_min_freq,
        "vocab_size": len(vocab),
        "fasttext_model_words": len(model_words),
        "coverage": {
            "in_vocab": in_vocab,
            "subword_only": subword_only,
            "fallback_unk": fallback,
            "total": total,
            "in_vocab_ratio": round(in_vocab / total, 6),
            "subword_only_ratio": round(subword_only / total, 6),
            "fallback_ratio": round(fallback / total, 6),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    if save_model:
        model_path = os.path.join(out_dir, "fasttext.model")
        model.save_model(model_path)
        print(f"fastText model : {model_path}")

    try:
        os.remove(corpus_path)
    except OSError:
        pass

    print("\n=== fastText embedding summary ===")
    print(f"  matrix        : {matrix.shape} ({matrix.dtype})")
    print(f"  in-vocab      : {in_vocab:,} / {total:,} ({in_vocab / total:.2%})")
    print(f"  subword-only  : {subword_only:,} / {total:,} ({subword_only / total:.2%})")
    print(f"  fallback <unk>: {fallback:,} / {total:,} ({fallback / total:.2%})")
    print(f"  saved         : {emb_path}")
    print(f"  saved         : {vocab_path}")
    print(f"  saved         : {meta_path}")
    return {
        "embeddings": emb_path,
        "vocab": vocab_path,
        "meta": meta_path,
        "shape": list(matrix.shape),
        "coverage": meta["coverage"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train fastText embeddings on Train and export a BiLSTM-ready matrix"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="ABSA_Project/data/processed/UIT-ViSFD/prepared_for_model/Train_prepared.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ABSA_Project/results/experiments_2026/fasttext",
    )
    parser.add_argument("--dim", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_n", type=int, default=3)
    parser.add_argument("--max_n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_min_freq", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=64000)
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Also save the native fastText .model file (larger, optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fasttext(
        train_csv=args.train_csv,
        out_dir=args.out_dir,
        dim=args.dim,
        epochs=args.epochs,
        min_count=args.min_count,
        window=args.window,
        min_n=args.min_n,
        max_n=args.max_n,
        seed=args.seed,
        vocab_min_freq=args.vocab_min_freq,
        vocab_size=args.vocab_size,
        save_model=args.save_model,
    )
