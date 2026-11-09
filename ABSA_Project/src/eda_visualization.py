"""
eda_visualization.py
Exploratory data analysis figures for UIT-ViSFD (Train split by default).

The script only reads existing artifacts:
  * data/original/UIT-ViSFD/{split}.csv
  * results/statistics/aspect_distribution.csv
  * results/statistics/sentiment_distribution.csv
  * results/statistics/aspect_cooccurrence_{count,jaccard}.csv
  * results/statistics/monthly_comment_distribution.csv
  * results/statistics/phobert_token_details.csv (or prepared_for_model)

All figures are written to `ABSA_Project/figures/reproduced` by default so the
figures used in the report are never overwritten.

Author: Nhom 2 - Thanh vien 1
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

ASPECT_ORDER = [
    "BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL",
    "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE",
]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {name}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_aspect_distribution(distribution, split, out_dir):
    frame = distribution[distribution["Dataset"] == split].copy()
    frame = frame.sort_values("Comment_Count", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(frame["Aspect"], frame["Comment_Count"], color="#1f77b4")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(frame["Comment_Count"]) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}", va="center", fontsize=11)
    ax.set_xlabel("Số lượng bình luận")
    ax.set_ylabel("Khía cạnh")
    ax.set_title(f"Phân bố số lượng bình luận theo khía cạnh - {split}")
    ax.set_xlim(0, max(frame["Comment_Count"]) * 1.08)
    save(fig, out_dir, "aspect_distribution_train.png")


def _sentiment_pivot(sentiment, split):
    frame = sentiment[sentiment["Dataset"] == split]
    pivot = frame.pivot(index="Aspect", columns="Sentiment", values="Count").fillna(0)
    totals = pivot.sum(axis=1)
    order = totals.sort_values(ascending=False).index.tolist()
    return pivot.loc[order], totals.loc[order]


def plot_sentiment_stacked(sentiment, split, out_dir):
    pivot, _ = _sentiment_pivot(sentiment, split)
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(pivot))
    for sentiment_name, color in zip(SENTIMENTS, SENTIMENT_COLORS):
        values = pivot[sentiment_name].values if sentiment_name in pivot else np.zeros(len(pivot))
        ax.bar(pivot.index, values, bottom=bottom, label=sentiment_name, color=color)
        bottom = bottom + values
    ax.set_xlabel("Khía cạnh")
    ax.set_ylabel("Số lượng nhãn")
    ax.set_title(f"Phân bố cảm xúc theo từng khía cạnh - {split}")
    ax.legend(title="Cực tính cảm xúc")
    ax.tick_params(axis="x", rotation=30)
    save(fig, out_dir, "sentiment_distribution_train.png")


def plot_sentiment_percentage(sentiment, split, out_dir):
    pivot, _ = _sentiment_pivot(sentiment, split)
    shares = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(shares))
    for sentiment_name, color in zip(SENTIMENTS, SENTIMENT_COLORS):
        values = shares[sentiment_name].values if sentiment_name in shares else np.zeros(len(shares))
        ax.bar(shares.index, values, bottom=bottom, label=sentiment_name, color=color)
        bottom = bottom + values
    ax.set_xlabel("Khía cạnh")
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Tỷ lệ cảm xúc theo từng khía cạnh - {split}")
    ax.legend(title="Cực tính cảm xúc")
    ax.tick_params(axis="x", rotation=30)
    save(fig, out_dir, "sentiment_percentage_train.png")


def _histogram(values, xlabel, title, name, out_dir, bins, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(values, bins=bins, kde=True, stat="density", color=color,
                 edgecolor="black", linewidth=0.6, alpha=0.5, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mật độ")
    ax.set_title(title)
    save(fig, out_dir, name)


def plot_text_lengths(original, split, out_dir):
    comments = original["comment"].astype(str)
    _histogram(comments.str.len().values,
               "Số ký tự",
               f"Phân bố số ký tự trong bình luận - {split}",
               "character_length_distribution.png", out_dir, bins=50)
    _histogram(comments.str.split().str.len().values,
               "Số đơn vị tách bằng khoảng trắng",
               f"Phân bố số đơn vị tách bằng khoảng trắng - {split}",
               "whitespace_length_distribution.png", out_dir, bins=40)


def plot_token_lengths(token_details, split, out_dir):
    values = token_details[token_details["Dataset"] == split]["Token_Length"].astype(int).values
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(values, bins=50, kde=True, stat="density", color="#1f77b4",
                 edgecolor="black", linewidth=0.6, alpha=0.5, ax=ax)
    ax.axvline(128, color="#1f77b4", linestyle="--", linewidth=2,
               label="Ngưỡng 128 token")
    ax.axvline(256, color="#1f77b4", linestyle="--", linewidth=2,
               label="Ngưỡng 256 token")
    ax.set_xlabel("Số token")
    ax.set_ylabel("Mật độ")
    ax.set_title(f"Phân bố độ dài token PhoBERT - {split}")
    ax.legend()
    save(fig, out_dir, "phobert_token_length_distribution.png")


def plot_cooccurrence(count_matrix, jaccard_matrix, split, out_dir):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(count_matrix, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax)
    ax.set_title(f"Aspect Co-occurrence Heatmap - {split}")
    ax.set_xlabel("Khía cạnh")
    ax.set_ylabel("Khía cạnh")
    save(fig, out_dir, "aspect_cooccurrence_count.png")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(jaccard_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title(f"Aspect Co-occurrence Jaccard Similarity - {split}")
    ax.set_xlabel("Khía cạnh")
    ax.set_ylabel("Khía cạnh")
    save(fig, out_dir, "aspect_cooccurrence_jaccard.png")


def plot_monthly(monthly, out_dir):
    frame = monthly.sort_values("Year_Month")
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(frame["Year_Month"], frame["Comment_Count"], marker="o",
            linewidth=2, color="#1f77b4")
    ax.set_xlabel("Thời gian (Năm-Tháng)")
    ax.set_ylabel("Số lượng bình luận")
    ax.set_title("Số lượng bình luận theo tháng - UIT-ViSFD")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.35)
    save(fig, out_dir, "monthly_comment_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="UIT-ViSFD EDA figures")
    parser.add_argument("--config", default=os.path.join(
        ROOT, "ABSA_Project", "configs", "preprocessing.yaml"))
    parser.add_argument("--split", default="Train")
    parser.add_argument("--stats-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(resolve(args.config))
    stats_dir = resolve(args.stats_dir or config["data"]["stats_dir"])
    data_dir = resolve(args.data_dir or config["data"]["original_dir"])
    out_dir = resolve(args.out_dir or "ABSA_Project/figures/reproduced")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading statistics from: {stats_dir}")
    distribution = pd.read_csv(os.path.join(stats_dir, "aspect_distribution.csv"),
                               encoding="utf-8-sig")
    sentiment = pd.read_csv(os.path.join(stats_dir, "sentiment_distribution.csv"),
                            encoding="utf-8-sig")
    monthly = pd.read_csv(os.path.join(stats_dir, "monthly_comment_distribution.csv"),
                          encoding="utf-8-sig")
    count_matrix = pd.read_csv(os.path.join(stats_dir, "aspect_cooccurrence_count.csv"),
                               encoding="utf-8-sig", index_col=0)
    jaccard_matrix = pd.read_csv(os.path.join(stats_dir, "aspect_cooccurrence_jaccard.csv"),
                                 encoding="utf-8-sig", index_col=0)

    token_path = os.path.join(stats_dir, "phobert_token_details.csv")
    if os.path.exists(token_path):
        token_details = pd.read_csv(token_path, encoding="utf-8-sig")
    else:
        prepared_path = os.path.join(
            resolve(config["data"]["prepared_dir"]), f"{args.split}_prepared.csv")
        prepared = pd.read_csv(prepared_path)
        token_details = prepared[["index", "token_length"]].rename(
            columns={"index": "Index", "token_length": "Token_Length"})
        token_details["Dataset"] = args.split

    original = pd.read_csv(os.path.join(data_dir, f"{args.split}.csv"),
                           encoding="utf-8-sig")

    plot_aspect_distribution(distribution, args.split, out_dir)
    plot_sentiment_stacked(sentiment, args.split, out_dir)
    plot_sentiment_percentage(sentiment, args.split, out_dir)
    plot_text_lengths(original, args.split, out_dir)
    plot_token_lengths(token_details, args.split, out_dir)
    plot_cooccurrence(count_matrix, jaccard_matrix, args.split, out_dir)
    plot_monthly(monthly, out_dir)

    print(f"\nFigures written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
