"""
benchmark_complexity.py
Benchmark execution latency, throughput, peak memory, and parameter counts
for PhoBERT vs BiLSTM baseline on local hardware.
Author: Nhom 2 - Thanh vien 3
"""

import time
import torch
import pandas as pd
from model_phobert import MultiHeadPhoBERTForABSA
from model_baseline import BiLSTMMultiHeadForABSA


def benchmark_model(model, device, batch_size=16, seq_len=160, num_runs=50, warmup=10):
    model.eval()
    model.to(device)

    dummy_input = torch.randint(1, 1000, (batch_size, seq_len), device=device)
    dummy_mask = torch.ones((batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input, dummy_mask)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input, dummy_mask)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    total_samples = batch_size * num_runs

    latency_per_sample_ms = (total_time / total_samples) * 1000
    throughput = total_samples / total_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0.0

    params = sum(p.numel() for p in model.parameters())

    return {
        'total_parameters': params,
        'latency_per_sample_ms': latency_per_sample_ms,
        'throughput_samples_per_sec': throughput,
        'peak_vram_mb': peak_vram_mb
    }


def run_benchmark(vocab_size=64000, seq_len=160, batch_size=16, num_runs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Benchmarking Models on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")
    print(f"BiLSTM vocabulary size: {vocab_size:,} | seq_len={seq_len} | batch={batch_size}")

    print("Benchmarking PhoBERT-base ABSA...")
    phobert = MultiHeadPhoBERTForABSA(dropout=0.1)
    res_phobert = benchmark_model(phobert, device, batch_size=batch_size,
                                  seq_len=seq_len, num_runs=num_runs)
    del phobert
    torch.cuda.empty_cache()

    print("Benchmarking BiLSTM Baseline ABSA...")
    bilstm = BiLSTMMultiHeadForABSA(vocab_size=vocab_size)
    res_bilstm = benchmark_model(bilstm, device, batch_size=batch_size,
                                 seq_len=seq_len, num_runs=num_runs)
    del bilstm
    torch.cuda.empty_cache()

    df = pd.DataFrame([
        {
            'Model': 'MultiHeadPhoBERT (Proposed)',
            'Parameters': f"{res_phobert['total_parameters']:,}",
            'Latency (ms/sample)': f"{res_phobert['latency_per_sample_ms']:.2f}",
            'Throughput (samples/s)': f"{res_phobert['throughput_samples_per_sec']:.1f}",
            'Peak VRAM (MB)': f"{res_phobert['peak_vram_mb']:.1f}",
            'Theoretical Time': 'O(N^2 * d + N * d^2)',
            'Parallelism': 'O(1) - Tensor Cores'
        },
        {
            'Model': 'BiLSTM Baseline (Reference)',
            'Parameters': f"{res_bilstm['total_parameters']:,}",
            'Latency (ms/sample)': f"{res_bilstm['latency_per_sample_ms']:.2f}",
            'Throughput (samples/s)': f"{res_bilstm['throughput_samples_per_sec']:.1f}",
            'Peak VRAM (MB)': f"{res_bilstm['peak_vram_mb']:.1f}",
            'Theoretical Time': 'O(N * d^2)',
            'Parallelism': 'O(N) - Sequential'
        }
    ])

    out_csv = 'ABSA_Project/results/evaluation/model_complexity_benchmark.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nComplexity benchmark saved to {out_csv}:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=64000,
                        help='BiLSTM embedding vocabulary size (use the trained vocab, e.g. 5079)')
    parser.add_argument('--seq_len', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_runs', type=int, default=50)
    args = parser.parse_args()
    run_benchmark(vocab_size=args.vocab_size, seq_len=args.seq_len,
                  batch_size=args.batch_size, num_runs=args.num_runs)
