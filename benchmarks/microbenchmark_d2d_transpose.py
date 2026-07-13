#!/usr/bin/env python3
"""Microbenchmark: D2D transpose (LAYERFIRST -> BLOCKFIRST) cost.

Measures the GPU-side cost of the BLOCKFIRST_BATCHED proposal:
  1. index_select  — gather needed blocks from the full GPU tensor
  2. transpose+contiguous — LAYERFIRST -> BLOCKFIRST layout change
  3. One-shot D2H  — single large cudaMemcpyAsync (the payoff)

Compares against the current per-layer D2H approach to quantify
the net benefit of replacing N small D2H calls with 1 D2D + 1 large D2H.

Usage:
  python benchmarks/microbenchmark_d2d_transpose.py
  python benchmarks/microbenchmark_d2d_transpose.py --sizes small,large
"""

import argparse
import time
import torch
import torch.cuda as tc


# ---------------------------------------------------------------------------
# Configs: (name, num_layers, num_blocks, head_dim, elems_per_block_int64)
# elems_per_block = heads_per_rank * head_dim * ES / sizeof(int64)
# For MLA (heads=1, hd=512, fp16): elems = 1 * 512 * 2 / 8 = 128
# For MHA (heads=1, hd=128, fp16): elems = 1 * 128 * 2 / 8 = 32
# ---------------------------------------------------------------------------
SIZES = {
    "small":  (32,  512,  128, 32),    # 32L, 512B, hd=128, 4MB
    "medium": (61,  2048, 512, 128),   # 61L, 2048B, hd=512, 122MB
    "large":  (80,  8192, 512, 128),   # 80L, 8192B, hd=512, 640MB
}


def benchmark_op(op_fn, warmup=5, iters=20):
    """Benchmark a GPU operation, return median latency in ms."""
    # Warmup
    for _ in range(warmup):
        op_fn()
    tc.synchronize()

    times = []
    for _ in range(iters):
        tc.synchronize()
        t0 = time.perf_counter()
        op_fn()
        tc.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def run_config(name, num_layers, num_blocks, head_dim, elems_per_block):
    """Run D2D transpose benchmark for one config."""
    ES = 2  # fp16
    chunk_bytes = elems_per_block * 8  # int64 elements * 8 bytes
    total_blocks = num_blocks  # assume we transfer all blocks
    total_elems = num_layers * total_blocks * elems_per_block

    # GPU source tensor (LAYERFIRST): [num_layers, total_blocks, elems_per_block]
    # Allocate directly as int64 to avoid float16->int64 size mismatch.
    gpu_src_i64 = torch.randint(0, 1000, (num_layers, total_blocks, elems_per_block),
                                 dtype=torch.int64, device="cuda")

    # Block IDs: use few_seg pattern (4 segments, realistic for prefill)
    q = total_blocks // 4
    base = torch.arange(total_blocks, dtype=torch.int64)
    block_ids = torch.cat([base[0:q], base[2*q:3*q], base[q:2*q], base[3*q:4*q]]).cuda()

    # Host staging buffer (pinned)
    staging_bytes = total_blocks * num_layers * chunk_bytes
    host_buf = torch.empty(staging_bytes // 2, dtype=torch.float16, pin_memory=True)
    host_buf_i64 = host_buf.view(torch.int64)

    # ---- Op 1: index_select only ----
    def op_index_select():
        # [num_layers, total_blocks, elems] -> [num_layers, num_blocks, elems]
        return torch.index_select(gpu_src_i64, 1, block_ids)

    # ---- Op 2: transpose + contiguous (LAYERFIRST -> BLOCKFIRST) ----
    gathered = torch.index_select(gpu_src_i64, 1, block_ids)  # pre-compute
    def op_transpose_contiguous():
        return gathered.transpose(0, 1).contiguous()

    # ---- Op 3: full pipeline (index_select + transpose + contiguous) ----
    def op_full_d2d():
        g = torch.index_select(gpu_src_i64, 1, block_ids)
        return g.transpose(0, 1).contiguous()

    # ---- Op 4: one-shot D2H (from BLOCKFIRST staging) ----
    bf_buf = gathered.transpose(0, 1).contiguous()  # [num_blocks, num_layers, elems]
    bf_bytes = bf_buf.numel() * 8  # int64
    def op_oneshot_d2h():
        host_buf_i64[:bf_buf.numel()].copy_(bf_buf.view(-1), non_blocking=True)
        tc.synchronize()

    # ---- Op 5: per-layer D2H (current approach, for comparison) ----
    def op_per_layer_d2h():
        for layer in range(num_layers):
            src = gpu_src_i64[layer]  # [total_blocks, elems]
            gathered_layer = torch.index_select(src, 0, block_ids)  # [num_blocks, elems]
            offset = layer * num_blocks * elems_per_block
            host_buf_i64[offset:offset + gathered_layer.numel()].copy_(
                gathered_layer.view(-1), non_blocking=True)
        tc.synchronize()

    # ---- Run benchmarks ----
    t_is = benchmark_op(op_index_select)
    t_tc = benchmark_op(op_transpose_contiguous)
    t_full = benchmark_op(op_full_d2d)
    t_oneshot = benchmark_op(op_oneshot_d2h)
    t_per_layer = benchmark_op(op_per_layer_d2h)

    # Data sizes
    d2h_bytes = total_blocks * num_layers * chunk_bytes
    d2d_bytes = total_blocks * num_layers * chunk_bytes  # same data, just rearranged

    print(f"\n  Config: {name} ({num_layers}L / {total_blocks}B / hd={head_dim} / "
          f"{d2h_bytes / 1e6:.1f} MB)")
    print(f"  {'Op':<35s} {'Latency':>10s}   {'Bandwidth':>12s}")
    print(f"  {'-'*60}")
    print(f"  {'index_select':.<35s} {t_is:>8.3f} ms   {d2d_bytes/t_is/1e6:>8.1f} GB/s")
    print(f"  {'transpose + contiguous':.<35s} {t_tc:>8.3f} ms   {d2d_bytes/t_tc/1e6:>8.1f} GB/s")
    print(f"  {'full D2D (is + tc)':.<35s} {t_full:>8.3f} ms   {d2d_bytes/t_full/1e6:>8.1f} GB/s")
    print(f"  {'one-shot D2H (BLOCKFIRST)':.<35s} {t_oneshot:>8.3f} ms   {d2h_bytes/t_oneshot/1e6:>8.1f} GB/s")
    print(f"  {'per-layer D2H (current)':.<35s} {t_per_layer:>8.3f} ms   {d2h_bytes/t_per_layer/1e6:>8.1f} GB/s")
    print(f"  {'-'*60}")

    # Summary
    new_total = t_full + t_oneshot
    speedup = t_per_layer / new_total if new_total > 0 else float('inf')
    print(f"  New approach:  D2D({t_full:.3f}) + D2H({t_oneshot:.3f}) = {new_total:.3f} ms")
    print(f"  Old approach:  per-layer D2H = {t_per_layer:.3f} ms")
    print(f"  Speedup:       {speedup:.2f}x")
    print(f"  D2D overhead:  {t_full:.3f} ms ({t_full/new_total*100:.1f}% of new total)")

    return {
        "name": name,
        "t_index_select": t_is,
        "t_transpose_contiguous": t_tc,
        "t_full_d2d": t_full,
        "t_oneshot_d2h": t_oneshot,
        "t_per_layer_d2h": t_per_layer,
        "new_total": new_total,
        "speedup": speedup,
        "d2d_pct": t_full / new_total * 100 if new_total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="D2D transpose microbenchmark")
    parser.add_argument("--sizes", default="all",
                        help="Comma-separated size names (small,medium,large) or 'all'")
    args = parser.parse_args()

    if not tc.is_available():
        print("CUDA not available!")
        return

    print("=" * 70)
    print("  D2D Transpose Microbenchmark: LAYERFIRST -> BLOCKFIRST")
    print("  Measures index_select + transpose + contiguous cost vs per-layer D2H")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    sizes = list(SIZES.keys()) if args.sizes == "all" else args.sizes.split(",")
    results = []
    for s in sizes:
        cfg = SIZES[s]
        r = run_config(s, *cfg)
        results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Config':<12s} {'D2D':>8s} {'1-shot D2H':>12s} {'New Total':>12s} "
          f"{'Per-layer':>12s} {'Speedup':>10s} {'D2D %':>8s}")
    print(f"  {'-'*76}")
    for r in results:
        print(f"  {r['name']:<12s} {r['t_full_d2d']:>7.3f}ms {r['t_oneshot_d2h']:>10.3f}ms "
              f"{r['new_total']:>10.3f}ms {r['t_per_layer_d2h']:>10.3f}ms "
              f"{r['speedup']:>9.2f}x {r['d2d_pct']:>6.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
