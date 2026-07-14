#!/usr/bin/env python3
"""
Benchmark: BF_TRANSPOSE D2D+D2H overlap (serial vs ping-pong).

Validates whether overlapping D2D transpose with D2H transfer
using dual streams + double buffering yields the expected speedup.

Approach 1 (serial, current):
  Step 1: D2D transpose ALL layers -> dev_staging (one big buffer)
  Step 2: D2H per-segment memcpy dev_staging -> CPU strided

Approach 2 (ping-pong, proposed):
  Per-layer loop with dual streams:
    d2d_stream: D2D transpose layer N -> buf[idx]
    d2h_stream: D2H transfer layer N-1 from buf[prev] -> CPU
  Overlap: D2D (GPU kernel) and D2H (GPU DMA) run concurrently on 2 streams.
  Memory: 2 x per-layer buffer (much smaller than full dev_staging).
"""

import argparse
import time
import torch
import numpy as np


def bench_serial(gpu_kv, cpu_kv, num_layers, num_blocks, iters=20):
    """Current approach: D2D all layers -> dev_staging, then D2H."""
    # dev_staging: [num_blocks, num_layers, tpb, shard] = BLOCKFIRST layout
    dev_staging = torch.empty(
        num_blocks, num_layers, *gpu_kv.shape[2:],
        dtype=gpu_kv.dtype, device=gpu_kv.device)

    # Warmup
    for _ in range(3):
        for l in range(num_layers):
            dev_staging[:, l].copy_(gpu_kv[l])
        torch.cuda.synchronize()
        cpu_kv.copy_(dev_staging)
        torch.cuda.synchronize()

    # Time
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for l in range(num_layers):
            dev_staging[:, l].copy_(gpu_kv[l])
        torch.cuda.synchronize()
        cpu_kv.copy_(dev_staging)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.median(times), dev_staging.element_size() * dev_staging.nelement()


def bench_pingpong(gpu_kv, cpu_kv, num_layers, num_blocks, iters=20):
    """Proposed: per-layer D2D + D2H overlap with dual streams + double buffer."""
    d2d_stream = torch.cuda.Stream()
    d2h_stream = torch.cuda.Stream()

    # Double buffer: each holds 1 layer of all blocks
    # [num_blocks, tpb, shard] = per-layer BLOCKFIRST slice
    bufA = torch.empty(num_blocks, *gpu_kv.shape[2:],
                       dtype=gpu_kv.dtype, device=gpu_kv.device)
    bufB = torch.empty_like(bufA)
    bufs = [bufA, bufB]

    # Events for ping-pong sync
    d2d_done = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
    d2h_done = [torch.cuda.Event(enable_timing=False) for _ in range(2)]

    def run_once():
        for l in range(num_layers):
            idx = l % 2
            prev_idx = 1 - idx

            # D2D: transpose layer l -> bufs[idx]
            with torch.cuda.stream(d2d_stream):
                bufs[idx].copy_(gpu_kv[l])
                d2d_done[idx].record(d2d_stream)

            # D2H: transfer bufs[prev_idx] -> CPU (if ready)
            if l >= 1:
                d2h_stream.wait_event(d2d_done[prev_idx])
                with torch.cuda.stream(d2h_stream):
                    cpu_kv[:, l - 1].copy_(bufs[prev_idx], non_blocking=True)
                    d2h_done[prev_idx].record(d2h_stream)

            # D2D must wait for D2H to finish reusing buf[idx]
            if l >= 2:
                d2d_stream.wait_event(d2h_done[idx])

        # Drain last layer
        last_idx = (num_layers - 1) % 2
        d2h_stream.wait_event(d2d_done[last_idx])
        with torch.cuda.stream(d2h_stream):
            cpu_kv[:, num_layers - 1].copy_(bufs[last_idx], non_blocking=True)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        run_once()

    # Time
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    total_bytes = gpu_kv.element_size() * gpu_kv.nelement()
    return np.median(times), total_bytes


def verify_correctness(gpu_kv, num_layers, num_blocks):
    """Verify serial and ping-pong produce same CPU result."""
    dev_staging = torch.empty(
        num_blocks, num_layers, *gpu_kv.shape[2:],
        dtype=gpu_kv.dtype, device=gpu_kv.device)

    # Serial result
    cpu_serial = torch.empty(
        num_blocks, num_layers, *gpu_kv.shape[2:],
        dtype=gpu_kv.dtype, pin_memory=True)
    for l in range(num_layers):
        dev_staging[:, l].copy_(gpu_kv[l])
    torch.cuda.synchronize()
    cpu_serial.copy_(dev_staging)
    torch.cuda.synchronize()

    # Ping-pong result
    cpu_pingpong = torch.empty_like(cpu_serial)
    bench_pingpong(gpu_kv, cpu_pingpong, num_layers, num_blocks, iters=1)

    # Compare
    if torch.allclose(cpu_serial, cpu_pingpong, atol=0, rtol=0):
        print("  Correctness: PASS (serial == ping-pong)")
        return True
    else:
        diff = (cpu_serial - cpu_pingpong).abs()
        print(f"  Correctness: FAIL (max diff={diff.max().item()}, "
              f"mismatch={diff.gt(0).sum().item()}/{diff.numel()})")
        return False


def main():
    parser = argparse.ArgumentParser(description="BF_TRANSPOSE D2D+D2H overlap benchmark")
    parser.add_argument("--num-layers", type=int, default=80)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--tpb", type=int, default=64)
    parser.add_argument("--hd", type=int, default=576)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    shard_hd = args.hd // args.num_gpus

    # GPU KV: LAYERFIRST [num_layers, num_blocks, tpb, shard_hd]
    gpu_kv = torch.randn(args.num_layers, args.num_blocks, args.tpb, shard_hd,
                         dtype=torch.float16, device=f"cuda:{args.device}")

    # CPU KV: BLOCKFIRST [num_blocks, num_layers, tpb, shard_hd] (pinned)
    cpu_kv = torch.empty(args.num_blocks, args.num_layers, args.tpb, shard_hd,
                         dtype=torch.float16, pin_memory=True)

    total_mb = gpu_kv.element_size() * gpu_kv.nelement() / 1e6

    print("=" * 80)
    print(f"  BF_TRANSPOSE D2D+D2H Overlap Benchmark")
    print(f"  layers={args.num_layers}, blocks={args.num_blocks}, "
          f"tpb={args.tpb}, hd={args.hd}, shard_hd={shard_hd}")
    print(f"  total D2H = {total_mb:.1f} MB, iters={args.iters}")
    print("=" * 80)
    print()

    # Correctness
    print("--- Correctness ---")
    verify_correctness(gpu_kv, args.num_layers, args.num_blocks)
    print()

    # Serial
    t_serial, _ = bench_serial(gpu_kv, cpu_kv, args.num_layers, args.num_blocks, args.iters)
    print(f"1. Serial (D2D all + D2H all):  {t_serial:8.1f} ms")

    # Ping-pong
    t_pp, _ = bench_pingpong(gpu_kv, cpu_kv, args.num_layers, args.num_blocks, args.iters)
    print(f"2. Ping-pong (D2D+D2H overlap): {t_pp:8.1f} ms")

    speedup = t_serial / t_pp if t_pp > 0 else 0
    saved = t_serial - t_pp
    print()
    print(f"  Speedup: {speedup:.2f}x  (saved {saved:.1f} ms)")
    print()

    # Memory comparison
    dev_staging_bytes = args.num_blocks * args.num_layers * args.tpb * shard_hd * 2  # fp16
    pingpong_buf_bytes = 2 * args.num_blocks * args.tpb * shard_hd * 2  # 2 buffers, 1 layer each
    print(f"  Memory: serial dev_staging = {dev_staging_bytes/1e6:.1f} MB, "
          f"ping-pong bufs = {pingpong_buf_bytes/1e6:.1f} MB "
          f"({dev_staging_bytes/pingpong_buf_bytes:.0f}x smaller)")
    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
