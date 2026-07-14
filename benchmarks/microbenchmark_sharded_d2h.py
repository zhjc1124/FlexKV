#!/usr/bin/env python3
"""Microbenchmark: sharded D2H optimization approaches comparison.

Compares 4 approaches for sharded MLA BLOCKFIRST D2H:
  1. baseline: per-block cudaMemcpyAsync + CPU scatter (current removed)
  2. memcpy2d: per-(layer,segment) cudaMemcpy2DAsync, no CPU scatter
  3. d2d+memcpy2d: D2D transpose shard + per-segment cudaMemcpy2DAsync
  4. rank0_only: D2D transpose full + one-shot contiguous D2H (reference)

Usage:
  python benchmarks/microbenchmark_sharded_d2h.py
  python benchmarks/microbenchmark_sharded_d2h.py --sizes small,large
"""

import argparse
import time
import ctypes
import torch
import torch.cuda as tc

# Load CUDA runtime for cudaMemcpy2DAsync
libcudart = ctypes.CDLL("libcudart.so")

# cudaMemcpyKind
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3

# struct cudaMemcpy3DParms is complex; use the 2D async API directly.
# cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)
libcudart.cudaMemcpy2DAsync.restype = ctypes.c_int
libcudart.cudaMemcpy2DAsync.argtypes = [
    ctypes.c_void_p,   # dst
    ctypes.c_size_t,   # dpitch
    ctypes.c_void_p,   # src
    ctypes.c_size_t,   # spitch
    ctypes.c_size_t,   # width
    ctypes.c_size_t,   # height
    ctypes.c_int,      # kind
    ctypes.c_void_p,   # stream
]


SIZES = {
    "small":  (32,  512,  128, 8),    # num_layers, num_blocks, head_dim, num_gpus
    "medium": (61,  2048, 512, 8),
    "large":  (80,  8192, 512, 8),
}


def benchmark_op(op_fn, warmup=5, iters=20):
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
    return times[len(times) // 2]


def make_few_seg_ids(num_blocks):
    """4-segment block id pattern (same as test_kv_transfer_correctness)."""
    q = num_blocks // 4
    base = torch.arange(num_blocks, dtype=torch.int64)
    ids = torch.cat([base[0:q], base[2*q:3*q], base[q:2*q], base[3*q:4*q]])
    return ids


def run_config(name, num_layers, num_blocks, head_dim, num_gpus):
    ES = 2  # fp16
    shard_size = head_dim * ES // num_gpus  # bytes per shard
    full_chunk = head_dim * ES               # bytes per full chunk
    shard_elems = shard_size // 8            # int64 elements per shard
    full_elems = full_chunk // 8             # int64 elements per full chunk
    block_stride = num_layers * full_chunk   # CPU BLOCKFIRST block stride (bytes)
    layer_stride = full_chunk                 # CPU BLOCKFIRST layer stride (bytes)

    # GPU source: [num_layers, num_blocks, full_elems] (LAYERFIRST, int64)
    gpu_src = torch.randint(0, 1000,
                            (num_layers, num_blocks, full_elems),
                            dtype=torch.int64, device="cuda")

    # CPU target: BLOCKFIRST interleave, pinned
    total_cpu_bytes = num_blocks * num_layers * full_chunk
    cpu_buf = torch.empty(total_cpu_bytes // 8, dtype=torch.int64,
                          pin_memory=True)
    cpu_buf.zero_()

    # Block IDs (few_seg pattern, 4 segments)
    block_ids = make_few_seg_ids(num_blocks)
    # Compute segments (runs of consecutive ids)
    segments = []
    k = 0
    while k < num_blocks:
        start = k
        while k + 1 < num_blocks and block_ids[k + 1].item() == block_ids[k].item() + 1:
            k += 1
        segments.append((block_ids[start].item(), k - start + 1))  # (start_block, run_len)
        k += 1

    # Per-GPU shard offset within full chunk
    shard_offsets = [i * shard_size for i in range(num_gpus)]

    total_data_mb = total_cpu_bytes / 1e6
    print(f"\n  Config: {name} ({num_layers}L / {num_blocks}B / hd={head_dim} / "
          f"{num_gpus} GPUs / {total_data_mb:.1f} MB / {len(segments)} segments)")
    print(f"  shard_size={shard_size}B  full_chunk={full_chunk}B  "
          f"block_stride={block_stride}B  layer_stride={layer_stride}B")
    print(f"  {'Approach':<45s} {'Latency':>10s}   {'Bandwidth':>12s}")
    print(f"  {'-'*72}")

    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    # ---- 1. baseline: per-block cudaMemcpyAsync + CPU scatter ----
    # Simulate 1 GPU (GPU 0) doing its shard. Real system has 8 GPUs in parallel,
    # but they share PCIe, so total time is same as 1 GPU doing 1/8 data... no,
    # each GPU does its own shard, total data = full. But for simplicity, measure
    # 1 GPU's shard (1/8 data), then multiply by 1 (shared PCIe = same total time).
    # Actually, on shared PCIe, 8 parallel D2H = same total time as 1 GPU doing
    # all data. So we simulate 1 GPU doing all shards sequentially.
    def op_baseline():
        # Per GPU i, per layer, per block: cudaMemcpyAsync + CPU scatter
        for gpu_i in range(num_gpus):
            shard_off = shard_offsets[gpu_i]
            for layer in range(num_layers):
                for seg_start, run_len in segments:
                    for b in range(run_len):
                        block_id = seg_start + b  # simplified: contiguous segment
                        src = gpu_src[layer, block_id].view(torch.uint8)[:shard_size]
                        # D2H to staging (simulate with .copy_)
                        # In real code: cudaMemcpyAsync(staging, gpu, shard_size, D2H)
                        # Then CPU scatter: memcpy(cpu, staging, shard_size)
                        # Here we do both in one step for simplicity
                        dst_offset = (block_id * block_stride + layer * layer_stride + shard_off) // 8
                        cpu_buf[dst_offset:dst_offset + shard_elems].copy_(
                            gpu_src[layer, block_id, shard_off // 8:shard_off // 8 + shard_elems],
                            non_blocking=True)
                tc.synchronize()  # per-layer sync (baseline does this)
                # CPU scatter would happen here in real code

    # Simplified baseline: just measure the per-block copy cost
    # (real baseline does cudaMemcpyAsync to staging + CPU memcpy scatter)
    def op_baseline_simplified():
        for gpu_i in range(num_gpus):
            shard_off = shard_offsets[gpu_i]
            for layer in range(num_layers):
                for seg_start, run_len in segments:
                    for b in range(run_len):
                        block_id = seg_start + b
                        dst_offset = (block_id * block_stride + layer * layer_stride + shard_off) // 8
                        # This simulates cudaMemcpyAsync (D2H) directly to CPU
                        cpu_buf[dst_offset:dst_offset + shard_elems].copy_(
                            gpu_src[layer, block_id, shard_off // 8:shard_off // 8 + shard_elems],
                            non_blocking=True)
                tc.synchronize()

    t_baseline = benchmark_op(op_baseline_simplified)
    print(f"  {'1. baseline (per-block copy)':.<45s} {t_baseline:>8.3f} ms   "
          f"{total_data_mb / t_baseline:>8.1f} MB/s")

    # ---- 2. memcpy2d: per-(layer,segment) cudaMemcpy2DAsync ----
    def op_memcpy2d():
        for gpu_i in range(num_gpus):
            shard_off = shard_offsets[gpu_i]
            for layer in range(num_layers):
                for seg_start, run_len in segments:
                    # src: GPU, spitch = full_chunk (stride between blocks)
                    src_ptr = gpu_src.data_ptr() + (
                        layer * num_blocks * full_chunk + seg_start * full_chunk + shard_off)
                    # dst: CPU, dpitch = block_stride (stride between blocks)
                    dst_ptr = cpu_buf.data_ptr() + (
                        seg_start * block_stride + layer * layer_stride + shard_off)
                    # width = shard_size, height = run_len
                    libcudart.cudaMemcpy2DAsync(
                        dst_ptr, block_stride,
                        src_ptr, full_chunk,
                        shard_size, run_len,
                        cudaMemcpyDeviceToHost, stream_ptr)
            tc.synchronize()

    t_memcpy2d = benchmark_op(op_memcpy2d)
    print(f"  {'2. memcpy2d (per-layer-seg cudaMemcpy2DAsync)':.<45s} {t_memcpy2d:>8.3f} ms   "
          f"{total_data_mb / t_memcpy2d:>8.1f} MB/s")

    # ---- 3. D2D transpose + per-segment cudaMemcpy2DAsync ----
    # D2D transpose each GPU's shard: [num_layers, num_blocks, shard_elems] -> [num_blocks, num_layers, shard_elems]
    # Then cudaMemcpy2DAsync with spitch=shard_size (contiguous), dpitch=layer_stride (interleave)
    def op_d2d_memcpy2d():
        for gpu_i in range(num_gpus):
            shard_off = shard_offsets[gpu_i]
            shard_start_elem = shard_off // 8

            # Step 1: D2D transpose
            # Slice shard from full tensor: [num_layers, num_blocks, shard_elems]
            shard_view = gpu_src[:, :, shard_start_elem:shard_start_elem + shard_elems]
            # index_select needed blocks, then transpose + contiguous
            gathered = torch.index_select(shard_view.reshape(num_layers, -1, shard_elems),
                                          1, block_ids.cuda())
            transposed = gathered.transpose(0, 1).contiguous()  # [num_blocks, num_layers, shard_elems]

            # Step 2: per-segment cudaMemcpy2DAsync
            for seg_start_idx, (seg_block, run_len) in enumerate(segments):
                # Find the offset in transposed for this segment
                # transposed is [num_blocks, num_layers, shard_elems] contiguous
                # After index_select, blocks are in block_ids order
                # seg_start_idx is the index within block_ids
                seg_offset = 0
                for si in range(len(segments)):
                    if si == seg_start_idx:
                        break
                    seg_offset += segments[si][1]  # run_len
                # src: transposed + seg_offset * num_layers * shard_size
                src_ptr = transposed.data_ptr() + seg_offset * num_layers * shard_size
                # dst: CPU interleave
                dst_ptr = cpu_buf.data_ptr() + (
                    seg_block * block_stride + shard_off)
                # spitch = shard_size (contiguous in transposed)
                # dpitch = layer_stride (interleave in CPU)
                # width = shard_size, height = num_layers * run_len
                libcudart.cudaMemcpy2DAsync(
                    dst_ptr, layer_stride,
                    src_ptr, shard_size,
                    shard_size, num_layers * run_len,
                    cudaMemcpyDeviceToHost, stream_ptr)
            tc.synchronize()

    t_d2d_memcpy2d = benchmark_op(op_d2d_memcpy2d)
    print(f"  {'3. D2D + memcpy2d (transpose + per-seg 2D)':.<45s} {t_d2d_memcpy2d:>8.3f} ms   "
          f"{total_data_mb / t_d2d_memcpy2d:>8.1f} MB/s")

    # ---- 4. rank0_only: D2D full + one-shot contiguous D2H (reference) ----
    def op_rank0_only():
        # D2D transpose full chunk
        gathered = torch.index_select(gpu_src.reshape(num_layers, -1, full_elems),
                                      1, block_ids.cuda())
        transposed = gathered.transpose(0, 1).contiguous()  # [num_blocks, num_layers, full_elems]
        # One-shot D2H (but strided for BLOCKFIRST... actually contiguous if BLOCKFIRST)
        # In BLOCKFIRST, block_stride = num_layers * full_chunk, and transposed is
        # [num_blocks, num_layers, full_elems] which is exactly BLOCKFIRST layout
        # So one big contiguous D2H works
        cpu_buf[:transposed.numel()].copy_(transposed.view(-1), non_blocking=True)
        tc.synchronize()

    t_rank0 = benchmark_op(op_rank0_only)
    print(f"  {'4. rank0_only (D2D full + one-shot D2H)':.<45s} {t_rank0:>8.3f} ms   "
          f"{total_data_mb / t_rank0:>8.1f} MB/s")

    # ---- Summary ----
    print(f"  {'-'*72}")
    print(f"  {'Summary':<45s}")
    print(f"  baseline → memcpy2d:       {t_baseline / t_memcpy2d:.1f}x speedup")
    print(f"  baseline → D2D+memcpy2d:   {t_baseline / t_d2d_memcpy2d:.1f}x speedup")
    print(f"  baseline → rank0_only:     {t_baseline / t_rank0:.1f}x speedup")
    print(f"  memcpy2d vs D2D+memcpy2d:  {t_memcpy2d / t_d2d_memcpy2d:.2f}x (D2D overhead={t_d2d_memcpy2d - t_memcpy2d:.3f}ms)")

    return {
        "name": name,
        "t_baseline": t_baseline,
        "t_memcpy2d": t_memcpy2d,
        "t_d2d_memcpy2d": t_d2d_memcpy2d,
        "t_rank0": t_rank0,
    }


def main():
    parser = argparse.ArgumentParser(description="Sharded D2H optimization benchmark")
    parser.add_argument("--sizes", default="all",
                        help="Comma-separated size names or 'all'")
    args = parser.parse_args()

    if not tc.is_available():
        print("CUDA not available!")
        return

    print("=" * 76)
    print("  Sharded D2H Optimization Comparison")
    print("  baseline vs memcpy2d vs D2D+memcpy2d vs rank0_only")
    print("=" * 76)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    sizes = list(SIZES.keys()) if args.sizes == "all" else args.sizes.split(",")
    results = []
    for s in sizes:
        cfg = SIZES[s]
        r = run_config(s, *cfg)
        results.append(r)

    # Summary table
    print("\n" + "=" * 76)
    print("  Summary")
    print("=" * 76)
    print(f"  {'Config':<12s} {'baseline':>10s} {'memcpy2d':>10s} {'D2D+2d':>10s} "
          f"{'rank0':>10s} {'base/2d':>8s} {'base/D2D':>9s} {'base/r0':>9s}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['name']:<12s} {r['t_baseline']:>9.3f}ms {r['t_memcpy2d']:>9.3f}ms "
              f"{r['t_d2d_memcpy2d']:>9.3f}ms {r['t_rank0']:>9.3f}ms "
              f"{r['t_baseline'] / r['t_memcpy2d']:>7.1f}x "
              f"{r['t_baseline'] / r['t_d2d_memcpy2d']:>8.1f}x "
              f"{r['t_baseline'] / r['t_rank0']:>8.1f}x")
    print("=" * 76)


if __name__ == "__main__":
    main()
