#!/usr/bin/env python3
"""Comprehensive D2H benchmark: all layouts × modes × approaches.

Evaluates every combination discussed:
  Layouts: LAYERFIRST, BLOCKFIRST
  Modes: MHA, MLA rank0_only, MLA sharded, MLA all_write
  Approaches:
    A. baseline: per-block cudaMemcpyAsync + CPU scatter (current STAGED_PER_BLOCK)
    B. memcpy2d: per-(layer,segment) cudaMemcpy2DAsync, no scatter
    C. d2d_3path: D2D transpose + 3-path (contiguous memcpy / per-seg memcpy / per-block memcpy)
    D. d2d_memcpy2d: D2D transpose + cudaMemcpy2DAsync (for sharded strided)
    E. d2d_oneshot: D2D transpose + one contiguous D2H (rank0_only reference)

Usage:
  python benchmarks/microbenchmark_d2h_comprehensive.py
  python benchmarks/microbenchmark_d2h_comprehensive.py --sizes small,large
  python benchmarks/microbenchmark_d2h_comprehensive.py --quick  # fewer iters
"""

import argparse
import time
import ctypes
import torch
import torch.cuda as tc

libcudart = ctypes.CDLL("libcudart.so")
cudaMemcpyDeviceToHost = 2
libcudart.cudaMemcpy2DAsync.restype = ctypes.c_int
libcudart.cudaMemcpy2DAsync.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
]

SIZES = {
    "small":  (32,  512,  128, 8),
    "medium": (61,  2048, 512, 8),
    "large":  (80,  8192, 512, 8),
}

PATTERNS = {
    "contiguous": "contig",
    "few_seg":    "fewseg",
    "scattered":  "scatter",
}


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    tc.synchronize()
    times = []
    for _ in range(iters):
        tc.synchronize()
        t0 = time.perf_counter()
        fn()
        tc.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def make_block_ids(pattern, num_blocks):
    if pattern == "contiguous":
        return torch.arange(num_blocks, dtype=torch.int64)
    elif pattern == "few_seg":
        q = num_blocks // 4
        b = torch.arange(num_blocks, dtype=torch.int64)
        return torch.cat([b[0:q], b[2*q:3*q], b[q:2*q], b[3*q:4*q]])
    else:  # scattered
        gen = torch.Generator().manual_seed(42)
        return torch.randperm(num_blocks, generator=gen, dtype=torch.int64)


def compute_segments(block_ids, num_blocks):
    """Return list of (start_block_id, run_len, start_index_in_ids)."""
    segs = []
    k = 0
    while k < num_blocks:
        start = k
        while k + 1 < num_blocks and block_ids[k + 1].item() == block_ids[k].item() + 1:
            k += 1
        segs.append((block_ids[start].item(), k - start + 1, start))
        k += 1
    return segs


def run_case(size_name, num_layers, num_blocks, head_dim, num_gpus,
             layout, mode, pattern):
    ES = 2  # fp16
    full_chunk = head_dim * ES
    full_elems = full_chunk // 8
    total_data = num_blocks * num_layers * full_chunk
    total_mb = total_data / 1e6

    # Layout-dependent strides
    if layout == "LAYERFIRST":
        cpu_block_stride = full_chunk
        cpu_layer_stride = num_blocks * full_chunk
    else:  # BLOCKFIRST
        cpu_block_stride = num_layers * full_chunk
        cpu_layer_stride = full_chunk

    # Mode-dependent chunk_size and GPU participation
    is_mla = mode.startswith("mla_")
    if mode == "mla_sharded":
        shard_size = full_chunk // num_gpus
        shard_elems = shard_size // 8
        chunk_size = shard_size
        participating_gpus = list(range(num_gpus))
        gpu_shard_offset = lambda i: i * shard_size
    elif mode == "mla_rank0_only":
        chunk_size = full_chunk
        participating_gpus = [0]
        gpu_shard_offset = lambda i: 0
    elif mode == "mla_all_write":
        chunk_size = full_chunk
        participating_gpus = list(range(num_gpus))
        gpu_shard_offset = lambda i: 0  # all write full, different CPU region
    else:  # MHA
        chunk_size = full_chunk
        participating_gpus = list(range(num_gpus))
        gpu_shard_offset = lambda i: 0  # MHA: each GPU has own head data

    # GPU source: [num_layers, num_blocks, full_elems] (LAYERFIRST, int64)
    # For MLA all GPUs have same data; for MHA each GPU has different data
    gpu_srcs = []
    for g in range(num_gpus):
        if is_mla and g > 0:
            gpu_srcs.append(gpu_srcs[0])  # MLA: same data
        else:
            gpu_srcs.append(torch.randint(0, 1000, (num_layers, num_blocks, full_elems),
                                          dtype=torch.int64, device=f"cuda:{g % torch.cuda.device_count()}"))

    # CPU target (pinned)
    if mode == "mla_all_write":
        total_cpu = total_data * num_gpus
    else:
        total_cpu = total_data
    cpu_buf = torch.empty(total_cpu // 8, dtype=torch.int64, pin_memory=True)
    cpu_buf.zero_()

    block_ids = make_block_ids(pattern, num_blocks)
    segments = compute_segments(block_ids, num_blocks)
    block_ids_cuda = block_ids.cuda()

    dev = torch.cuda.current_device()
    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    results = {}

    # === A. baseline: per-block copy ===
    def op_baseline():
        for gi in participating_gpus:
            gsrc = gpu_srcs[gi]
            shard_off = gpu_shard_offset(gi)
            for layer in range(num_layers):
                for seg_bid, run_len, seg_k in segments:
                    for b in range(run_len):
                        bid = seg_bid + b
                        dst_off = (bid * cpu_block_stride + layer * cpu_layer_stride + shard_off) // 8
                        src_off = shard_off // 8
                        cpu_buf[dst_off:dst_off + chunk_size // 8].copy_(
                            gsrc[layer, bid, src_off:src_off + chunk_size // 8],
                            non_blocking=True)
                tc.synchronize()

    # === B. memcpy2d: per-(layer,segment) cudaMemcpy2DAsync ===
    def op_memcpy2d():
        for gi in participating_gpus:
            gsrc = gpu_srcs[gi]
            shard_off = gpu_shard_offset(gi)
            for layer in range(num_layers):
                for seg_bid, run_len, seg_k in segments:
                    src_ptr = gsrc.data_ptr() + (
                        layer * num_blocks * full_chunk + seg_bid * full_chunk + shard_off)
                    dst_ptr = cpu_buf.data_ptr() + (
                        seg_bid * cpu_block_stride + layer * cpu_layer_stride + shard_off)
                    libcudart.cudaMemcpy2DAsync(
                        dst_ptr, cpu_block_stride,
                        src_ptr, full_chunk,
                        chunk_size, run_len,
                        cudaMemcpyDeviceToHost, stream_ptr)
            tc.synchronize()

    # === C. d2d_3path: D2D transpose + 3-path cudaMemcpyAsync ===
    def op_d2d_3path():
        for gi in participating_gpus:
            gsrc = gpu_srcs[gi]
            shard_off = gpu_shard_offset(gi)
            shard_start = shard_off // 8

            # D2D: index_select + transpose + contiguous
            if mode == "mla_sharded":
                # Transpose only shard: [L, B, shard_elems] -> [B, L, shard_elems]
                shard_view = gsrc[:, :, shard_start:shard_start + shard_elems]
                gathered = torch.index_select(shard_view.reshape(num_layers, -1, shard_elems),
                                              1, block_ids_cuda)
                transposed = gathered.transpose(0, 1).contiguous()
                d2h_chunk = shard_size
            else:
                # Transpose full: [L, B, full_elems] -> [B, L, full_elems]
                gathered = torch.index_select(gsrc.reshape(num_layers, -1, full_elems),
                                              1, block_ids_cuda)
                transposed = gathered.transpose(0, 1).contiguous()
                d2h_chunk = full_chunk

            # 3-path D2H based on segments
            if len(segments) == 1 and segments[0][1] == num_blocks:
                # BULK_CONTIG: one shot
                dst_ptr = cpu_buf.data_ptr() + shard_off
                src_ptr = transposed.data_ptr()
                size = num_blocks * num_layers * d2h_chunk
                cpu_buf_shard = ctypes.cast(dst_ptr, ctypes.POINTER(ctypes.c_int8))
                # Use torch copy for contiguous
                cpu_buf_view = torch.frombuffer(
                    (ctypes.c_int8 * size).from_address(dst_ptr),
                    dtype=torch.uint8).view(torch.int64)
                cpu_buf_view[:size // 8].copy_(transposed.view(-1), non_blocking=True)
            else:
                # SEGMENTED / SCATTERED: per-segment or per-block
                blk_stride = num_layers * d2h_chunk
                for seg_bid, run_len, seg_k in segments:
                    if run_len > 1:
                        # SEGMENTED_DIRECT: contiguous per segment
                        dst_off = seg_bid * cpu_block_stride + shard_off
                        src_off = seg_k * num_layers * d2h_chunk
                        size = run_len * num_layers * d2h_chunk
                        cpu_buf_view = torch.frombuffer(
                            (ctypes.c_int8 * size).from_address(
                                cpu_buf.data_ptr() + dst_off),
                            dtype=torch.uint8).view(torch.int64)
                        src_view = transposed.view(-1)[src_off // 8:(src_off + size) // 8]
                        cpu_buf_view[:size // 8].copy_(src_view, non_blocking=True)
                    else:
                        # Per-block
                        for b in range(run_len):
                            bid = seg_bid + b
                            dst_off = bid * cpu_block_stride + shard_off
                            src_off = (seg_k + b) * num_layers * d2h_chunk
                            size = num_layers * d2h_chunk
                            cpu_buf_view = torch.frombuffer(
                                (ctypes.c_int8 * size).from_address(
                                    cpu_buf.data_ptr() + dst_off),
                                dtype=torch.uint8).view(torch.int64)
                            src_view = transposed.view(-1)[src_off // 8:(src_off + size) // 8]
                            cpu_buf_view[:size // 8].copy_(src_view, non_blocking=True)
            tc.synchronize()

    # === D. d2d_memcpy2d: D2D transpose + cudaMemcpy2DAsync ===
    def op_d2d_memcpy2d():
        for gi in participating_gpus:
            gsrc = gpu_srcs[gi]
            shard_off = gpu_shard_offset(gi)
            shard_start = shard_off // 8

            if mode == "mla_sharded":
                shard_view = gsrc[:, :, shard_start:shard_start + shard_elems]
                gathered = torch.index_select(shard_view.reshape(num_layers, -1, shard_elems),
                                              1, block_ids_cuda)
                transposed = gathered.transpose(0, 1).contiguous()
                d2h_chunk = shard_size
            else:
                gathered = torch.index_select(gsrc.reshape(num_layers, -1, full_elems),
                                              1, block_ids_cuda)
                transposed = gathered.transpose(0, 1).contiguous()
                d2h_chunk = full_chunk

            # Per-segment cudaMemcpy2DAsync
            for seg_bid, run_len, seg_k in segments:
                src_ptr = transposed.data_ptr() + seg_k * num_layers * d2h_chunk
                dst_ptr = cpu_buf.data_ptr() + seg_bid * cpu_block_stride + shard_off
                libcudart.cudaMemcpy2DAsync(
                    dst_ptr, cpu_layer_stride,
                    src_ptr, d2h_chunk,
                    d2h_chunk, num_layers * run_len,
                    cudaMemcpyDeviceToHost, stream_ptr)
            tc.synchronize()

    # === E. d2d_oneshot: D2D transpose + one contiguous D2H (rank0_only ref) ===
    def op_d2d_oneshot():
        gsrc = gpu_srcs[0]
        gathered = torch.index_select(gsrc.reshape(num_layers, -1, full_elems),
                                      1, block_ids_cuda)
        transposed = gathered.transpose(0, 1).contiguous()
        size = num_blocks * num_layers * full_elems
        cpu_buf[:size].copy_(transposed.view(-1), non_blocking=True)
        tc.synchronize()

    # Skip baseline for large (too slow)
    skip_baseline = (size_name == "large" and num_blocks >= 2048)

    ops = [
        ("B_memcpy2d", op_memcpy2d),
        ("C_d2d_3path", op_d2d_3path),
        ("D_d2d_memcpy2d", op_d2d_memcpy2d),
        ("E_d2d_oneshot", op_d2d_oneshot),
    ]
    if not skip_baseline:
        ops.insert(0, ("A_baseline", op_baseline))

    label = f"{size_name[:3]} {layout[:2]} {mode.replace('mla_','').replace('non_mla','mha')[:6]:6s} {PATTERNS[pattern]:7s}"
    row = {"label": label, "size": size_name, "layout": layout, "mode": mode, "pattern": pattern}

    for op_name, op_fn in ops:
        try:
            t = bench(op_fn, warmup=2, iters=5)
            row[op_name] = t
        except Exception as e:
            row[op_name] = float('inf')

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="all")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not tc.is_available():
        print("CUDA not available!")
        return

    print("=" * 120)
    print("  Comprehensive D2H Benchmark: layout × mode × pattern × approach")
    print("=" * 120)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    sizes = list(SIZES.keys()) if args.sizes == "all" else args.sizes.split(",")

    layouts = ["LAYERFIRST", "BLOCKFIRST"]
    modes = ["non_mla", "mla_rank0_only", "mla_sharded", "mla_all_write"]
    patterns = ["contiguous", "few_seg", "scattered"]

    all_rows = []
    for sz in sizes:
        cfg = SIZES[sz]
        for layout in layouts:
            for mode in modes:
                for pat in patterns:
                    # Skip scattered when num_blocks <= 2 (can't form segments)
                    if pat == "scattered" and cfg[1] <= 2:
                        continue
                    row = run_case(sz, *cfg, layout, mode, pat)
                    all_rows.append(row)

    # Print results table
    print("\n" + "=" * 120)
    print("  Results (ms) — lower is better, inf = skipped/failed")
    print("=" * 120)

    approach_names = ["A_baseline", "B_memcpy2d", "C_d2d_3path", "D_d2d_memcpy2d", "E_d2d_oneshot"]
    hdr = f"  {'Config':<35s}"
    for a in approach_names:
        hdr += f" {a:>13s}"
    print(hdr)
    print(f"  {'-'*105}")

    for r in all_rows:
        line = f"  {r['label']:<35s}"
        for a in approach_names:
            v = r.get(a, float('inf'))
            if v == float('inf'):
                line += f" {'skip':>13s}"
            else:
                line += f" {v:>12.3f}ms"
        print(line)

    # Find best approach per config
    print(f"\n  {'-'*105}")
    print("  Best approach per config:")
    print(f"  {'Config':<35s} {'Best':>15s} {'Time':>10s}")
    print(f"  {'-'*65}")
    for r in all_rows:
        best_name = None
        best_time = float('inf')
        for a in approach_names:
            v = r.get(a, float('inf'))
            if v < best_time:
                best_time = v
                best_name = a
        if best_name:
            print(f"  {r['label']:<35s} {best_name:>15s} {best_time:>9.3f}ms")

    # Summary by approach (wins count)
    print(f"\n  {'-'*65}")
    print("  Win count by approach:")
    wins = {a: 0 for a in approach_names}
    for r in all_rows:
        best_name = None
        best_time = float('inf')
        for a in approach_names:
            v = r.get(a, float('inf'))
            if v < best_time:
                best_time = v
                best_name = a
        if best_name:
            wins[best_name] += 1
    for a in approach_names:
        print(f"    {a:>15s}: {wins[a]:>3d} wins")
    print("=" * 120)


if __name__ == "__main__":
    main()
