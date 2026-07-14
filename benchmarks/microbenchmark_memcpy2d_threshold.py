"""
Microbenchmark: memcpy2d run_len threshold — when does cudaMemcpy2DAsync
pay off vs staging+scatter?

Sweeps segment run_len (contiguous blocks per segment) from 1 to N.
For each run_len, measures D2H time with:
  - enable_memcpy2d=1  -> memcpy2d branch
  - enable_memcpy2d=0  -> staging+scatter

Modes:
  --sharded : STAGED_BLOCK (sharded D2H, force_path=3, mla_d2h_mode=sharded)
  default   : STAGED_MERGE (rank0_only, force_path=2, mla_d2h_mode=rank0_only)

Usage:
    # STAGED_MERGE (rank0_only)
    python benchmarks/microbenchmark_memcpy2d_threshold.py --num-gpus 8 --iters 20

    # STAGED_BLOCK (sharded D2H)
    python benchmarks/microbenchmark_memcpy2d_threshold.py --num-gpus 8 --iters 20 --sharded
"""

import argparse
import sys
import time

import torch

try:
    from flexkv.c_ext import TPTransferThreadGroup
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    FLEXKV_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: FlexKV not available ({e})")
    sys.exit(1)

DTYPE = torch.float16
ES = DTYPE.itemsize


def make_pattern(num_blocks, run_len):
    """Block-id permutation with exactly run_len contiguous blocks per segment."""
    n_segs = num_blocks // run_len
    ids = torch.arange(num_blocks, dtype=torch.int64)
    chunks = [ids[i*run_len:(i+1)*run_len] for i in range(n_segs)]
    perm = torch.randperm(n_segs, generator=torch.Generator().manual_seed(42))
    return torch.cat([chunks[i] for i in perm]).pin_memory()


def bench_one(num_gpus, num_layers, num_blocks, tpb, head_dim, run_len,
              enable_memcpy2d, iters, sharded=False):
    """Benchmark D2H. If sharded, uses STAGED_BLOCK (force_path=3); else STAGED_MERGE (force_path=2)."""
    is_mla = True
    kv_dim = 1
    heads = 1

    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=heads, head_size=head_dim, is_mla=is_mla)

    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=heads, head_size=head_dim, is_mla=is_mla)

    cpu_stride_kv = cpu_layout.get_kv_stride() * ES
    cpu_stride_layer = cpu_layout.get_layer_stride() * ES
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus
    chunk_size = gpu_layout.get_chunk_size() * ES

    # GPU data
    all_gpu = []
    gpu_ptrs = []
    for g in range(num_gpus):
        full = torch.zeros(
            (num_layers, kv_dim, num_blocks, tpb, heads, head_dim),
            dtype=DTYPE, device=f"cuda:{g}")
        full[:] = torch.randn_like(full) * 0.01
        per_layer = [full[i] for i in range(num_layers)]
        all_gpu.append(per_layer)
        for l in range(num_layers):
            gpu_ptrs.append(per_layer[l].data_ptr())

    cpu_kv = torch.zeros(tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)

    mode = "sharded" if sharded else "rank0_only"
    force_path = 3 if sharded else 2  # STAGED_BLOCK=3, STAGED_MERGE=2

    tp = TPTransferThreadGroup(
        num_gpus=num_gpus,
        gpu_block_ptrs_flat=gpu_ptrs,
        num_tensors_per_gpu=num_layers,
        cpu_blocks_ptr=cpu_kv.data_ptr(),
        num_layers=num_layers,
        gpu_kv_strides_in_bytes=[gpu_layout.get_kv_stride() * ES] * num_gpus,
        gpu_block_strides_in_bytes=[gpu_layout.get_block_stride() * ES] * num_gpus,
        gpu_layer_strides_in_bytes=[gpu_layout.get_layer_stride() * ES] * num_gpus,
        gpu_chunk_sizes_in_bytes=[chunk_size] * num_gpus,
        gpu_device_ids=list(range(num_gpus)),
        enable_nvcomp=False,
        ce_path_opt=True,
        ce_segment_threshold=999,
        ce_force_path=force_path,
        ce_enable_memcpy2d=enable_memcpy2d,
        ce_is_blockfirst=True,
        ce_is_mla=True)

    block_ids = make_pattern(num_blocks, run_len)

    # Warmup
    for _ in range(3):
        tp.tp_group_transfer(
            gpu_block_id_tensor=block_ids, cpu_block_id_tensor=block_ids,
            cpu_kv_stride_in_bytes=cpu_stride_kv,
            cpu_layer_stride_in_bytes=cpu_stride_layer,
            cpu_block_stride_in_bytes=cpu_stride_block,
            cpu_tp_stride_in_bytes=cpu_stride_tp,
            transfer_num_cta=16, is_host_to_device=False, use_ce_transfer=True,
            layer_id=0, layer_granularity=num_layers, is_mla=True,
            mla_d2h_mode=mode)
        torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tp.tp_group_transfer(
            gpu_block_id_tensor=block_ids, cpu_block_id_tensor=block_ids,
            cpu_kv_stride_in_bytes=cpu_stride_kv,
            cpu_layer_stride_in_bytes=cpu_stride_layer,
            cpu_block_stride_in_bytes=cpu_stride_block,
            cpu_tp_stride_in_bytes=cpu_stride_tp,
            transfer_num_cta=16, is_host_to_device=False, use_ce_transfer=True,
            layer_id=0, layer_granularity=num_layers, is_mla=True,
            mla_d2h_mode=mode)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(sorted(times)[2:-2]) / max(len(times) - 4, 1)
    del tp
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=80)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--tpb", type=int, default=16)
    parser.add_argument("--sharded", action="store_true",
                        help="Test STAGED_BLOCK (sharded D2H) instead of STAGED_MERGE (rank0_only)")
    args = parser.parse_args()

    num_gpus = args.num_gpus
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        print(f"Need {num_gpus} GPUs, found {torch.cuda.device_count()}")
        sys.exit(1)

    run_lens = [1, 2, 4, 8, 16, 32, 64, 128, args.num_blocks]
    run_lens = [r for r in run_lens if r <= args.num_blocks and args.num_blocks % r == 0]

    chunk = args.tpb * 1 * args.head_dim * 1 * ES
    total_mb = args.num_layers * args.num_blocks * chunk / 1024 / 1024
    mode_label = "STAGED_BLOCK (sharded)" if args.sharded else "STAGED_MERGE (rank0_only)"

    print("=" * 80)
    print(f"  memcpy2d Threshold Benchmark — {mode_label}")
    print(f"  GPUs={num_gpus}, layers={args.num_layers}, blocks={args.num_blocks}, "
          f"hd={args.head_dim}, tpb={args.tpb}")
    print(f"  chunk_size={chunk} bytes, total D2H = {total_mb:.1f} MB")
    print("=" * 80)
    print()
    print(f"{'run_len':>8} {'segs':>6} {'memcpy2d(ms)':>14} {'staging(ms)':>14} "
          f"{'speedup':>10} {'winner':>10}")
    print("-" * 70)

    results = []
    for rl in run_lens:
        n_segs = args.num_blocks // rl
        try:
            t_m2d = bench_one(num_gpus, args.num_layers, args.num_blocks,
                              args.tpb, args.head_dim, rl, True, args.iters, args.sharded)
            t_stg = bench_one(num_gpus, args.num_layers, args.num_blocks,
                              args.tpb, args.head_dim, rl, False, args.iters, args.sharded)
        except Exception as e:
            print(f"{rl:>8} {n_segs:>6} FAILED: {e}")
            continue

        speedup = t_stg / t_m2d if t_m2d > 0 else 0
        winner = "memcpy2d" if t_m2d < t_stg else "staging"
        print(f"{rl:>8} {n_segs:>6} {t_m2d:>14.3f} {t_stg:>14.3f} "
              f"{speedup:>9.2f}x {winner:>10}")
        results.append((rl, n_segs, t_m2d, t_stg, speedup, winner))

    print()
    print("=" * 70)
    crossover = next((rl for rl, _, _, _, _, w in results if w == "memcpy2d"), None)
    if crossover:
        print(f"  memcpy2d becomes faster at run_len >= {crossover}")
        print(f"  Recommended memcpy2d threshold: run_len >= {crossover}")
    else:
        print(f"  memcpy2d never faster than staging in this configuration")
        print(f"  Recommended: keep enable_memcpy2d=0 for {mode_label}")
    print("=" * 70)


if __name__ == "__main__":
    main()
