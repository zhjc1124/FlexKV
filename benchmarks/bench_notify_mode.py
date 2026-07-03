"""
Benchmark: cudaLaunchHostFunc vs event polling notification overhead.

Measures the stream-blocking effect of cudaLaunchHostFunc callbacks.
When callback overhead dominates transfer time, polling should be faster.

Key parameters that maximize the ratio:
  - layer_granularity=1 (max batches = max callbacks)
  - num_layers=80 (80 callbacks per transfer)
  - tp_size=8 (8 eventfd write() per callback)
  - head_dim=16 (tiny transfer per batch)
  - num_blocks=1 (minimal data)

Usage:
    # Default (hostfunc on NVIDIA)
    python benchmarks/bench_notify_mode.py

    # Force polling
    FLEXKV_LAYERWISE_NOTIFY_MODE=polling python benchmarks/bench_notify_mode.py

    # Test both modes back-to-back
    python benchmarks/bench_notify_mode.py --both
"""

import argparse
import os
import sys
import time
import struct

import numpy as np
import torch

try:
    from flexkv.c_ext import LayerwiseTransferGroup
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    FLEXKV_AVAILABLE = True
except ImportError as e:
    FLEXKV_AVAILABLE = False
    print(f"ERROR: FlexKV not available ({e})")
    sys.exit(1)

DTYPE = torch.float16
ES = DTYPE.itemsize


def create_eventfds(tp_size, num_layers, num_counters=2):
    """Create real eventfd file descriptors."""
    total = num_counters * tp_size * num_layers
    fds = []
    for _ in range(total):
        fd = os.eventfd(0, os.EFD_CLOEXEC)
        fds.append(fd)
    fds_tensor = torch.tensor(fds, dtype=torch.int32)
    return fds_tensor, fds


def drain_eventfds(fds):
    """Read and drain all eventfds (non-blocking)."""
    for fd in fds:
        try:
            os.read(fd, 8)
        except BlockingIOError:
            pass


def bench_one(num_gpus, num_layers, num_blocks, head_dim, tp_size,
              layer_granularity, iters):
    """Run layerwise_transfer with tiny data and measure wall time."""

    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=1,
        head_size=head_dim, is_mla=True)

    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=1,
        head_size=head_dim, is_mla=True)

    gpu_blocks = []
    for g in range(num_gpus):
        per_gpu = []
        for l in range(num_layers):
            t = torch.zeros((num_blocks, 1, 1, head_dim), dtype=DTYPE,
                           device=f"cuda:{g}")
            per_gpu.append(t)
        gpu_blocks.append(per_gpu)

    cpu_blocks = torch.zeros(tuple(cpu_layout.kv_shape), dtype=DTYPE,
                             pin_memory=True)

    fds_tensor, fds_list = create_eventfds(tp_size, num_layers, num_counters=2)

    ssd_files = {}

    gpu_kv_stride = gpu_layout.get_kv_stride() * ES
    gpu_block_stride = gpu_layout.get_block_stride() * ES
    gpu_layer_stride = gpu_layout.get_layer_stride() * ES
    gpu_chunk_size = gpu_layout.get_chunk_size() * ES

    gpu_kv_strides = torch.tensor([gpu_kv_stride] * num_gpus, dtype=torch.int64)
    gpu_block_strides = torch.tensor([gpu_block_stride] * num_gpus, dtype=torch.int64)
    gpu_layer_strides = torch.tensor([gpu_layer_stride] * num_gpus, dtype=torch.int64)
    gpu_chunk_sizes = torch.tensor([gpu_chunk_size] * num_gpus, dtype=torch.int64)

    ltg = LayerwiseTransferGroup(
        num_gpus=num_gpus,
        gpu_blocks=gpu_blocks,
        cpu_blocks=cpu_blocks,
        ssd_files=ssd_files,
        num_layers=num_layers,
        gpu_kv_strides_tensor=gpu_kv_strides,
        gpu_block_strides_tensor=gpu_block_strides,
        gpu_layer_strides_tensor=gpu_layer_strides,
        gpu_chunk_sizes_tensor=gpu_chunk_sizes,
        iouring_entries=32,
        iouring_flags=0,
        layer_eventfds_tensor=fds_tensor,
        tp_size=tp_size,
    )

    gpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()

    cpu_kv_stride = cpu_layout.get_kv_stride() * ES
    cpu_layer_stride = cpu_layout.get_layer_stride() * ES
    cpu_block_stride = cpu_layout.get_block_stride() * ES
    cpu_chunk_size = cpu_layout.get_chunk_size() * ES
    cpu_tp_stride = cpu_block_stride // tp_size

    empty_tensor = torch.empty(0, dtype=torch.int64).pin_memory()

    # Warmup
    for _ in range(3):
        ltg.layerwise_transfer(
            empty_tensor, empty_tensor,
            0, 0,
            0, 0,
            16,
            gpu_block_ids, cpu_block_ids,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            cpu_chunk_size,
            cpu_kv_stride, cpu_layer_stride,
            cpu_tp_stride,
            4,
            False,
            num_layers,
            layer_granularity,
            True,
            0,
        )
        drain_eventfds(fds_list)

    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(iters):
        drain_eventfds(fds_list)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ltg.layerwise_transfer(
            empty_tensor, empty_tensor,
            0, 0,
            0, 0,
            16,
            gpu_block_ids, cpu_block_ids,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            cpu_chunk_size,
            cpu_kv_stride, cpu_layer_stride,
            cpu_tp_stride,
            4,
            False,
            num_layers,
            layer_granularity,
            True,
            0,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    del ltg

    for fd in fds_list:
        os.close(fd)

    return {
        "avg_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: hostfunc vs polling notification overhead")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (default: all available)")
    parser.add_argument("--iters", type=int, default=50,
                        help="Timing iterations (default: 50)")
    parser.add_argument("--both", action="store_true",
                        help="Test both modes back-to-back")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count() if args.num_gpus <= 0 else min(args.num_gpus, torch.cuda.device_count())
    if num_gpus < 1:
        print("ERROR: need at least 1 GPU")
        sys.exit(1)

    CONFIGS = [
        ("extreme",   80, 1, 16,  min(num_gpus, 8), 1),
        ("high",      80, 1, 64,  min(num_gpus, 4), 1),
        ("medium",    32, 1, 128, min(num_gpus, 4), 1),
        ("low",       80, 64, 512, min(num_gpus, 4), 10),
    ]

    modes_to_test = ["hostfunc", "polling"] if args.both else [None]

    print("=" * 80)
    print("  Layerwise Notification Mode Benchmark")
    print("=" * 80)
    print(f"  GPUs: {num_gpus}")
    print(f"  Iters: {args.iters}")
    if args.both:
        print(f"  Modes: hostfunc, polling")
    else:
        current = os.environ.get("FLEXKV_LAYERWISE_NOTIFY_MODE", "hostfunc (default)")
        print(f"  Mode: {current}")
    print("=" * 80)

    for label, nl, nb, hd, tp, gran in CONFIGS:
        actual_tp = min(tp, num_gpus)
        data_bytes = nl * 1 * nb * 1 * 1 * hd * ES
        num_batches = (nl + gran - 1) // gran
        writes_per_call = actual_tp * gran
        total_writes = num_batches * writes_per_call

        print(f"\n--- {label}: {nl}L / {nb}B / hd={hd} / tp={actual_tp} / gran={gran} ---")
        print(f"    data={data_bytes} bytes, batches={num_batches}, "
              f"writes/batch={writes_per_call}, total_writes={total_writes}")

        for mode in modes_to_test:
            if mode is not None:
                os.environ["FLEXKV_LAYERWISE_NOTIFY_MODE"] = mode
                tag = mode
            else:
                tag = os.environ.get("FLEXKV_LAYERWISE_NOTIFY_MODE", "hostfunc")
                if tag == "":
                    tag = "hostfunc"

            print(f"  [{tag:>8}] ", end="", flush=True)
            try:
                r = bench_one(num_gpus, nl, nb, hd, actual_tp, gran, args.iters)
                print(f"avg={r['avg_ms']:.3f}ms  p50={r['p50_ms']:.3f}ms  "
                      f"p99={r['p99_ms']:.3f}ms  min={r['min_ms']:.3f}ms")
            except Exception as e:
                print(f"FAILED: {e}")

    if args.both:
        print("\n" + "=" * 80)
        print("  Comparison Summary")
        print("=" * 80)
        print("  If polling is faster on 'extreme' config, the stream-blocking")
        print("  effect of cudaLaunchHostFunc is measurable. In production (large")
        print("  data transfers), this difference is negligible.")
        print("=" * 80)


if __name__ == "__main__":
    main()
