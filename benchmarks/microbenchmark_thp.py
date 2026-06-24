"""
Microbenchmark: THP (Transparent Huge Pages) staging buffer evaluation.

Compares D2H and H2D transfer performance with and without THP-enabled CPU
staging buffers. THP uses mmap + madvise(MADV_HUGEPAGE) + optional
cudaHostRegister, matching the FlexKV CPUAllocator THP path.

Allocation modes:
  - normal   : torch.empty(pin_memory=False)  — baseline 4KB pages
  - pinned   : torch.empty(pin_memory=True)   — cudaHostRegister'd 4KB pages
  - thp      : mmap + madvise(MADV_HUGEPAGE)  — 2MB transparent huge pages
  - thp_pin  : mmap + madvise(MADV_HUGEPAGE) + cudaHostRegister — THP + async DMA

Metrics per allocation mode:
  - D2H latency (GPU→CPU copy + sync), bandwidth GB/s
  - H2D latency (CPU→GPU copy + sync), bandwidth GB/s
  - Allocation time (how long mmap+tensor creation takes)

Model sizes:
  - small :  32 layers,  512 head_dim,  128 blocks  (~0.5 GB)
  - medium:  61 layers, 2048 head_dim,  512 blocks  (~12 GB)
  - large :  80 layers, 8192 head_dim,  512 blocks  (~48 GB)

Usage:
    python benchmarks/microbenchmark_thp.py --iters 20
    python benchmarks/microbenchmark_thp.py --iters 20 --sizes small medium
    python benchmarks/microbenchmark_thp.py --mode thp thp_pin normal pinned
"""

import argparse
import ctypes
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    print("ERROR: PyTorch not available")
    sys.exit(1)

# ---------------------------------------------------------------------------
# THP allocation helpers (mirror FlexKV CPUAllocator)
# ---------------------------------------------------------------------------

_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_LIBC.munmap.restype = ctypes.c_int
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_MAP_FAILED = ctypes.c_void_p(-1).value
_MADV_HUGEPAGE = 14

DTYPE = torch.float16
ES = DTYPE.itemsize
WARMUP = 3

SIZES = {
    "small":  {"num_layers": 32, "head_dim": 512,  "num_blocks": 128},
    "medium": {"num_layers": 61, "head_dim": 2048, "num_blocks": 512},
    "large":  {"num_layers": 80, "head_dim": 8192, "num_blocks": 512},
}


def alloc_normal(num_bytes, dtype, total_size):
    return torch.empty(total_size, dtype=dtype, device="cpu", pin_memory=False)

def alloc_pinned(num_bytes, dtype, total_size):
    return torch.empty(total_size, dtype=dtype, device="cpu", pin_memory=True)

def alloc_thp(num_bytes, dtype, total_size):
    aligned = (num_bytes + 4095) & ~4095
    ptr = _LIBC.mmap(None, aligned, _PROT_READ | _PROT_WRITE,
                     _MAP_PRIVATE | _MAP_ANONYMOUS, -1, 0)
    if ptr == _MAP_FAILED:
        raise RuntimeError("mmap failed: " + os.strerror(ctypes.get_errno()))
    _LIBC.madvise(ptr, aligned, _MADV_HUGEPAGE)
    tensor = torch.frombuffer(
        (ctypes.c_char * num_bytes).from_address(ptr), dtype=dtype
    ).reshape(total_size)
    return tensor, ptr, aligned

def alloc_thp_pin(num_bytes, dtype, total_size):
    tensor, ptr, aligned = alloc_thp(num_bytes, dtype, total_size)
    try:
        err = torch.cuda.cudart().cudaHostRegister(ptr, aligned, 0)
        if isinstance(err, tuple):
            err = err[0]
        if err != 0:
            print("  [WARN] cudaHostRegister failed (err=" + str(err) + ")")
    except Exception as e:
        print("  [WARN] cudaHostRegister exception: " + str(e))
    return tensor, ptr, aligned

def free_thp(ptr, aligned):
    _LIBC.munmap(ptr, aligned)

ALLOC_MODES = {
    "normal":  alloc_normal,
    "pinned":  alloc_pinned,
    "thp":     alloc_thp,
    "thp_pin": alloc_thp_pin,
}


def compute_buffer_size(num_layers, head_dim, num_blocks):
    return 2 * num_layers * head_dim * num_blocks * ES


def run_transfer_bench(gpu_tensor, cpu_tensor, num_iters, direction):
    latencies = []
    for i in range(num_iters + WARMUP):
        if i < WARMUP:
            if direction == "d2h":
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
            else:
                cpu_tensor.copy_(gpu_tensor, non_blocking=True)
            torch.cuda.synchronize()
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if direction == "d2h":
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        else:
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    arr = np.array(latencies)
    return float(arr.mean()), float(arr.std())


def run_size_bench(size_name, size_cfg, modes, num_iters, gpu_id):
    nl = size_cfg["num_layers"]
    hd = size_cfg["head_dim"]
    nb = size_cfg["num_blocks"]
    num_bytes = compute_buffer_size(nl, hd, nb)
    total_elements = num_bytes // ES
    size_gb = num_bytes / (1024 ** 3)

    print("\n" + "=" * 70)
    print("Size: " + size_name + "  (" + str(nl) + "L, hd=" + str(hd) + ", " + str(nb) + "B, " + str(round(size_gb, 2)) + " GB, dtype=fp16)")
    print("=" * 70)

    device = "cuda:" + str(gpu_id)
    gpu_tensor = torch.empty(total_elements, dtype=DTYPE, device=device)
    gpu_tensor.fill_(1.0)

    results = {}

    for mode in modes:
        print("\n  [" + mode + "] allocating CPU buffer (" + str(round(size_gb, 2)) + " GB)...")
        t_alloc_start = time.perf_counter()
        fn = ALLOC_MODES[mode]
        if mode in ("thp", "thp_pin"):
            cpu_tensor, mmap_ptr, mmap_size = fn(num_bytes, DTYPE, total_elements)
        else:
            cpu_tensor = fn(num_bytes, DTYPE, total_elements)
            mmap_ptr = None
        torch.cuda.synchronize()
        t_alloc_end = time.perf_counter()
        alloc_ms = (t_alloc_end - t_alloc_start) * 1000
        print("    allocation: " + str(round(alloc_ms, 1)) + " ms")

        d2h_mean, d2h_std = run_transfer_bench(gpu_tensor, cpu_tensor, num_iters, "d2h")
        d2h_bw = size_gb / (d2h_mean / 1000) if d2h_mean > 0 else 0

        h2d_mean, h2d_std = run_transfer_bench(gpu_tensor, cpu_tensor, num_iters, "h2d")
        h2d_bw = size_gb / (h2d_mean / 1000) if h2d_mean > 0 else 0

        results[mode] = {
            "alloc_ms": alloc_ms,
            "d2h_mean_ms": d2h_mean,
            "d2h_std_ms": d2h_std,
            "d2h_bw_gbs": d2h_bw,
            "h2d_mean_ms": h2d_mean,
            "h2d_std_ms": h2d_std,
            "h2d_bw_gbs": h2d_bw,
        }

        print("    D2H: " + str(round(d2h_mean, 2)) + " +/- " + str(round(d2h_std, 2)) + " ms  (" + str(round(d2h_bw, 2)) + " GB/s)")
        print("    H2D: " + str(round(h2d_mean, 2)) + " +/- " + str(round(h2d_std, 2)) + " ms  (" + str(round(h2d_bw, 2)) + " GB/s)")

        del cpu_tensor
        if mmap_ptr is not None:
            free_thp(mmap_ptr, mmap_size)
        torch.cuda.empty_cache()

    del gpu_tensor
    torch.cuda.empty_cache()

    print("\n  --- Summary: " + size_name + " (" + str(round(size_gb, 2)) + " GB) ---")
    print("  " + "Mode".ljust(12) + "Alloc".rjust(9) + "D2H ms".rjust(11) + "D2H GB/s".rjust(11) + "H2D ms".rjust(11) + "H2D GB/s".rjust(11))
    print("  " + "-" * 12 + " " + "-" * 8 + " " + "-" * 10 + " " + "-" * 10 + " " + "-" * 10 + " " + "-" * 10)
    for mode in modes:
        r = results[mode]
        print("  " + mode.ljust(12) + str(round(r["alloc_ms"], 1)).rjust(8) + " " +
              str(round(r["d2h_mean_ms"], 2)).rjust(10) + " " +
              str(round(r["d2h_bw_gbs"], 2)).rjust(10) + " " +
              str(round(r["h2d_mean_ms"], 2)).rjust(10) + " " +
              str(round(r["h2d_bw_gbs"], 2)).rjust(10))

    if "normal" in results:
        base_d2h = results["normal"]["d2h_mean_ms"]
        base_h2d = results["normal"]["h2d_mean_ms"]
        print("\n  Speedup vs normal (4KB pages):")
        for mode in modes:
            if mode == "normal":
                continue
            r = results[mode]
            d2x = base_d2h / r["d2h_mean_ms"] if r["d2h_mean_ms"] > 0 else 0
            h2x = base_h2d / r["h2d_mean_ms"] if r["h2d_mean_ms"] > 0 else 0
            print("    " + mode.ljust(12) + "D2H " + str(round(d2x, 2)) + "x  H2D " + str(round(h2x, 2)) + "x")

    return results


def main():
    parser = argparse.ArgumentParser(description="THP staging buffer microbenchmark")
    parser.add_argument("--iters", type=int, default=20, help="Iterations per measurement")
    parser.add_argument("--sizes", nargs="+", default=["small", "medium"],
                        choices=list(SIZES.keys()), help="Model sizes to test")
    parser.add_argument("--modes", nargs="+", default=list(ALLOC_MODES.keys()),
                        choices=list(ALLOC_MODES.keys()),
                        help="Allocation modes to test")
    args = parser.parse_args()

    if not CUDA_AVAILABLE:
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_id = 0
    print("GPU: " + torch.cuda.get_device_name(gpu_id))
    print("Iterations: " + str(args.iters) + " (warmup: " + str(WARMUP) + ")")
    print("Modes: " + str(args.modes))
    print("Sizes: " + str(args.sizes))

    all_results = {}
    for size_name in args.sizes:
        cfg = SIZES[size_name]
        all_results[size_name] = run_size_bench(size_name, cfg, args.modes, args.iters, gpu_id)

    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)
    for size_name, results in all_results.items():
        cfg = SIZES[size_name]
        size_gb = compute_buffer_size(cfg["num_layers"], cfg["head_dim"], cfg["num_blocks"]) / (1024 ** 3)
        print("\n  " + size_name + " (" + str(round(size_gb, 2)) + " GB):")
        for mode, r in results.items():
            print("    " + mode.ljust(12) + " D2H " + str(round(r["d2h_mean_ms"], 2)).rjust(8) + "ms (" +
                  str(round(r["d2h_bw_gbs"], 2)).rjust(6) + " GB/s)  H2D " +
                  str(round(r["h2d_mean_ms"], 2)).rjust(8) + "ms (" +
                  str(round(r["h2d_bw_gbs"], 2)).rjust(6) + " GB/s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
