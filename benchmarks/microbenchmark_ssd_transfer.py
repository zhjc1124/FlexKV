#!/usr/bin/env python3
"""D2H/H2D SSD transfer microbenchmark: optimization before vs after.

Evaluates SSD↔CPU transfer performance across:
  - Model sizes: small / medium / large
  - Layouts: BLOCKFIRST / LAYERFIRST
  - MLA modes: MHA (multi-head) / MLA (multi-latent)
  - Block ordering: contiguous (optimized) vs scattered (fragmented)

Optimization paths in transfer_ssd.cpp:
  BLOCKFIRST + contiguous  → single large I/O per block (enable_block_first_transfer)
  LAYERFIRST + contiguous  → layer-major batch I/O (new optimization)
  LAYERFIRST + scattered   → per-layer fragmented I/O (original unoptimized path)

Usage:
  PYTHONPATH=. python benchmarks/microbenchmark_ssd_transfer.py
  PYTHONPATH=. python benchmarks/microbenchmark_ssd_transfer.py --models MHA MLA --sizes small medium large --engines fallback
"""
import argparse
import mmap
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

TOKENS_PER_BLOCK = 16


@dataclass
class ModelPreset:
    name: str
    num_layers: int
    num_kv_heads: int
    head_size: int
    is_mla: bool
    dtype: torch.dtype = torch.bfloat16

    @property
    def kv_dim(self) -> int:
        return 1 if self.is_mla else 2

    @property
    def elem_size(self) -> int:
        return self.dtype.itemsize

    @property
    def chunk_size(self) -> int:
        return TOKENS_PER_BLOCK * self.num_kv_heads * self.head_size

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_size * self.elem_size


MODEL_SIZES: Dict[str, Dict[str, int]] = {
    "small":  {"num_layers": 32, "num_kv_heads": 8,  "head_size": 128},
    "medium": {"num_layers": 80, "num_kv_heads": 8,  "head_size": 128},
    "large":  {"num_layers": 80, "num_kv_heads": 16, "head_size": 256},
}


def build_presets(sizes: List[str], mla_modes: List[str]) -> Dict[str, ModelPreset]:
    presets = {}
    for size in sizes:
        cfg = MODEL_SIZES[size]
        for mode in mla_modes:
            if mode == "MHA":
                name = f"{size}-MHA"
                presets[name] = ModelPreset(name, cfg["num_layers"], cfg["num_kv_heads"], cfg["head_size"], is_mla=False)
            elif mode == "MLA":
                name = f"{size}-MLA"
                presets[name] = ModelPreset(name, cfg["num_layers"], 1, cfg["head_size"], is_mla=True)
    return presets


def make_layout(layout_type: KVCacheLayoutType, preset: ModelPreset, num_blocks: int) -> KVCacheLayout:
    return KVCacheLayout(
        type=layout_type,
        num_layer=preset.num_layers,
        num_block=num_blocks,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=preset.num_kv_heads,
        head_size=preset.head_size,
        is_mla=preset.is_mla,
    )


def compute_strides(layout: KVCacheLayout, preset: ModelPreset) -> dict:
    return {
        "chunk_bytes": preset.chunk_bytes,
        "block_stride_bytes": layout.get_block_stride() * preset.elem_size,
        "layer_stride_bytes": layout.get_layer_stride() * preset.elem_size,
        "kv_stride_bytes": layout.get_kv_stride() * preset.elem_size,
        "total_bytes": layout.get_total_elements() * preset.elem_size,
    }


@dataclass
class BenchResult:
    model: str
    size: str
    layout: str
    mla_mode: str
    num_blocks: int
    block_order: str
    direction: str
    io_engine: str
    threads: int
    time_ms: float
    bandwidth_gbs: float
    io_pattern: str
    data_bytes: int


def run_single_transfer(ioctx, transfer_fn, cpu_tensor, preset, strides,
                         ssd_block_ids, cpu_block_ids, is_read, num_threads) -> float:
    layer_ids = torch.arange(0, preset.num_layers, dtype=torch.int32)
    ssd_ids_t = torch.from_numpy(ssd_block_ids.astype(np.int64))
    cpu_ids_t = torch.from_numpy(cpu_block_ids.astype(np.int64))

    start = time.perf_counter()
    transfer_fn(
        ioctx=ioctx,
        cpu_layer_id_list=layer_ids,
        cpu_tensor_ptr=cpu_tensor.data_ptr(),
        ssd_block_ids=ssd_ids_t,
        cpu_block_ids=cpu_ids_t,
        cpu_layer_stride_in_bytes=strides["layer_stride_bytes"],
        cpu_kv_stride_in_bytes=strides["kv_stride_bytes"],
        ssd_layer_stride_in_bytes=strides["layer_stride_bytes"],
        ssd_kv_stride_in_bytes=strides["kv_stride_bytes"],
        chunk_size_in_bytes=strides["chunk_bytes"],
        block_stride_in_bytes=strides["block_stride_bytes"],
        is_read=is_read,
        num_blocks_per_file=len(ssd_block_ids),
        round_robin=1,
        num_threads_per_device=num_threads,
        is_mla=preset.is_mla,
    )
    try:
        ioctx.get_iouring().wait_completion()
    except Exception:
        pass
    return (time.perf_counter() - start) * 1000


def make_block_ids(num_blocks: int, block_order: str) -> np.ndarray:
    if block_order == "contiguous":
        return np.arange(num_blocks, dtype=np.int64)
    elif block_order == "scattered":
        rng = np.random.default_rng(42)
        ids = np.arange(num_blocks, dtype=np.int64)
        rng.shuffle(ids)
        return ids
    else:
        raise ValueError(f"Unknown block_order: {block_order}")


def benchmark_config(
    preset: ModelPreset, size_label: str, layout_type: KVCacheLayoutType,
    num_blocks: int, block_order: str, rounds: int, warmup: int,
    threads: int, io_engine: str, ssd_path: Path, SSDIOCTX, transfer_fn,
) -> List[BenchResult]:
    layout = make_layout(layout_type, preset, num_blocks)
    strides = compute_strides(layout, preset)
    total_bytes = strides["total_bytes"]

    mm = mmap.mmap(-1, total_bytes, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    cpu_tensor = torch.frombuffer(mm, dtype=preset.dtype)
    rng = np.random.default_rng(42)
    mm.write(rng.integers(0, 255, size=total_bytes, dtype=np.uint8).tobytes())
    mm.seek(0)

    with open(ssd_path, "wb") as f:
        f.truncate(total_bytes)
        os.fsync(f.fileno())
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = os.open(str(ssd_path), os.O_RDWR)
    libc.fallocate(fd, 0, 0, total_bytes)
    os.close(fd)

    iouring_entries = 512 if io_engine == "iouring" else 0
    ioctx = SSDIOCTX({0: [str(ssd_path)]}, 1, iouring_entries, 0)

    block_ids = make_block_ids(num_blocks, block_order)
    is_bf = strides["block_stride_bytes"] > strides["layer_stride_bytes"]

    if is_bf:
        io_pattern = "BF: single large I/O per block"
    elif block_order == "contiguous":
        io_pattern = "LF-opt: layer-major batch I/O"
    else:
        io_pattern = "LF-orig: per-layer fragmented I/O"

    mla_mode = "MLA" if preset.is_mla else "MHA"
    results = []

    for direction, is_read in [("SSD→CPU", True), ("CPU→SSD", False)]:
        times = []
        for r in range(warmup + rounds):
            t = run_single_transfer(
                ioctx, transfer_fn, cpu_tensor, preset, strides,
                block_ids, block_ids, is_read, threads,
            )
            if r >= warmup:
                times.append(t)
        med = statistics.median(times)
        bw = total_bytes / (med / 1000) / (1024**3) if med > 0 else 0
        results.append(BenchResult(
            model=preset.name, size=size_label, layout=layout_type.value,
            mla_mode=mla_mode, num_blocks=num_blocks, block_order=block_order,
            direction=direction, io_engine=io_engine, threads=threads,
            time_ms=med, bandwidth_gbs=bw, io_pattern=io_pattern,
            data_bytes=total_bytes,
        ))

    mm.close()
    return results


def print_results(results: List[BenchResult]):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r.size, r.mla_mode, r.num_blocks, r.io_engine, r.direction, r.threads)
        groups[key].append(r)

    print()
    print("=" * 120)
    print("  D2H/H2D SSD Transfer Microbenchmark: Optimization Before vs After")
    print("=" * 120)

    for (size, mla, nblocks, engine, direction, threads), rs in sorted(groups.items()):
        data_gb = rs[0].data_bytes / (1024**3)
        print(f"\n  [{size}] {mla} | blocks={nblocks} ({data_gb:.2f}GB) | {engine} | {direction} | threads={threads}")
        print(f"  {'Pattern':<45} {'Time(ms)':>10} {'BW(GB/s)':>10}")
        print(f"  {'-'*65}")
        for r in sorted(rs, key=lambda x: x.time_ms):
            print(f"  {r.io_pattern:<45} {r.time_ms:>10.2f} {r.bandwidth_gbs:>10.2f}")

    # Optimization impact summary
    print("\n" + "=" * 120)
    print("  Optimization Impact: LF layer-major vs LF fragmented (contiguous vs scattered)")
    print("=" * 120)
    print(f"  {'Size':<8} {'MLA':<5} {'Blocks':<8} {'Dir':<10} {'LF-opt(ms)':>12} {'LF-orig(ms)':>12} {'Speedup':>10}")
    print("  " + "-" * 75)

    for (size, mla, nblocks, engine, direction, threads), rs in sorted(groups.items()):
        lf_opt = next((r for r in rs if r.layout == "LAYERFIRST" and r.block_order == "contiguous"), None)
        lf_orig = next((r for r in rs if r.layout == "LAYERFIRST" and r.block_order == "scattered"), None)
        if not lf_opt or not lf_orig:
            continue
        speedup = lf_orig.time_ms / lf_opt.time_ms if lf_opt.time_ms > 0 else 0
        print(f"  {size:<8} {mla:<5} {nblocks:<8} {direction:<10} {lf_opt.time_ms:>12.2f} {lf_orig.time_ms:>12.2f} {speedup:>9.2f}x")

    # BF vs LF comparison
    print("\n" + "=" * 120)
    print("  Layout Comparison: BLOCKFIRST vs LAYERFIRST (contiguous blocks)")
    print("=" * 120)
    print(f"  {'Size':<8} {'MLA':<5} {'Blocks':<8} {'Dir':<10} {'BF(ms)':>10} {'LF-opt(ms)':>12} {'BF/LF':>8}")
    print("  " + "-" * 70)

    for (size, mla, nblocks, engine, direction, threads), rs in sorted(groups.items()):
        bf = next((r for r in rs if r.layout == "BLOCKFIRST" and r.block_order == "contiguous"), None)
        lf = next((r for r in rs if r.layout == "LAYERFIRST" and r.block_order == "contiguous"), None)
        if not bf or not lf:
            continue
        ratio = lf.time_ms / bf.time_ms if bf.time_ms > 0 else 0
        print(f"  {size:<8} {mla:<5} {nblocks:<8} {direction:<10} {bf.time_ms:>10.2f} {lf.time_ms:>12.2f} {ratio:>7.2f}x")
    print("=" * 120)


def main(args):
    try:
        from flexkv.c_ext import SSDIOCTX, transfer_kv_blocks_ssd
    except ImportError as e:
        print(f"ERROR: c_ext not built: {e}")
        return

    presets = build_presets(args.sizes, args.models)
    layouts = [KVCacheLayoutType(l.upper()) for l in args.layouts]
    block_orders = args.block_orders
    engines = args.engines.split(",")
    threads = args.threads

    all_results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="flexkv_ssd_xfer_bench_"))
    print(f">> Temp dir: {tmpdir}")
    print(f">> Sizes: {args.sizes} | Models: {args.models} | Layouts: {[l.value for l in layouts]}")
    print(f">> Block orders: {block_orders} | Engines: {engines} | Threads: {threads}")
    print(f">> Blocks: {args.blocks} | Rounds: {args.rounds} (warmup: {args.warmup})")
    print()

    try:
        for name, preset in presets.items():
            size_label = name.rsplit("-", 1)[0]
            for num_blocks in args.blocks:
                ssd_path = tmpdir / f"ssd_{name}_{num_blocks}.bin"
                for engine in engines:
                    for layout_type in layouts:
                        for block_order in block_orders:
                            # Skip scattered for BLOCKFIRST (BF always uses single I/O per block)
                            if layout_type.value == "BLOCKFIRST" and block_order == "scattered":
                                continue
                            print(f">> {name} blocks={num_blocks} engine={engine} layout={layout_type.value} order={block_order} ...")
                            rs = benchmark_config(
                                preset, size_label, layout_type, num_blocks, block_order,
                                args.rounds, args.warmup, threads, engine, ssd_path,
                                SSDIOCTX, transfer_kv_blocks_ssd,
                            )
                            all_results.extend(rs)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print_results(all_results)


def parse_args():
    parser = argparse.ArgumentParser(description="D2H/H2D SSD transfer microbenchmark")
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large"],
                        choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--models", nargs="+", default=["MHA", "MLA"],
                        choices=["MHA", "MLA"])
    parser.add_argument("--layouts", nargs="+", default=["BLOCKFIRST", "LAYERFIRST"],
                        choices=["blockfirst", "layerfirst", "BLOCKFIRST", "LAYERFIRST"])
    parser.add_argument("--block-orders", nargs="+", default=["contiguous", "scattered"],
                        choices=["contiguous", "scattered"],
                        help="contiguous triggers layer-major optimization; scattered simulates original fragmented I/O")
    parser.add_argument("--blocks", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--engines", type=str, default="fallback",
                        help="Comma-separated: fallback,iouring")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
