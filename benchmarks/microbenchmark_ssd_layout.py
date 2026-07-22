#!/usr/bin/env python3
"""SSD I/O layout comparison: BLOCKFIRST vs LAYERFIRST.

Measures raw SSD↔CPU transfer throughput for both CPU buffer layouts,
exercising the same transfer_kv_blocks_ssd code path used in production.

Key difference:
  BLOCKFIRST → enable_block_first_transfer=true → single large pread/pwrite per block
  LAYERFIRST → enable_block_first_transfer=false → per-layer pread/pwrite (fragmented I/O)
"""
import argparse
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


# ── Model presets ────────────────────────────────────────────────────────
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
        """Per-K-or-V chunk in elements."""
        return TOKENS_PER_BLOCK * self.num_kv_heads * self.head_size

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_size * self.elem_size


TOKENS_PER_BLOCK = 16

PRESETS: Dict[str, ModelPreset] = {
    "MHA": ModelPreset("MHA", num_layers=80, num_kv_heads=8, head_size=128, is_mla=False),
    "MLA": ModelPreset("MLA", num_layers=80, num_kv_heads=1, head_size=512, is_mla=True),
}


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
    """Compute byte-level strides matching KVCacheLayout methods."""
    chunk_bytes = preset.chunk_bytes
    block_stride_elems = layout.get_block_stride()
    layer_stride_elems = layout.get_layer_stride()
    kv_stride_elems = layout.get_kv_stride()
    return {
        "chunk_bytes": chunk_bytes,
        "block_stride_bytes": block_stride_elems * preset.elem_size,
        "layer_stride_bytes": layer_stride_elems * preset.elem_size,
        "kv_stride_bytes": kv_stride_elems * preset.elem_size,
        "total_bytes": layout.get_total_elements() * preset.elem_size,
    }


# ── Benchmark ────────────────────────────────────────────────────────────
@dataclass
class BenchResult:
    layout: str
    model: str
    num_blocks: int
    num_layers: int
    threads: int
    io_engine: str
    direction: str
    time_ms: float
    bandwidth_gbs: float
    io_pattern: str
    data_bytes: int


def run_single_transfer(
    ioctx,
    transfer_fn,
    cpu_tensor: torch.Tensor,
    layout: KVCacheLayout,
    preset: ModelPreset,
    strides: dict,
    ssd_block_ids: np.ndarray,
    cpu_block_ids: np.ndarray,
    is_read: bool,
    num_threads: int,
) -> float:
    """Execute one transfer_kv_blocks_ssd call, return elapsed ms."""
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
    # io_uring path submits+waits inside; fallback thread path joins inside.
    # But if io_uring is enabled, ensure completion.
    try:
        ioctx.get_iouring().wait_completion()
    except Exception:
        pass
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed


def benchmark_layout(
    layout_type: KVCacheLayoutType,
    preset: ModelPreset,
    num_blocks: int,
    rounds: int,
    warmup: int,
    threads_list: List[int],
    io_engine: str,
    ssd_path: Path,
    SSDIOCTX,
    transfer_fn,
) -> List[BenchResult]:
    layout = make_layout(layout_type, preset, num_blocks)
    strides = compute_strides(layout, preset)
    total_bytes = strides["total_bytes"]

    # CPU buffer: use mmap for page-aligned memory (required by O_DIRECT)
    import mmap
    mm = mmap.mmap(-1, total_bytes, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    cpu_tensor = torch.frombuffer(mm, dtype=preset.dtype)
    # Fill with random pattern for write tests
    rng = np.random.default_rng(42)
    pattern = rng.integers(0, 255, size=total_bytes, dtype=np.uint8)
    mm.write(pattern.tobytes())
    mm.seek(0)

    # SSD file: pre-allocate with fallocate for O_DIRECT compatibility
    with open(ssd_path, "wb") as f:
        f.truncate(total_bytes)
        os.fsync(f.fileno())
    # Also use fallocate to ensure physical allocation
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    FALLOC_FL_KEEP_SIZE = 0x01
    fd = os.open(str(ssd_path), os.O_RDWR)
    libc.fallocate(fd, 0, 0, total_bytes)
    os.close(fd)

    # io_uring entries: 0 = disabled (fallback pread/pwrite), >0 = enabled
    iouring_entries = 512 if io_engine == "iouring" else 0
    ioctx = SSDIOCTX({0: [str(ssd_path)]}, 1, iouring_entries, 0)

    block_ids = np.arange(num_blocks, dtype=np.int64)
    results = []

    # Determine I/O pattern
    is_block_first = strides["block_stride_bytes"] > strides["layer_stride_bytes"]
    io_pattern = "block-first (single large I/O per block)" if is_block_first else "layer-first (per-layer fragmented I/O)"

    data_bytes = total_bytes  # both layouts transfer the same total KV data

    for direction, is_read in [("SSD→CPU", True), ("CPU→SSD", False)]:
        for num_threads in threads_list:
            times = []
            for r in range(warmup + rounds):
                t = run_single_transfer(
                    ioctx, transfer_fn, cpu_tensor, layout, preset, strides,
                    block_ids, block_ids, is_read, num_threads,
                )
                if r >= warmup:
                    times.append(t)
            med = statistics.median(times)
            bw = data_bytes / (med / 1000) / (1024**3) if med > 0 else 0
            results.append(BenchResult(
                layout=layout_type.value,
                model=preset.name,
                num_blocks=num_blocks,
                num_layers=preset.num_layers,
                threads=num_threads,
                io_engine=io_engine,
                direction=direction,
                time_ms=med,
                bandwidth_gbs=bw,
                io_pattern=io_pattern,
                data_bytes=data_bytes,
            ))

    mm.close()
    return results


def print_results(results: List[BenchResult]):
    # Group by (model, num_blocks, io_engine, direction) and compare layouts
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r.model, r.num_blocks, r.io_engine, r.direction, r.threads)
        groups[key].append(r)

    print()
    print("=" * 110)
    print("  SSD layout comparison: BLOCKFIRST vs LAYERFIRST")
    print("=" * 110)

    for (model, nblocks, engine, direction, threads), rs in sorted(groups.items()):
        bf = next((r for r in rs if r.layout == "BLOCKFIRST"), None)
        lf = next((r for r in rs if r.layout == "LAYERFIRST"), None)
        if not bf or not lf:
            continue
        speedup = lf.time_ms / bf.time_ms if bf.time_ms > 0 else 0
        winner = "BF" if bf.time_ms < lf.time_ms else "LF"
        data_gb = bf.data_bytes / (1024**3)
        print(f"  {model} | blocks={nblocks} ({data_gb:.2f}GB) | {engine} | {direction} | threads={threads}")
        print(f"    {'BF':>12}  {bf.time_ms:>10.2f}ms  {bf.bandwidth_gbs:>8.2f} GB/s  | {bf.io_pattern}")
        print(f"    {'LF':>12}  {lf.time_ms:>10.2f}ms  {lf.bandwidth_gbs:>8.2f} GB/s  | {lf.io_pattern}")
        print(f"    {'speedup':>12}  BF/LF = {speedup:.2f}x  [{winner} wins]")
        print()

    # Summary table
    print("=" * 110)
    print("  Summary: BF vs LF speedup (BF time / LF time, >1.0 = BF faster)")
    print("=" * 110)
    print(f"  {'Model':<8} {'Blocks':<8} {'Engine':<10} {'Direction':<10} {'Threads':<8} {'BF/LF':>8} {'Winner':>8}")
    print("  " + "-" * 70)
    for (model, nblocks, engine, direction, threads), rs in sorted(groups.items()):
        bf = next((r for r in rs if r.layout == "BLOCKFIRST"), None)
        lf = next((r for r in rs if r.layout == "LAYERFIRST"), None)
        if not bf or not lf:
            continue
        speedup = lf.time_ms / bf.time_ms if bf.time_ms > 0 else 0
        winner = "BF" if bf.time_ms < lf.time_ms else "LF"
        print(f"  {model:<8} {nblocks:<8} {engine:<10} {direction:<10} {threads:<8} {speedup:>7.2f}x {winner:>8}")
    print("=" * 110)


def main(args):
    try:
        from flexkv.c_ext import SSDIOCTX, transfer_kv_blocks_ssd
    except ImportError as e:
        print(f"ERROR: c_ext not built or SSD support disabled: {e}")
        return

    presets_to_test = [PRESETS[name] for name in args.models]
    layouts_to_test = [KVCacheLayoutType(l.upper()) for l in args.layouts]
    threads_list = sorted(set(args.threads))
    engines = args.engines.split(",") if args.engines else ["fallback"]

    all_results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="flexkv_ssd_bench_"))
    print(f">> SSD benchmark temp dir: {tmpdir}")
    print(f">> Models: {[p.name for p in presets_to_test]}")
    print(f">> Layouts: {[l.value for l in layouts_to_test]}")
    print(f">> Threads: {threads_list}")
    print(f">> Engines: {engines}")
    print(f">> Block counts: {args.blocks}")
    print()

    try:
        for preset in presets_to_test:
            for num_blocks in args.blocks:
                ssd_path = tmpdir / f"ssd_{preset.name}_{num_blocks}.bin"
                for engine in engines:
                    for layout_type in layouts_to_test:
                        print(f">> Running {preset.name} blocks={num_blocks} engine={engine} layout={layout_type.value} ...")
                        rs = benchmark_layout(
                            layout_type, preset, num_blocks,
                            args.rounds, args.warmup,
                            threads_list, engine,
                            ssd_path, SSDIOCTX, transfer_kv_blocks_ssd,
                        )
                        all_results.extend(rs)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print_results(all_results)


def parse_args():
    parser = argparse.ArgumentParser(description="SSD I/O layout benchmark: BLOCKFIRST vs LAYERFIRST")
    parser.add_argument("--models", nargs="+", default=["MHA", "MLA"],
                        choices=list(PRESETS.keys()),
                        help="Model presets to test")
    parser.add_argument("--layouts", nargs="+", default=["BLOCKFIRST", "LAYERFIRST"],
                        choices=["blockfirst", "layerfirst", "BLOCKFIRST", "LAYERFIRST"],
                        help="CPU/SSD layouts to test")
    parser.add_argument("--blocks", nargs="+", type=int, default=[64, 256, 1024, 4096],
                        help="Number of blocks to transfer per round")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4, 16],
                        help="Threads per device")
    parser.add_argument("--engines", type=str, default="fallback",
                        help="Comma-separated: fallback,iouring")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Measurement rounds (median reported)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup rounds (not counted)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
