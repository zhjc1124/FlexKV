"""
Microbenchmark: MLA D2H transfer modes — end-to-end round-trip, REAL FlexKV API.

Measures the full offload+reload path (D2H then H2D = one round-trip) across
the 4 transfer strategies, and — for the CE engine — proves the optimization
progression baseline -> optimized.

Compares 4 transfer strategies:
  - MHA (non-MLA):    each TP rank owns different head partition (cpu_tp_stride)
  - MLA sharded:     each GPU writes 1/N shard, H2D all read full KV
  - MLA all_write:   each GPU writes full KV, H2D each reads own copy
  - MLA rank0_only:  only rank0 writes, H2D all read from rank0's slot

Matrix:
  - Strategy:  MHA / sharded / all_write / rank0_only  (4 types)
  - Engine:    CUDA kernel / CE  (probed, unsupported skipped)
  - CE config: baseline / opt  (CE only; kernel ignores path_opt)
  - H2D reload engine: TP-group / LayerwiseTransferGroup  (CE only, --no-layerwise)
  - Layout:    LAYERFIRST / BLOCKFIRST  (CPU side)
  - Size:      small / medium / large  (each gets its own summary)

path_opt / segment_threshold are passed per-construction to the
group ctor (NOT via env), matching production and the correctness tests.
Uses KVCacheLayout for stride computation (same as production worker.py).

Usage:
    python benchmarks/microbenchmark_mla_d2h_modes.py --num-gpus 4 --iters 10
    python benchmarks/microbenchmark_mla_d2h_modes.py --sizes small large
    python benchmarks/microbenchmark_mla_d2h_modes.py --no-ce --no-layerwise
"""

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# GPU / FlexKV detection
# ---------------------------------------------------------------------------
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    CUDA_AVAILABLE = False
    NUM_GPUS = 0
    print("ERROR: PyTorch not available")
    sys.exit(1)

try:
    from flexkv.c_ext import TPTransferThreadGroup, LayerwiseTransferGroup
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    FLEXKV_AVAILABLE = True
except ImportError as e:
    FLEXKV_AVAILABLE = False
    print("ERROR: FlexKV not available ({})".format(e))
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

DTYPE = torch.float16
ES = DTYPE.itemsize

# Model-representative configs: (num_layers, num_blocks, head_dim)
#   num_blocks = max_batch * max_seq_len / tokens_per_block (tpb=1 for MLA)
#   head_dim   = MLA latent_dim (the per-head KV size)
#
# MLA models (DeepSeek-V2/V3, Kimi-K2, etc.):
#   DeepSeek-V3:  61 layers, kv_heads=1, latent_dim=512, fp8/bf16
#   Kimi-K2:      61 layers, kv_heads=1, latent_dim=512, bf16
# MHA models (Llama-3, Qwen2, etc.):
#   Llama-3-8B:   32 layers, kv_heads=8, head_dim=128, bf16
#   Llama-3-70B:  80 layers, kv_heads=8, head_dim=128, bf16
#   Qwen2-72B:    80 layers, kv_heads=8, head_dim=128, bf16
#
# For MHA, num_heads is set to num_gpus at runtime (heads_per_rank=1).
SIZES = {
    # Small: ~Llama-3-8B scale
    "small":  (32,   512,  128),
    # Medium: ~DeepSeek-V3 / Kimi-K2 (MLA) or Llama-3-70B (MHA)
    "medium": (61,  2048,  512),
    # Large: 80-layer model with long context
    "large":  (80,  8192,  512),
}

LAYOUTS = {
    "lfirst": KVCacheLayoutType.LAYERFIRST,
    "bfirst": KVCacheLayoutType.BLOCKFIRST,
}

# 4 strategies. MLA modes use is_mla=True; MHA uses is_mla=False.
STRATEGIES = [
    ("MHA",         False, "sharded"),     # non-MLA, mode ignored
    ("MLA-sharded", True,  "sharded"),
    ("MLA-all_write", True, "all_write"),
    ("MLA-rank0_only", True, "rank0_only"),
]

# CE optimization config, as two cumulative levels. Only meaningful for the
# CE engine (the CUDA kernel engine ignores path_opt). Proves the
# progression baseline -> optimized.
#   (label, path_opt)
CE_CONFIGS = [
    ("baseline",   False),   # PER_BLOCK, no optimization
    ("opt",        True),    # optimized strategies
]

WARMUP_ITERS = 3


# ---------------------------------------------------------------------------
# Engine probe
# ---------------------------------------------------------------------------

_probe_cache = {}

def probe_engine(use_ce):
    key = use_ce
    if key in _probe_cache:
        return _probe_cache[key]
    try:
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=1, num_block=1, tokens_per_block=1,
            num_head=1, head_size=16, is_mla=True)
        g = torch.zeros((1, 1, 1, 1, 1, 16), dtype=DTYPE, device="cuda:0")
        c = torch.zeros(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)
        ids = torch.arange(1, dtype=torch.int64).pin_memory()
        tp = TPTransferThreadGroup(
            num_gpus=1, gpu_block_ptrs_flat=[g[0].data_ptr()],
            num_tensors_per_gpu=1, cpu_blocks_ptr=c.data_ptr(), num_layers=1,
            gpu_kv_strides_in_bytes=[layout.get_kv_stride() * ES],
            gpu_block_strides_in_bytes=[layout.get_block_stride() * ES],
            gpu_layer_strides_in_bytes=[layout.get_layer_stride() * ES],
            gpu_chunk_sizes_in_bytes=[layout.get_chunk_size() * ES],
            gpu_device_ids=[0], enable_nvcomp=False)
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=layout.get_kv_stride() * ES,
            cpu_layer_stride_in_bytes=layout.get_layer_stride() * ES,
            cpu_block_stride_in_bytes=layout.get_block_stride() * ES,
            cpu_tp_stride_in_bytes=layout.get_block_stride() * ES,
            transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
            layer_id=0, layer_granularity=1, is_mla=True, mla_d2h_mode="sharded")
        torch.cuda.synchronize()
        del tp
        _probe_cache[key] = True
        return True
    except Exception:
        _probe_cache[key] = False
        return False


# ---------------------------------------------------------------------------
# Helpers (matching production worker.py)
# ---------------------------------------------------------------------------

def make_layouts(num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus):
    """Create GPU (LAYERFIRST) and CPU layouts.

    MLA: kv_dim=1, head=1 (all ranks identical).
    MHA: kv_dim=2, head=num_gpus (each rank gets 1 head).
    GPU uses heads_per_rank (per-rank), CPU uses full num_head.
    """
    num_head = 1 if is_mla else num_gpus
    heads_per_rank = 1  # MLA: 1 shared head; MHA: num_gpus/num_gpus = 1
    # GPU layout uses per-rank heads (tensor only has heads_per_rank)
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=heads_per_rank,
        head_size=head_dim, is_mla=is_mla)
    # CPU layout uses full heads (all ranks' data on one buffer)
    cpu_layout = KVCacheLayout(
        type=cpu_layout_type,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return gpu_layout, cpu_layout


def cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks, head_dim,
                             is_mla, mode, num_gpus):
    """Return (cpu_kv_sb, cpu_layer_sb, cpu_block_sb, cpu_tp_sb, total_blocks).

    For MLA all_write: CPU holds N copies, total = num_blocks * num_gpus.
    For MHA: div_head(tp_size) on BLOCKFIRST for per-rank CPU strides.
    """
    total = num_blocks * num_gpus if (is_mla and mode == "all_write") else num_blocks
    num_head = 1 if is_mla else num_gpus

    layout_for_kv_stride = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)

    # For non-MLA BLOCKFIRST, div_head to get per-rank strides
    if not is_mla and cpu_layout.type == KVCacheLayoutType.BLOCKFIRST:
        layout_for_kv_stride = layout_for_kv_stride.div_head(num_gpus)

    kv_sb = layout_for_kv_stride.get_kv_stride() * ES
    layer_sb = layout_for_kv_stride.get_layer_stride() * ES
    block_sb = cpu_layout.get_block_stride() * ES
    tp_sb = block_sb // num_gpus
    return kv_sb, layer_sb, block_sb, tp_sb, total


def make_gpu_tensors(num_layers, kv_dim, num_blocks, heads_per_rank, head_dim, device):
    """Contiguous [num_layers, kv_dim, num_blocks, 1, heads_per_rank, head_dim]."""
    full = torch.empty(
        (num_layers, kv_dim, num_blocks, 1, heads_per_rank, head_dim),
        dtype=DTYPE, device="cuda:{}".format(device))
    return [full[i] for i in range(num_layers)]


def make_cpu_tensor(cpu_layout, num_layers, total_blocks, head_dim, is_mla, num_gpus):
    num_head = 1 if is_mla else num_gpus
    layout = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return torch.empty(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)


def fill_gpu(all_gpu, gpu_id, num_layers, num_blocks, head_dim):
    for layer in range(num_layers):
        torch.manual_seed(gpu_id * 10000 + layer)
        all_gpu[gpu_id][layer].uniform_()


def block_ids(n):
    return torch.arange(n, dtype=torch.int64).pin_memory()


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  ce_path_opt=True,
                  ce_segment_threshold=8,
                  ce_is_mla=False, ce_is_blockfirst=False):
    """TPTransferThreadGroup with CE config passed per-construction.

    path_opt / segment_threshold go into the C++ CETransferConfig
    via ctor args (NOT env) -- matching production and the correctness tests.
    """
    gpu_ptrs = []
    for g in range(num_gpus):
        for l in range(num_layers):
            gpu_ptrs.append(all_gpu[g][l].data_ptr())
    return TPTransferThreadGroup(
        num_gpus=num_gpus, gpu_block_ptrs_flat=gpu_ptrs,
        num_tensors_per_gpu=num_layers, cpu_blocks_ptr=cpu_ptr,
        num_layers=num_layers,
        gpu_kv_strides_in_bytes=[gpu_layout.get_kv_stride() * ES] * num_gpus,
        gpu_block_strides_in_bytes=[gpu_layout.get_block_stride() * ES] * num_gpus,
        gpu_layer_strides_in_bytes=[gpu_layout.get_layer_stride() * ES] * num_gpus,
        gpu_chunk_sizes_in_bytes=[gpu_layout.get_chunk_size() * ES] * num_gpus,
        gpu_device_ids=list(range(num_gpus)),
        enable_nvcomp=False,
        ce_segment_threshold=ce_segment_threshold,
        ce_path_opt=ce_path_opt,
        ce_is_mla=ce_is_mla,
        ce_is_blockfirst=ce_is_blockfirst)


def make_layerwise_group(cpu_kv_tensor, all_gpu, num_gpus, gpu_layout,
                         num_layers, ce_path_opt=True,
                         ce_segment_threshold=8,
                         ce_is_mla=False, ce_is_blockfirst=False):
    """LayerwiseTransferGroup (H2D reload engine) with CE config per-ctor.

    Mirrors LayerwiseTransferWorker construction; SSD/eventfd/indexer disabled.
    Every trailing optional ctor arg is passed explicitly so pybind11 never has
    to synthesize a torch::Tensor() default (see pybind11-construct-debug).
    """
    def strides_tensor(getter):
        return torch.tensor([getter() * ES] * num_gpus, dtype=torch.int64)
    empty_tensor = torch.Tensor()
    return LayerwiseTransferGroup(
        num_gpus=num_gpus,
        gpu_blocks=all_gpu,
        cpu_blocks=cpu_kv_tensor,
        ssd_files={},
        num_layers=num_layers,
        gpu_kv_strides_tensor=strides_tensor(gpu_layout.get_kv_stride),
        gpu_block_strides_tensor=strides_tensor(gpu_layout.get_block_stride),
        gpu_layer_strides_tensor=strides_tensor(gpu_layout.get_layer_stride),
        gpu_chunk_sizes_tensor=strides_tensor(gpu_layout.get_chunk_size),
        iouring_entries=0,
        iouring_flags=0,
        layer_eventfds_tensor=torch.empty(0, dtype=torch.int32),
        tp_size=num_gpus,
        indexer_gpu_blocks=[],
        indexer_cpu_blocks=empty_tensor,
        indexer_gpu_kv_strides_tensor=empty_tensor,
        indexer_gpu_block_strides_tensor=empty_tensor,
        indexer_gpu_layer_strides_tensor=empty_tensor,
        indexer_gpu_chunk_sizes_tensor=empty_tensor,
        indexer_ssd_files={},
        ce_segment_threshold=ce_segment_threshold,
        ce_path_opt=ce_path_opt,
        ce_is_mla=ce_is_mla,
        ce_is_blockfirst=ce_is_blockfirst)


def layerwise_h2d(lw_group, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                  chunk_size, num_layers, is_mla, mode, use_ce=True):
    """Issue one H2D through LayerwiseTransferGroup (production reload path).

    use_ce controls whether the internal transfer_kv_blocks uses CE
    (cudaMemcpyAsync) or the CUDA kernel path. LayerwiseTransferGroup supports
    both — the CE engine benefits from path_opt staging; the kernel
    engine is the same grid-stride loop as TP-group H2D but driven per-layer
    by the layerwise pipeline.
    Every trailing optional param is passed explicitly (pybind11-construct-debug).
    """
    empty_ids = torch.empty(0, dtype=torch.int64).pin_memory()
    empty_indexer = torch.Tensor()
    lw_group.layerwise_transfer(
        ssd_block_ids=empty_ids,
        cpu_block_ids_d2h=empty_ids,
        ssd_layer_stride_in_bytes=0,
        ssd_kv_stride_in_bytes=0,
        num_blocks_per_file=0, round_robin=0, num_threads_per_device=0,
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_kv_sb,
        cpu_layer_stride_in_bytes=cpu_ly_sb,
        cpu_block_stride_in_bytes=cpu_bl_sb,
        cpu_chunk_size_in_bytes=chunk_size,
        h2d_cpu_kv_stride_in_bytes=cpu_kv_sb,
        h2d_cpu_layer_stride_in_bytes=cpu_ly_sb,
        cpu_tp_stride_in_bytes=cpu_tp_sb,
        transfer_cta_num=4, use_ce_transfer=use_ce,
        num_layers=num_layers, layer_granularity=num_layers,
        is_mla=is_mla,
        counter_id=0,
        indexer_gpu_block_id_tensor=empty_indexer,
        indexer_cpu_block_id_tensor=empty_indexer,
        indexer_cpu_block_stride_in_bytes=0,
        indexer_cpu_layer_stride_in_bytes=0,
        indexer_h2d_cpu_kv_stride_in_bytes=0,
        indexer_h2d_cpu_layer_stride_in_bytes=0,
        indexer_ssd_block_ids=empty_indexer,
        indexer_cpu_block_ids_d2h=empty_indexer,
        indexer_ssd_layer_stride_in_bytes=0,
        indexer_ssd_kv_stride_in_bytes=0,
        indexer_cpu_chunk_size_in_bytes=0,
        indexer_num_blocks_per_file=0,
        mla_d2h_mode=mode)


def sync_all(num_gpus):
    for g in range(num_gpus):
        torch.cuda.synchronize(g)


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def bench_one(strategy_label, is_mla, mode, use_ce, cpu_layout_type,
              num_gpus, num_layers, num_blocks, head_dim, iters,
              ce_path_opt=True, ce_segment_threshold=8,
              h2d_engine="tp"):
    """Run one benchmark configuration: full D2H+H2D round-trip.

    Measures the entire offload+reload path as a single timing:
      D2H (GPU->CPU, always TP-group) + H2D (CPU->GPU) = one round-trip.
    The H2D leg uses either the TP-group (h2d_engine="tp") or the production
    LayerwiseTransferGroup reload engine (h2d_engine="layerwise", CE-only).

    ce_path_opt / ce_segment_threshold are passed per-ctor to
    both groups (they only affect the CE engine). Returns round-trip timing.
    """
    kv_dim = 1 if is_mla else 2
    heads_per_rank = 1  # MLA: 1 head shared; MHA: num_gpus heads / num_gpus = 1 per rank

    gpu_layout, cpu_layout = make_layouts(
        num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus)
    cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb, total_blocks = \
        cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks, head_dim,
                                  is_mla, mode, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, kv_dim, num_blocks, heads_per_rank,
                                head_dim, g) for g in range(num_gpus)]
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks, head_dim,
                             is_mla, num_gpus)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                       num_layers, ce_path_opt=ce_path_opt,
                       ce_segment_threshold=ce_segment_threshold,
                       ce_is_mla=is_mla,
                       ce_is_blockfirst=(cpu_layout_type == KVCacheLayoutType.BLOCKFIRST))

    use_layerwise = (h2d_engine == "layerwise")
    lw = None
    chunk_size = gpu_layout.get_chunk_size() * ES
    if use_layerwise:
        lw = make_layerwise_group(cpu_kv, all_gpu, num_gpus, gpu_layout,
                                  num_layers, ce_path_opt=ce_path_opt,
                                  ce_segment_threshold=ce_segment_threshold,
                                  ce_is_mla=is_mla,
                                  ce_is_blockfirst=(cpu_layout_type == KVCacheLayoutType.BLOCKFIRST))

    gpu_ids = block_ids(num_blocks)
    cpu_ids = block_ids(num_blocks)

    def do_d2h():
        tp.tp_group_transfer(
            gpu_block_id_tensor=gpu_ids, cpu_block_id_tensor=cpu_ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
            layer_id=0, layer_granularity=num_layers, is_mla=is_mla, mla_d2h_mode=mode)

    def do_h2d():
        if use_layerwise:
            layerwise_h2d(lw, gpu_ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb,
                          cpu_tp_sb, chunk_size, num_layers, is_mla, mode,
                          use_ce=use_ce)
        else:
            tp.tp_group_transfer(
                gpu_block_id_tensor=gpu_ids, cpu_block_id_tensor=cpu_ids,
                cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
                cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
                transfer_num_cta=4, is_host_to_device=True, use_ce_transfer=use_ce,
                layer_id=0, layer_granularity=num_layers, is_mla=is_mla, mla_d2h_mode=mode)

    # Warmup: full D2H + H2D round-trip
    for _ in range(WARMUP_ITERS):
        for g in range(num_gpus):
            fill_gpu(all_gpu, g, num_layers, num_blocks, head_dim)
        sync_all(num_gpus)
        do_d2h()
        for g in range(num_gpus):
            for l in range(num_layers):
                all_gpu[g][l].zero_()
        sync_all(num_gpus)
        do_h2d()
        sync_all(num_gpus)

    # Timing: full D2H + H2D round-trip per iteration
    times = []
    for _ in range(iters):
        for g in range(num_gpus):
            fill_gpu(all_gpu, g, num_layers, num_blocks, head_dim)
        sync_all(num_gpus)
        t0 = time.perf_counter()
        do_d2h()
        for g in range(num_gpus):
            for l in range(num_layers):
                all_gpu[g][l].zero_()
        sync_all(num_gpus)
        do_h2d()
        sync_all(num_gpus)
        times.append((time.perf_counter() - t0) * 1000)

    del tp
    if lw is not None:
        del lw
    return {
        "avg_ms": float(np.mean(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _col(text, width):
    return str(text).ljust(width)


def print_results_table(results):
    """Per (size, h2d_engine) block: list all strategy x layout x engine x config rows.

    The two-dimensional primary axis is (size, h2d_engine); within each block,
    every combination of strategy x layout x engine x config is listed so the
    full performance landscape under that size + reload engine is visible.
    """
    # Block key: (size, h2d_engine)
    blocks = defaultdict(list)
    for r in results:
        blocks[(r["size"], r.get("h2d_engine", "tp"))].append(r)

    print("")
    for (size, h2d) in sorted(blocks.keys()):
        rows = blocks[(size, h2d)]
        print("=" * 96)
        print("  Block: size={} | h2d={}  ({} rows)".format(size, h2d, len(rows)))
        print("=" * 96)
        w = [12, 7, 16, 7, 12, 9, 9, 9]
        hdr = (_col("Layout", w[1]) + _col("Strategy", w[2])
               + _col("Engine", w[3]) + _col("Config", w[4])
               + _col("Avg ms", w[5]) + _col("P99 ms", w[6]) + _col("Min ms", w[7]))
        sep = "-" * len(hdr)
        print("  " + hdr)
        print("  " + sep)
        # Sort: layout, strategy, engine (CUDA before CE), config (baseline/opt/n/a)
        cfg_order = {"baseline": 0, "opt": 1, "n/a": 2}
        rows_sorted = sorted(rows, key=lambda r: (
            r["layout"], r["strategy"], 0 if r["engine"] == "CUDA" else 1,
            cfg_order.get(r.get("config", "n/a"), 9)))
        for r in rows_sorted:
            line = (_col(r["layout"], w[1]) + _col(r["strategy"], w[2])
                    + _col(r["engine"], w[3]) + _col(r.get("config", "-"), w[4])
                    + _col("{:.3f}".format(r["avg_ms"]), w[5])
                    + _col("{:.3f}".format(r["p99_ms"]), w[6])
                    + _col("{:.3f}".format(r["min_ms"]), w[7]))
            print("  " + line)
        print("  " + sep)
        print("")


def print_analysis(results):
    """Prove the CE optimization progression baseline -> opt.

    Organized as (size, h2d_engine) two-dimensional blocks: within each block,
    every (layout, strategy) combination shows baseline / opt
    side by side with speedup vs baseline and the fastest marked. A global
    verdict tallies how often opt wins across all blocks.
    """
    print("\n" + "=" * 96)
    print("CE optimization analysis (D2H+H2D round-trip): baseline vs opt")
    print("  two-dimensional primary axis: (size, h2d_engine)")
    print("=" * 96)

    ce_results = [r for r in results if r["engine"] == "CE"]
    if not ce_results:
        print("  (no CE results — nothing to analyze)")
        return

    # group key: (size, h2d, layout, strategy)
    def gkey(r):
        return (r["size"], r.get("h2d_engine", "tp"), r["layout"], r["strategy"])

    groups = defaultdict(dict)
    for r in ce_results:
        groups[gkey(r)][r.get("config", "-")] = r["avg_ms"]

    opt_wins = 0
    opt_total = 0

    # Outer loop: (size, h2d) blocks
    block_keys = sorted(set((k[0], k[1]) for k in groups.keys()))
    for (size, h2d) in block_keys:
        print("\n  === size={} | h2d={} ===".format(size, h2d))
        print("  {:<7} {:<16} {:>10} {:>10}  {}".format(
            "layout", "strategy", "baseline", "opt",
            "opt vs base"))
        print("  " + "-" * 68)
        # Inner: (layout, strategy) rows within this block
        block_groups = {k: v for k, v in groups.items()
                        if k[0] == size and k[1] == h2d}
        for key in sorted(block_groups.keys()):
            _, _, layout, strat = key
            cfgs = groups[key]
            base = cfgs.get("baseline")
            opt = cfgs.get("opt")
            fastest = min((v for v in (base, opt) if v is not None),
                          default=None)

            def fmt(v):
                if v is None:
                    return "{:>10}".format("-")
                star = "*" if (v == fastest) else " "
                return "{:>9.3f}{}".format(v, star)

            speedup = ""
            if base and opt:
                speedup = "{:.2f}x".format(base / opt)
                opt_total += 1
                if opt == fastest:
                    opt_wins += 1

            print("  {:<7} {:<16} {} {}  {:>10}".format(
                layout, strat, fmt(base), fmt(opt), speedup))

    # --- Global verdict ---
    print("\n  " + "-" * 68)
    if opt_total:
        print("  opt was the fastest config in {}/{} CE groups ({:.0f}%).".format(
            opt_wins, opt_total, 100.0 * opt_wins / opt_total))
        if opt_wins == opt_total:
            print("  => optimized paths are uniformly the best CE config.")
        else:
            print("  => optimized paths win in most cases; inspect the "
                  "groups above where it does not (usually tiny transfers where "
                  "staging overhead dominates).")

    # --- Fastest overall per (size, strategy), across layout+config+h2d ---
    print("\n  Fastest CE config per (size, strategy):")
    print("  " + "-" * 68)
    per = defaultdict(list)
    for r in ce_results:
        per[(r["size"], r["strategy"])].append(r)
    for (size, strat) in sorted(per.keys()):
        best = min(per[(size, strat)], key=lambda r: r["avg_ms"])
        print("    {:<8} {:<16} -> {} / {} / h2d={} : {:.3f} ms".format(
            size, strat, best["layout"], best.get("config", "-"),
            best.get("h2d_engine", "tp"), best["avg_ms"]))

    print("\n  Note: Performance depends on hardware (NUMA topology, PCIe/NVLink")
    print("        bandwidth, GPU model). Always benchmark on your target machine.")
    print("=" * 96)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KV transfer strategies (REAL FlexKV API)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (default: 0 = all available)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timing iterations per config (default: 10)")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=[s[0] for s in STRATEGIES],
                        help="Strategies to test")
    parser.add_argument("--sizes", type=str, nargs="+",
                        default=list(SIZES.keys()),
                        choices=list(SIZES.keys()),
                        help="Data sizes to test (default: all)")
    parser.add_argument("--layouts", type=str, nargs="+",
                        default=list(LAYOUTS.keys()),
                        choices=list(LAYOUTS.keys()),
                        help="CPU layouts to test (default: both)")
    parser.add_argument("--no-kernel", action="store_true",
                        help="Skip CUDA Kernel tests")
    parser.add_argument("--no-ce", action="store_true",
                        help="Skip CE tests")
    parser.add_argument("--no-layerwise", action="store_true",
                        help="Skip the LayerwiseTransferGroup H2D reload engine "
                             "variant (CE only)")
    parser.add_argument("--segment-threshold", type=int, default=8,
                        help="CE segment_threshold for optimized configs "
                             "(default: 8)")
    args = parser.parse_args()

    num_gpus = NUM_GPUS if args.num_gpus <= 0 else min(args.num_gpus, NUM_GPUS)
    if num_gpus < 2:
        print("ERROR: need at least 2 GPUs, found {}".format(NUM_GPUS))
        sys.exit(1)

    # Engine probe
    engines = []
    if not args.no_kernel:
        if probe_engine(use_ce=False):
            engines.append(("CUDA", False))
        else:
            print("WARNING: CUDA kernel probe failed, skipping kernel tests")
    if not args.no_ce:
        if probe_engine(use_ce=True):
            engines.append(("CE", True))
        else:
            print("WARNING: CE probe failed, skipping CE tests")
    if not engines:
        print("ERROR: no transfer engine available")
        sys.exit(1)

    # Filter strategies
    active_strategies = [s for s in STRATEGIES if s[0] in args.strategies]

    print("=" * 90)
    print("  FlexKV KV Transfer Benchmark")
    print("=" * 90)
    print("  GPUs:        {}".format(num_gpus))
    print("  Strategies:  {}".format([s[0] for s in active_strategies]))
    print("  Sizes:       {}".format(args.sizes))
    print("  Layouts:     {}".format(args.layouts))
    print("  Engines:     {}".format([e[0] for e in engines]))
    print("  Iters:       {}".format(args.iters))
    print("  Dtype:       {}".format(DTYPE))
    print("=" * 90)

    # Two-dimensional primary axis: (size, h2d_engine). cuda also runs
    # layerwise (LayerwiseTransferGroup supports use_ce_transfer=False -> the
    # internal transfer_kv_blocks takes the kernel path, driven per-layer by the
    # layerwise pipeline). This lets us compare tp vs layerwise reload under
    # both CUDA-kernel and CE engines.
    h2d_engines_all = ["tp"] if args.no_layerwise else ["tp", "layerwise"]

    results = []
    for size_name in args.sizes:
        num_layers, num_blocks, head_dim = SIZES[size_name]
        kv_bytes = num_layers * 1 * num_blocks * 1 * 1 * head_dim * ES
        print("\n--- Size: {} ({} layers, {} blocks, hd={}, {:.1f} MB) ---".format(
            size_name, num_layers, num_blocks, head_dim, kv_bytes / (1024**2)))

        for h2d_engine in h2d_engines_all:
            print("\n  >>> h2d_engine = {} <<<".format(h2d_engine))
            for engine_name, use_ce in engines:
                # path_opt only matters for the CE engine. For the
                # CUDA kernel engine, run a single "n/a" config (ignored).
                configs = CE_CONFIGS if use_ce else [("n/a", True)]

                for layout_name in args.layouts:
                    for strat_label, is_mla, mode in active_strategies:
                        cpu_layout_type = LAYOUTS[layout_name]
                        for cfg_label, path_opt in configs:
                            label = "{} | h2d={} | {} | {} | {} | {}".format(
                                size_name, h2d_engine, engine_name, layout_name,
                                strat_label, cfg_label)
                            print("  Running: {} ...".format(label),
                                  end=" ", flush=True)
                            try:
                                r = bench_one(
                                    strat_label, is_mla, mode, use_ce,
                                    cpu_layout_type, num_gpus, num_layers,
                                    num_blocks, head_dim, args.iters,
                                    ce_path_opt=path_opt,
                                    ce_segment_threshold=args.segment_threshold,
                                    h2d_engine=h2d_engine)
                                r.update({
                                    "size": size_name,
                                    "layout": layout_name,
                                    "strategy": strat_label,
                                    "engine": engine_name,
                                    "h2d_engine": h2d_engine,
                                    "config": cfg_label,
                                })
                                results.append(r)
                                print("avg={:.3f}ms".format(r["avg_ms"]))
                            except Exception as e:
                                print("FAILED: {}".format(e))

    print_results_table(results)
    print_analysis(results)


if __name__ == "__main__":
    main()
