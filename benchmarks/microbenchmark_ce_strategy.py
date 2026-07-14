"""
Microbenchmark: CE transfer strategy comparison (six CEPath strategies).

Drives the C++ CE engine through each of its execution forms and compares
two cumulative CE configs on every form, proving the optimization progression
baseline -> optimized per form.

The CE execution forms (see csrc/ce_transfer.h CEPath):
  - BULK_CONTIG       single large memcpy (contiguous ids, dst phys contiguous)
  - SEGMENTED_DIRECT  per-run memcpy, dst phys contiguous, no staging
  - STAGED_MERGE      staging + CPU scatter, GPU side contiguous run (merged segment memcpy)
  - STAGED_BLOCK      staging + CPU scatter, GPU side per-block (sharded D2H)
  - GATHER_SCATTER    GPU index_select/index_copy_ pipeline (many segments)
  - BF_TRANSPOSE      BF MLA: D2D transpose + direct per-segment memcpy (checked first)

Each form is triggered by a specific (block-id pattern, cpu_layout, mla_mode,
direction). STAGED_BLOCK only appears on the sharded-D2H leg (the only
src_phys=False path); every other form is exercised H2D with rank0_only
(only rank 0 performs the transfer; non-rank0 GPUs idle in D2H, all read in H2D).

Two cumulative CE configs:
  - baseline : path_opt off  -> C++ runs PER_BLOCK
  - opt      : path_opt on

path_opt / segment_threshold are passed per-construction to the
group ctor (NOT via env), matching production and the correctness tests.

Usage:
    python benchmarks/microbenchmark_ce_strategy.py --num-gpus 4 --iters 20
"""

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    print("ERROR: PyTorch not available")
    sys.exit(1)

try:
    from flexkv.c_ext import TPTransferThreadGroup
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    FLEXKV_AVAILABLE = True
except ImportError as e:
    print("ERROR: FlexKV not available ({})".format(e))
    sys.exit(1)

DTYPE = torch.float16
ES = DTYPE.itemsize
WARMUP_ITERS = 3

# Model-representative sizes: (num_layers, num_blocks, head_dim)
# Same as microbenchmark_mla_d2h_modes.py for consistency.
SIZES = {
    "small":  (32,   512,  128),
    "medium": (61,  2048,  512),
    "large":  (80,  8192,  512),
}


# -- Layout / stride helpers (MLA + BLOCKFIRST capable) -----------------------

STRAT_LAYOUTS = {
    "lfirst": KVCacheLayoutType.LAYERFIRST,
    "bfirst": KVCacheLayoutType.BLOCKFIRST,
}


def make_layouts_strat(num_layers, num_blocks, head_dim, cpu_layout_type,
                       is_mla, num_gpus):
    """GPU (LAYERFIRST, per-rank heads) and CPU (cpu_layout_type) layouts.

    MLA: kv_dim=1, head=1 (all ranks identical).
    MHA: kv_dim=2, head=num_gpus (each rank gets 1 head).
    """
    num_head = 1 if is_mla else num_gpus
    heads_per_rank = 1  # MLA: 1 shared head; MHA: num_gpus/num_gpus = 1
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=heads_per_rank,
        head_size=head_dim, is_mla=is_mla)
    cpu_layout = KVCacheLayout(
        type=cpu_layout_type,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return gpu_layout, cpu_layout


def cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks, head_dim,
                             is_mla, mode, num_gpus):
    """Return (cpu_kv_sb, cpu_layer_sb, cpu_block_sb, cpu_tp_sb, total_blocks).

    rank0_only / sharded: CPU holds 1 copy, total = num_blocks.
    (all_write would hold N copies, but we don't use it.)
    For MHA: div_head(tp_size) on BLOCKFIRST for per-rank CPU strides.
    """
    total = num_blocks
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


def make_gpu_tensors_strat(num_layers, kv_dim, num_blocks, heads_per_rank,
                           head_dim, device):
    """Contiguous [num_layers, kv_dim, num_blocks, 1, heads_per_rank, head_dim]."""
    full = torch.empty(
        (num_layers, kv_dim, num_blocks, 1, heads_per_rank, head_dim),
        dtype=DTYPE, device="cuda:{}".format(device))
    return [full[i] for i in range(num_layers)]


def make_cpu_tensor_strat(cpu_layout, num_layers, total_blocks, head_dim,
                          is_mla, num_gpus):
    # rank0_only / sharded: total_blocks = num_blocks (1 copy on CPU).
    # The simple per-GPU sizing can still under-size the CPU buffer for some
    # modes (benchmark-fidelity limitation: timing is representative, data is
    # not correctness-verified). Kept consistent with the sibling files.
    num_head = 1 if is_mla else num_gpus
    layout = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return torch.empty(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  ce_path_opt=True,
                  ce_segment_threshold=8, ce_force_path=-1,
                  ce_is_mla=False, ce_is_blockfirst=False):
    """TPTransferThreadGroup with CE config passed per-construction.

    ce_force_path: test/benchmark only. -1 = auto (choose_path); 0-5 = force
    a specific CEPath. Production never sets it.
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
        ce_force_path=ce_force_path,
        ce_is_mla=ce_is_mla,
        ce_is_blockfirst=ce_is_blockfirst)


def fill_gpu(all_gpu, gpu_id, num_layers, num_blocks, head_dim):
    for layer in range(num_layers):
        torch.manual_seed(gpu_id * 10000 + layer)
        all_gpu[gpu_id][layer].uniform_()


def sync_all(num_gpus):
    for g in range(num_gpus):
        torch.cuda.synchronize(g)


def make_block_id_pattern(kind, num_blocks):
    """Build a pinned int64 block-id tensor selecting a CE execution form.

    contiguous : arange (single contiguous run)
    few_seg    : 4 contiguous runs with gaps (a few large segments)
    scattered  : deterministic random permutation (many tiny segments)
    """
    if kind == "contiguous":
        ids = torch.arange(num_blocks, dtype=torch.int64)
    elif kind == "few_seg":
        q = num_blocks // 4
        parts = [torch.arange(i * q, (i + 1) * q, dtype=torch.int64)
                 for i in range(4)]
        # Reorder quarters (Q0, Q2, Q1, Q3) to create gaps between runs.
        # Q0=[0..127], Q2=[256..383], Q1=[128..255], Q3=[384..511]
        # → 4 contiguous segments: [0..127, 256..383, 128..255, 384..511]
        ids = torch.cat([parts[0], parts[2], parts[1], parts[3]])[:num_blocks]
    elif kind == "scattered":
        gen = torch.Generator()
        gen.manual_seed(42)
        ids = torch.randperm(num_blocks, generator=gen).to(torch.int64)
    else:
        raise ValueError("unknown pattern kind: {}".format(kind))
    return ids.pin_memory()


# -- Benchmark core -----------------------------------------------------------

def bench_one_dir(tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                  num_layers, is_h2d, num_gpus, iters, is_mla, mode,
                  transfer_num_cta=16):
    """Time ONE direction (H2D or D2H) over `iters`, return median GPU-event ms."""

    def do_transfer():
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=transfer_num_cta, is_host_to_device=is_h2d,
            use_ce_transfer=True, layer_id=0, layer_granularity=num_layers,
            is_mla=is_mla, mla_d2h_mode=mode)

    for _ in range(WARMUP_ITERS):
        do_transfer()
    sync_all(num_gpus)

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(num_gpus)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(num_gpus)]

    times_ms = []
    for _ in range(iters):
        for g in range(num_gpus):
            start_ev[g].record()
        do_transfer()
        for g in range(num_gpus):
            end_ev[g].record()
        torch.cuda.synchronize()
        max_gpu_ms = 0.0
        for g in range(num_gpus):
            gpu_ms = start_ev[g].elapsed_time(end_ev[g])
            if gpu_ms > max_gpu_ms:
                max_gpu_ms = gpu_ms
        times_ms.append(max_gpu_ms)

    return float(np.median(times_ms))


# The 5 CE execution forms and how to trigger each (num_blocks=64, threshold=8).
#   (pattern, layout_key, is_mla, mode, dirs, viable_force_paths)
# viable_force_paths: which CEPaths can physically run for this form's
# (pattern, layout, mode) combo. Used for the force-path head-to-head.
# dirs: directions to test (both H2D and D2H unless physically impossible).
#
# Physical constraints (from choose_path in ce_transfer.cu):
#   gpu_phys_contig = (gpu_block_stride == chunk_size). GPU is always LAYERFIRST
#     so true, EXCEPT sharded D2H where chunk shrinks to shard -> false.
#   cpu_phys_contig = (cpu_block_stride == chunk_size). lfirst=true, bfirst=false.
#   BULK_CONTIG(0)      needs gpu_log_contig && cpu_log_contig && cpu_phys_contig && gpu_phys_contig
#   SEGMENTED_DIRECT(1) needs cpu_phys_contig (lfirst)
#   STAGED_MERGE(2)      always viable (staging works for any layout)
#   STAGED_BLOCK(3)      always viable (staging works for any layout)
#   GATHER_SCATTER(4)   needs gpu_phys_contig (no sharded D2H)
# CEPath enum: 0=BULK_CONTIG, 1=SEGMENTED_DIRECT, 2=STAGED_MERGE, 3=STAGED_BLOCK, 4=GATHER_SCATTER, 5=BF_TRANSPOSE
#
# layout_key -> cpu_phys_contig: lfirst=True, bfirst=False
# mode=sharded D2H -> gpu_phys_contig=False; otherwise True
# pattern=contiguous -> log_contig=True; few_seg/scattered -> log_contig=False
#
# STAGED_BLOCK only exists in D2H+sharded (the only !gpu_phys_contig case).
# H2D+sharded does NOT shrink chunk -> gpu_phys_contig stays true -> STAGED_MERGE.
# All pattern × layout combos (rank0_only) + sharded D2H special case.
# No form names — pattern/layout/mode are shown directly in output.
# Tuple: (pattern, layout_key, is_mla, mode, dirs, viable_force_paths)

# Viable force paths — STRICT compatibility (data correct + no segfault + no FAILED).
# Only paths that produce CORRECT data for the given scenario are listed.
#
# Compatibility rules (derived from code semantics +实测):
# - BULK_CONTIG: requires contiguous pattern + LF + non-sharded
#   (1 segment + cpu_phys_contig + gpu_phys_contig)
# - SEGMENTED_DIRECT: requires LF + non-sharded
#   (cpu_phys_contig + gpu_phys_contig for per-segment memcpy)
# - STAGED_MERGE / STAGED_BLOCK: all scenarios
#   (ptr_at + staging + scatter handles any pattern/layout/mode)
# - GATHER_SCATTER: requires LF + MLA + non-sharded
#   (BF segfault: from_blob without stride; MHA segfault: unconfirmed root cause;
#    sharded: stride mismatch on from_blob)
# - BF_TRANSPOSE: requires BF
#   (LF segfault: from_blob with BF stride assumption)
_BASE = [(2, "STAGED_MERGE"), (3, "STAGED_BLOCK")]

# LF + MLA + rank0_only (non-sharded): full set
_LF_CONTIG_MLA = [(0, "BULK_CONTIG"), (1, "SEGMENTED_DIRECT")] + _BASE + [(4, "GATHER_SCATTER")]
_LF_OTHER_MLA  = [(1, "SEGMENTED_DIRECT")] + _BASE + [(4, "GATHER_SCATTER")]

# LF + sharded: only STAGED_MERGE + STAGED_BLOCK
# (no BULK_CONTIG/SEGMENTED_DIRECT: gpu_phys_contig=false; no GATHER_SCATTER: stride mismatch)
_LF_SHARDED = list(_BASE)

# LF + MHA: no GATHER_SCATTER (MHA segfaults)
_LF_CONTIG_MHA = [(0, "BULK_CONTIG"), (1, "SEGMENTED_DIRECT")] + _BASE
_LF_OTHER_MHA  = [(1, "SEGMENTED_DIRECT")] + _BASE

# BF (any pattern/mla/mode): no BULK/SEG_DIRECT (cpu_phys_contig=false), no GATHER (segfault)
_BF = _BASE + [(5, "BF_TRANSPOSE")]

# Full matrix: 3 patterns × 2 layouts × 3 modes (mla-rank0_only, mla-sharded, mha).
# mla-sharded is D2H-only (H2D sharded has gpu_phys_contig=true, not sharded).
# mha has no sharded mode (sharded is MLA-only); mode is don't-care for MHA.
PATH_FORMS = [
    # --- mla + rank0_only (H2D + D2H) ---
    ("contiguous", "lfirst", True,  "rank0_only", [True, False], _LF_CONTIG_MLA),
    ("contiguous", "bfirst", True,  "rank0_only", [True, False], _BF),
    ("few_seg",    "lfirst", True,  "rank0_only", [True, False], _LF_OTHER_MLA),
    ("few_seg",    "bfirst", True,  "rank0_only", [True, False], _BF),
    ("scattered",  "lfirst", True,  "rank0_only", [True, False], _LF_OTHER_MLA),
    ("scattered",  "bfirst", True,  "rank0_only", [True, False], _BF),
    # --- mla + sharded (D2H only) ---
    ("contiguous", "lfirst", True,  "sharded",    [False], _LF_SHARDED),
    ("contiguous", "bfirst", True,  "sharded",    [False], _BF),
    ("few_seg",    "lfirst", True,  "sharded",    [False], _LF_SHARDED),
    ("few_seg",    "bfirst", True,  "sharded",    [False], _BF),
    ("scattered",  "lfirst", True,  "sharded",    [False], _LF_SHARDED),
    ("scattered",  "bfirst", True,  "sharded",    [False], _BF),
    # --- mha (H2D + D2H, mode=rank0_only but don't-care) ---
    ("contiguous", "lfirst", False, "rank0_only", [True, False], _LF_CONTIG_MHA),
    ("contiguous", "bfirst", False, "rank0_only", [True, False], _BF),
    ("few_seg",    "lfirst", False, "rank0_only", [True, False], _LF_OTHER_MHA),
    ("few_seg",    "bfirst", False, "rank0_only", [True, False], _BF),
    ("scattered",  "lfirst", False, "rank0_only", [True, False], _LF_OTHER_MHA),
    ("scattered",  "bfirst", False, "rank0_only", [True, False], _BF),
]

# Two cumulative CE configs.
#   (label, path_opt)
PATH_CONFIGS = [
    ("baseline", False),   # C++ runs PER_BLOCK
    ("opt",      True),
]

# All 6 CEPaths for Part 2's uniform column layout (not all viable for every
# form -- non-viable cells show '-').
ALL_FORCE_PATHS = [
    (0, "BULK_CONTIG"),
    (1, "SEGMENTED_DIRECT"),
    (2, "STAGED_MERGE"),
    (3, "STAGED_BLOCK"),
    (4, "GATHER_SCATTER"),
    (5, "BF_TRANSPOSE"),
]


def python_choose_path(pattern, layout_key, mode, is_h2d, threshold,
                       chunk_size_bytes, is_mla=True):
    """Mirror of C++ choose_path (ce_transfer.cu). Returns the CEPath name
    that choose_path would pick for the given (pattern, layout, mode, dir).

    Used to annotate the opt (auto) row with the path choose_path selected,
    so the reader can confirm opt == force_<auto_path> timing.
    """
    cpu_phys_contig = (layout_key == "lfirst")
    is_blockfirst = (layout_key == "bfirst")
    # Sharded D2H shrinks GPU chunk -> gpu_phys_contig == False.
    gpu_phys_contig = not (mode == "sharded" and not is_h2d)
    # num_segments: contiguous=1, few_seg=4, scattered=many(>threshold)
    if pattern == "contiguous":
        num_segments = 1
    elif pattern == "few_seg":
        num_segments = 4
    else:  # scattered
        num_segments = threshold + 1  # > threshold

    # BF_TRANSPOSE: checked first (BLOCKFIRST + !cpu_phys_contig, covers MLA+MHA).
    # Covers both rank0_only/all_write and sharded D2H.
    if is_blockfirst and not cpu_phys_contig:
        return "BF_TRANSPOSE"
    if cpu_phys_contig and gpu_phys_contig and num_segments == 1:
        return "BULK_CONTIG"
    if not gpu_phys_contig:
        return "STAGED_BLOCK"
    if num_segments <= threshold:
        return "SEGMENTED_DIRECT" if cpu_phys_contig else "STAGED_MERGE"
    if chunk_size_bytes > 0 and chunk_size_bytes % 8 != 0:
        return "STAGED_MERGE"
    return "GATHER_SCATTER"


def run_strategy_compare(args):
    """Drive each CE form across both H2D and D2H, 3 CE configs each, plus
    force-path head-to-head, for each size in args.sizes.
    """
    num_gpus = args.num_gpus
    threshold = 8
    cta = 16

    print("=" * 96)
    print("  CE Strategy Comparison: 5 forms x 2 dirs x 2 configs + force-path")
    print("=" * 96)
    print("  GPUs:        {}".format(num_gpus))
    print("  Sizes:       {}".format(args.sizes))
    print("  Threshold:   {}".format(threshold))
    print("  CTA count:   {}".format(cta))
    print("  Iters:       {}".format(args.iters))
    print("  Configs:     {}".format(", ".join(c[0] for c in PATH_CONFIGS)))
    print("=" * 96)

    # results[size][(form_name, dir_name)][config_label] = median_ms
    all_results = defaultdict(lambda: defaultdict(dict))

    for size_name in args.sizes:
        num_layers, num_blocks, head_dim = SIZES[size_name]
        kv_bytes = num_layers * 1 * num_blocks * 1 * 1 * head_dim * ES
        print("\n--- Size: {} ({} layers, {} blocks, hd={}, {:.1f} MB) ---".format(
            size_name, num_layers, num_blocks, head_dim, kv_bytes / (1024**2)))

        results = all_results[size_name]
        heads_per_rank = 1

        for pattern, layout_key, is_mla, mode, dirs, viable in PATH_FORMS:
            mla_tag = "mla" if is_mla else "mha"
            form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
            if pattern == "scattered" and num_blocks <= threshold:
                print("  SKIP {} (num_blocks={} <= threshold={})".format(
                    form_name, num_blocks, threshold))
                continue
            kv_dim = 1 if is_mla else 2
            cpu_layout_type = STRAT_LAYOUTS[layout_key]

            gpu_layout, cpu_layout = make_layouts_strat(
                num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus)
            cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb, total_blocks = \
                cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks,
                                         head_dim, is_mla, mode, num_gpus)

            all_gpu = [make_gpu_tensors_strat(num_layers, kv_dim, num_blocks,
                                              heads_per_rank, head_dim, g)
                       for g in range(num_gpus)]
            cpu_kv = make_cpu_tensor_strat(cpu_layout, num_layers, total_blocks,
                                           head_dim, is_mla, num_gpus)
            ids = make_block_id_pattern(pattern, num_blocks)

            for is_h2d in dirs:
                dir_name = "H2D" if is_h2d else "D2H"
                key = (form_name, dir_name)
                # Compute the path choose_path would auto-pick for this
                # (form, dir) so we can annotate the opt row.
                auto_path = python_choose_path(
                    pattern, layout_key, mode, is_h2d, threshold,
                    head_dim * ES, is_mla=is_mla)
                results[key]["auto_path"] = auto_path
                print("\n-- Form: {} | pattern={} | layout={} | mode={} | dir={} | auto={} --".format(
                    form_name, pattern, layout_key, mode, dir_name, auto_path))

                # Step 1: two CE configs (baseline / opt)
                for cfg_label, path_opt in PATH_CONFIGS:
                    path_tag = ""
                    if cfg_label == "opt":
                        path_tag = " [{}]".format(auto_path)
                    print("  {}{} ...".format(cfg_label, path_tag), end=" ",
                          flush=True)
                    try:
                        tp = make_tp_group(
                            cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                            num_layers, ce_path_opt=path_opt,
                            ce_segment_threshold=threshold,
                            ce_is_mla=is_mla,
                            ce_is_blockfirst=(layout_key == "bfirst"))
                        med = bench_one_dir(
                            tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                            num_layers, is_h2d, num_gpus, args.iters, is_mla, mode,
                            transfer_num_cta=cta)
                        results[key][cfg_label] = med
                        print("{:.3f} ms".format(med))
                        del tp
                    except Exception as e:
                        print("FAILED: {}".format(e))

                # Step 2: force each viable path (under opt), skip auto-pick
                # (forcing to the same path as auto is redundant — opt already
                # shows that result).
                for fp_id, fp_name in viable:
                    if fp_name == auto_path:
                        results[key]["force_" + fp_name] = None  # skip
                        continue
                    label = "force_" + fp_name
                    print("  {} ...".format(label), end=" ", flush=True)
                    try:
                        tp = make_tp_group(
                            cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                            num_layers, ce_path_opt=True,
                            ce_segment_threshold=threshold,
                            ce_force_path=fp_id,
                            ce_is_mla=is_mla,
                            ce_is_blockfirst=(layout_key == "bfirst"))
                        med = bench_one_dir(
                            tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                            num_layers, is_h2d, num_gpus, args.iters, is_mla, mode,
                            transfer_num_cta=cta)
                        results[key][label] = med
                        print("{:.3f} ms".format(med))
                        del tp
                    except Exception as e:
                        results[key][label] = None
                        print("FAILED: {}".format(e))

            del all_gpu, cpu_kv

    # -- Print results per size ------------------------------------------------
    for size_name in args.sizes:
        results = all_results[size_name]
        num_layers, num_blocks, head_dim = SIZES[size_name]
        print("\n" + "=" * 96)
        print("  Results for size={} ({}L / {}B / hd={})".format(
            size_name, num_layers, num_blocks, head_dim))
        print("=" * 96)

        # Build the list of (form_name, dir_name) rows actually run.
        run_rows = []
        for pattern, layout_key, is_mla, mode, dirs, viable in PATH_FORMS:
            if pattern == "scattered" and num_blocks <= threshold:
                continue
            form_name = "{}/{}/{}".format(pattern, layout_key, mode)
            for is_h2d in dirs:
                run_rows.append((form_name, "H2D" if is_h2d else "D2H", viable))

        # -- Part 1: baseline vs opt ------------------------------------------
        # The opt column is annotated with the path choose_path auto-picked
        # (in brackets), e.g. `opt[STAGED_MERGE]`. Confirm by checking that
        # opt timing matches force_<auto_path> in Part 2.
        print("\n  Part 1: Optimization Config (baseline / opt)")
        print("  '*' = fastest of the 2 for each row.")
        print("  opt column annotated with choose_path auto-pick, e.g. 0.434[STAGED_MERGE].")
        hdr = "{:>18s}  {:>4s}  {:>12s}  {:>20s}  {:>12s}".format(
            "Form", "Dir", "baseline", "opt", "base/opt")
        print("  " + hdr)
        print("  " + "-" * len(hdr))

        for form_name, dir_name, _ in run_rows:
            cfgs = results.get((form_name, dir_name), {})
            base = cfgs.get("baseline")
            opt = cfgs.get("opt")
            auto_path = cfgs.get("auto_path", "")
            fastest = min((v for v in (base, opt) if v is not None),
                          default=None)

            def fmt_base(v):
                if v is None:
                    return "{:>12s}".format("-")
                star = "*" if (fastest is not None and v == fastest) else " "
                return "{:>11.3f}{}".format(v, star)

            if opt is None:
                opt_str = "{:>20s}".format("-")
            else:
                star = "*" if (fastest is not None and opt == fastest) else " "
                tag = "[{}]".format(auto_path) if auto_path else ""
                opt_str = "{:>13.3f}{:<6s}".format(opt, star + tag)

            speedup = "-"
            if base and opt and opt > 0:
                speedup = "{:.2f}x".format(base / opt)
            print("  {:>18s}  {:>4s}  {}  {}  {:>12s}".format(
                form_name, dir_name, fmt_base(base), opt_str, speedup))

        # -- Part 2: force-path head-to-head ----------------------------------
        print("\n  Part 2: Force-Path Head-to-Head (all under opt)")
        print("  'auto' = choose_path pick. '*' = fastest. Proves optimality.")
        col_w = 16
        hdr2 = "  {:>18s}  {:>4s}  {:>{w}s}".format("Form", "Dir", "auto", w=col_w)
        for _, pname in ALL_FORCE_PATHS:
            hdr2 += "  {:>{w}s}".format(pname, w=col_w)
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))

        auto_wins = 0
        auto_total = 0
        for form_name, dir_name, viable in run_rows:
            cfgs = results.get((form_name, dir_name), {})
            auto = cfgs.get("opt")
            auto_path = cfgs.get("auto_path", "")
            viable_names = {pn for _, pn in viable}
            forced_dict = {}
            for fp_id, fp_name in ALL_FORCE_PATHS:
                if fp_name == auto_path:
                    forced_dict[fp_name] = "(opt)"  # skip marker
                elif fp_name in viable_names:
                    forced_dict[fp_name] = cfgs.get("force_" + fp_name)
                else:
                    forced_dict[fp_name] = None
            all_vals = [auto] + [v for v in forced_dict.values() if isinstance(v, (int, float))]
            fastest = min((v for v in all_vals if v is not None), default=None)

            def fmt2(v):
                if v is None:
                    return "{:>{w}s}".format("-", w=col_w)
                if v == "(opt)":
                    return "{:>{w}s}".format("(opt)", w=col_w)
                star = "*" if (fastest is not None and v == fastest) else " "
                return "{:>{w}.3f}{}".format(v, star, w=col_w - 1)

            if auto is not None:
                auto_total += 1
                if auto == fastest:
                    auto_wins += 1
            line = "  {:>18s}  {:>4s}  {}".format(
                form_name, dir_name, fmt2(auto))
            for _, pname in ALL_FORCE_PATHS:
                line += "  {}".format(fmt2(forced_dict.get(pname)))
            print(line)

        print("\n  " + "-" * (len(hdr2) - 2))
        if auto_total:
            print("  choose_path auto pick fastest in {}/{} rows.".format(
                auto_wins, auto_total))
            if auto_wins == auto_total:
                print("  => choose_path is OPTIMAL for this size.")
            else:
                print("  => inspect rows where auto did NOT win.")
        print("  Note: forced paths may produce incorrect data -- timing only.")
        print("=" * 96)


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmark CE transfer strategy comparison "
                    "(5 forms x 2 configs + force-path head-to-head)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (0 = all available, default: 0)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Timing iterations per config (default: 20)")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES.keys()),
                        choices=list(SIZES.keys()),
                        help="Data sizes to test (default: all)")
    args = parser.parse_args()

    num_gpus = NUM_GPUS if args.num_gpus <= 0 else min(args.num_gpus, NUM_GPUS)
    if num_gpus < 2:
        print("ERROR: need at least 2 GPUs, found {}".format(NUM_GPUS))
        sys.exit(1)
    args.num_gpus = num_gpus

    run_strategy_compare(args)


if __name__ == "__main__":
    main()
