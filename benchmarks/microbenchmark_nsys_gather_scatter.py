"""
Minimal nsys profiling script for GATHER_SCATTER CE path.

Runs ONLY the GATHER_SCATTER path (scattered block IDs, bfirst layout,
rank0_only mode) for D2H and H2D, with enough iterations for nsys to
capture meaningful kernel statistics.

Usage:
    nsys profile -o ce_gather_scatter_profile \
        --trace=cuda,nvtx,osrt \
        --stats=true \
        python benchmarks/microbenchmark_nsys_gather_scatter.py --num-gpus 8

Then inspect the .nsys-rep file with:
    nsys stats ce_gather_scatter_profile.nsys-rep
    # Or open in Nsight Systems GUI

The script uses NVTX ranges to mark:
  - "GATHER_SCATTER_D2H" around the D2H iterations
  - "GATHER_SCATTER_H2D" around the H2D iterations
  - "WARMUP" around warmup iterations (ignore these in analysis)
"""

import argparse
import sys

import torch

try:
    import nvtx
except ImportError:
    # Fallback: no-op decorator if nvtx not available
    class nvtx:
        @staticmethod
        def start_range(message, color="blue"):
            class _Range:
                def end(self):
                    pass
            return _Range()
        @staticmethod
        def end_range(range_id):
            pass

try:
    from flexkv.c_ext import TPTransferThreadGroup
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
except ImportError as e:
    print("ERROR: FlexKV not available ({})".format(e))
    sys.exit(1)

DTYPE = torch.float16
ES = DTYPE.itemsize


def make_layouts(num_layers, num_blocks, head_dim, is_mla, num_gpus):
    """GPU LAYERFIRST, CPU BLOCKFIRST (forces STAGED/GATHER_SCATTER path)."""
    num_head = 1 if is_mla else num_gpus
    heads_per_rank = 1
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=heads_per_rank,
        head_size=head_dim, is_mla=is_mla)
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return gpu_layout, cpu_layout


def main():
    parser = argparse.ArgumentParser(description="nsys profiling for GATHER_SCATTER")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=61,
                        help="Model layers (default: 61, DeepSeek-like)")
    parser.add_argument("--num-blocks", type=int, default=2048,
                        help="Number of blocks (default: 2048)")
    parser.add_argument("--head-dim", type=int, default=512,
                        help="Head dimension (default: 512)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Iterations per direction (default: 20)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations (default: 3)")
    args = parser.parse_args()

    num_gpus = args.num_gpus
    num_layers = args.num_layers
    num_blocks = args.num_blocks
    head_dim = args.head_dim
    is_mla = True
    mode = "rank0_only"
    kv_dim = 1
    threshold = 8
    cta = 16

    print("=" * 80)
    print("  nsys Profiling: GATHER_SCATTER (scattered, bfirst, MLA rank0_only)")
    print("=" * 80)
    print("  GPUs:       {}".format(num_gpus))
    print("  Layers:     {}".format(num_layers))
    print("  Blocks:     {}".format(num_blocks))
    print("  Head dim:   {}".format(head_dim))
    print("  Data size:  {:.1f} MB".format(
        num_layers * num_blocks * head_dim * ES / (1024**2)))
    print("  Iters:      {} (+{} warmup)".format(args.iters, args.warmup))
    print("  Threshold:  {}".format(threshold))
    print("=" * 80)

    # Setup layouts
    gpu_layout, cpu_layout = make_layouts(num_layers, num_blocks, head_dim,
                                           is_mla, num_gpus)

    # CPU strides
    total = num_blocks
    num_head = 1  # MLA
    layout_for_stride = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    cpu_kv_sb = layout_for_stride.get_kv_stride() * ES
    cpu_layer_sb = layout_for_stride.get_layer_stride() * ES
    cpu_block_sb = cpu_layout.get_block_stride() * ES
    cpu_tp_sb = cpu_block_sb // num_gpus

    # GPU tensors
    all_gpu = []
    for g in range(num_gpus):
        full = torch.empty(
            (num_layers, kv_dim, num_blocks, 1, 1, head_dim),
            dtype=DTYPE, device="cuda:{}".format(g))
        full.uniform_()
        all_gpu.append([full[i] for i in range(num_layers)])

    # CPU tensor
    cpu_kv = torch.empty(tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)

    # Scattered block IDs (triggers GATHER_SCATTER: many segments > threshold)
    gen = torch.Generator()
    gen.manual_seed(42)
    ids = torch.randperm(num_blocks, generator=gen).to(torch.int64).pin_memory()

    # Create TP group with GATHER_SCATTER forced
    gpu_ptrs = []
    for g in range(num_gpus):
        for l in range(num_layers):
            gpu_ptrs.append(all_gpu[g][l].data_ptr())

    tp = TPTransferThreadGroup(
        num_gpus=num_gpus, gpu_block_ptrs_flat=gpu_ptrs,
        num_tensors_per_gpu=num_layers, cpu_blocks_ptr=cpu_kv.data_ptr(),
        num_layers=num_layers,
        gpu_kv_strides_in_bytes=[gpu_layout.get_kv_stride() * ES] * num_gpus,
        gpu_block_strides_in_bytes=[gpu_layout.get_block_stride() * ES] * num_gpus,
        gpu_layer_strides_in_bytes=[gpu_layout.get_layer_stride() * ES] * num_gpus,
        gpu_chunk_sizes_in_bytes=[gpu_layout.get_chunk_size() * ES] * num_gpus,
        gpu_device_ids=list(range(num_gpus)),
        enable_nvcomp=False,
        ce_segment_threshold=threshold,
        ce_path_opt=True,
        ce_force_path=3)  # 3 = GATHER_SCATTER

    def do_transfer(is_h2d):
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb,
            cpu_layer_stride_in_bytes=cpu_layer_sb,
            cpu_block_stride_in_bytes=cpu_block_sb,
            cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=cta, is_host_to_device=is_h2d,
            use_ce_transfer=True, layer_id=0, layer_granularity=num_layers,
            is_mla=is_mla, mla_d2h_mode=mode)

    # Warmup
    print("\n[Warmup] {} iters...".format(args.warmup))
    warmup_range = nvtx.start_range("WARMUP", color="gray")
    for _ in range(args.warmup):
        do_transfer(is_h2d=False)  # D2H
        do_transfer(is_h2d=True)   # H2D
    nvtx.end_range(warmup_range)
    torch.cuda.synchronize()

    # D2H profiling
    print("\n[D2H] {} iters with NVTX range...".format(args.iters))
    d2h_range = nvtx.start_range("GATHER_SCATTER_D2H", color="red")
    for i in range(args.iters):
        do_transfer(is_h2d=False)
    nvtx.end_range(d2h_range)
    torch.cuda.synchronize()
    print("  D2H done.")

    # H2D profiling
    print("\n[H2D] {} iters with NVTX range...".format(args.iters))
    h2d_range = nvtx.start_range("GATHER_SCATTER_H2D", color="green")
    for i in range(args.iters):
        do_transfer(is_h2d=True)
    nvtx.end_range(h2d_range)
    torch.cuda.synchronize()
    print("  H2D done.")

    print("\n" + "=" * 80)
    print("  Profiling complete. Inspect with:")
    print("    nsys stats ce_gather_scatter_profile.nsys-rep")
    print("    # Or open .nsys-rep in Nsight Systems GUI")
    print("  Look for:")
    print("    - index_select / index_copy_ kernel durations")
    print("    - cudaMemcpyAsync durations")
    print("    - SM utilization during GATHER_SCATTER ranges")
    print("    - Memory bandwidth utilization")
    print("=" * 80)


if __name__ == "__main__":
    main()
