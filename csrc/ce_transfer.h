/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CE adaptive transfer: host-side analysis + multi-path execution.
 * Extracted from transfer.cu for maintainability and future extension
 * (e.g. multi-process scatter).
 */
#pragma once

#include "gtensor_handler.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

// ============================================================================
// CE transfer configuration (passed from Python GLOBAL_CONFIG_FROM_ENV)
// ============================================================================

struct CETransferConfig {
  int64_t segment_threshold = 8;
  // path_opt_enabled: master switch between the PER_BLOCK baseline (one memcpy
  // per block, the slow reference used to quantify optimization gains) and the
  // adaptive optimized strategies chosen by choose_path(). false = PER_BLOCK
  // always; true = pick BULK_CONTIG / SEGMENTED_DIRECT / STAGED_MERGE /
  // STAGED_BLOCK / GATHER_SCATTER based on block-id contiguity + CPU/GPU layout.
  bool path_opt_enabled = true;
  // force_path: test/benchmark only. -1 = auto (choose_path); 0-4 = force a
  // specific CEPath (0=BULK_CONTIG, 1=SEGMENTED_DIRECT, 2=STAGED_MERGE,
  // 3=STAGED_BLOCK, 4=GATHER_SCATTER). Production MUST leave this at -1.
  // Used by microbenchmark_ce_strategy.py to prove choose_path picks the
  // fastest strategy for each case (runs all viable paths head-to-head).
  // NOTE: BF_TRANSPOSE is no longer a CEPath enum value; it is a
  // preprocess entry checked before choose_path (see transfer.cu).
  int force_path = -1;
  // enable_memcpy2d: when true, STAGED_MERGE/STAGED_BLOCK D2H uses cudaMemcpy2DAsync
  // (strided D2H directly to CPU positions). Fast on NVIDIA (H20:
  // 58ms, 24 GiB/s), catastrophically slow on P800/Kunlunxin (12.8s, 0.11
  // GiB/s — the DMA engine does not handle 2D strided patterns). Default
  // false: use staging buffer + CPU scatter (works on all platforms).
  // Set to 1 on NVIDIA via FLEXKV_ENABLE_MEMCPY2D=1.
  bool enable_memcpy2d = false;
  // is_blockfirst: CPU KV cache layout is BLOCKFIRST (vs LAYERFIRST).
  // Set from FLEXKV_CPU_LAYOUT env var via worker.py/layerwise.py.
  // The BF MLA preprocess (transfer.cu) uses this to select BF_TRANSPOSE
  // only for actual BLOCKFIRST layouts (not LAYERFIRST non-MLA where
  // !cpu_phys_contig is also true due to per-rank chunk_size < cpu_block_stride).
  bool is_blockfirst = false;
  // is_mla: whether the model uses MLA (kv_dim=1, no head split).
  // BF_TRANSPOSE preprocess is only triggered when is_blockfirst && is_mla
  // (BF MHA has tp-strided layout that D2D transpose cannot fix).
  bool is_mla = false;
};

// ============================================================================
// CE transfer strategy taxonomy
// ============================================================================
//
// Every CE transfer is one of six execution strategies plus the BF MLA
// preprocess. path_opt_enabled selects PER_BLOCK (the baseline) vs the five
// optimized strategies; among the optimized ones choose_path() picks based on
// the CEAnalysis flags. BF_TRANSPOSE is a preprocess entry checked before
// choose_path (see transfer.cu), not a CEPath enum value.
//
//   PER_BLOCK       baseline: one cudaMemcpyAsync per block. No merging, no
//                   staging. Correct for every layout; slowest. Only used when
//                   path_opt_enabled == false.
//   BULK_CONTIG     cpu_phys_contig && gpu_phys_contig && num_segments == 1:
//                   a single large memcpy per (layer, kv). No staging.
//   SEGMENTED_DIRECT few merged runs, dst physically contiguous: one memcpy
//                   per contiguous run, straight CPU<->GPU. No staging, so
//                   ping-pong does not apply.
//   STAGED_MERGE    BLOCKFIRST + GPU contiguous, few segments: copy via a
//                   pinned staging buffer (merged segment memcpy), then CPU
//                   scatter/gather to the strided destination. Uses staging
//                   (D2H ping-pong enabled).
//   STAGED_BLOCK    sharded D2H (GPU NOT contiguous), few segments: per-block
//                   memcpy via a pinned staging buffer, then CPU scatter/gather.
//                   Uses staging (D2H ping-pong disabled when both sides are
//                   non-contiguous — per-block granularity makes event overhead
//                   dominate).
//   STAGED_MERGE and STAGED_BLOCK used to share a single ce_transfer_staged_scatter
//   implementation that selected the inner copy loop at runtime by
//   gpu_phys_contig; they are now split into two self-describing functions
//   (ce_transfer_staged_merge / ce_transfer_staged_block) that share the
//   staging buffer allocator (get_cached_hugepage_buffer), scatter_to_cpu /
//   gather_from_cpu, and the cached ping-pong event helper
//   (get_cached_event_pair).
//   GATHER_SCATTER  many scattered segments (> segment_threshold), GPU blocks
//                   physically contiguous: GPU index_select gather (D2H) /
//                   index_copy_ scatter (H2D) through a staging buffer. Uses
//                   staging (D2H ping-pong enabled).
//
// ping-pong summary: D2H only, applies to STAGED_MERGE, STAGED_BLOCK (when
// not both-sides-non-contiguous), and GATHER_SCATTER.
enum class CEPath : int {
  PER_BLOCK = -1,       // baseline (path_opt_enabled == false)
  BULK_CONTIG = 0,
  SEGMENTED_DIRECT = 1,
  STAGED_MERGE = 2,     // BLOCKFIRST + GPU contiguous: merged segment memcpy + staging + scatter
  STAGED_BLOCK = 3,     // sharded D2H (GPU non-contiguous): per-block memcpy + staging + scatter
  GATHER_SCATTER = 4,
};

// ============================================================================
// Analysis structs
// ============================================================================

struct CESegment {
  int start_k;    // start block index in the id arrays
  int run_len;    // number of blocks in this segment
};

// CEAnalysis: contiguity analysis of the block-id arrays.
//
// Naming is direction-agnostic: "gpu_*" always refers to the GPU side,
// "cpu_*" always refers to the CPU side, regardless of transfer direction.
//   D2H: GPU is the source, CPU is the destination (names match intuition).
//   H2D: CPU is the source, GPU is the destination (names are swapped vs
//        intuition). H2D paths must use cpu_* flags for the actual source
//        and gpu_* flags for the actual destination. This prevents the
//        silent data-corruption bug that would arise from treating src=CPU
//        when the field is named "src" but actually means GPU.
//
// Note: gpu_log_contig / cpu_log_contig are no longer used by choose_path
// (BULK_CONTIG now uses num_segments == 1). They remain in use by
// ce_transfer_gather_scatter and ce_transfer_bf_transpose to decide
// whether GPU index_select / index_copy_ is needed, so they are still
// computed by analyze_ce_transfer.
struct CEAnalysis {
  bool gpu_log_contig;   // gpu_block_ids[k+1] == gpu_block_ids[k]+1
  bool cpu_log_contig;   // cpu_block_ids[k+1] == cpu_block_ids[k]+1
  bool cpu_phys_contig;  // cpu_block_stride == chunk_size (LAYERFIRST + non-sharded)
  bool gpu_phys_contig;  // gpu_block_stride == chunk_size (non-sharded D2H)
  int num_segments;
  std::vector<CESegment> segments;
};

// ============================================================================
// Analysis & path selection
// ============================================================================

CEAnalysis analyze_ce_transfer(
    const int64_t *gpu_block_ids, const int64_t *cpu_block_ids,
    int num_blocks, int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t gpu_block_stride_in_bytes);

CEPath choose_path(const CEAnalysis &a, const CETransferConfig &ce_config,
                   int64_t chunk_size_in_bytes = 0);

// ============================================================================
// Cached staging buffers & events
// ============================================================================

void *get_cached_hugepage_buffer(size_t size);
void *get_cached_device_buffer(size_t size, int slot = 0);

// ============================================================================
// PER_BLOCK (baseline): one memcpy per block, no merging / no staging.
//   Used when path_opt_enabled == false. Correct for every layout; slowest.
// ============================================================================
template <BackendType Type>
void ce_transfer_per_block(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device);

// ============================================================================
// BULK_CONTIG: single large memcpy per (layer, kv_dim). No staging.
//   Requires: cpu_phys_contig && gpu_phys_contig && num_segments == 1
// ============================================================================
template <BackendType Type>
void ce_transfer_bulk_contig(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device);

// ============================================================================
// SEGMENTED_DIRECT: per-merged-run memcpy, dst physically contiguous.
//   No staging buffer, so ping-pong does not apply.
//   Chosen when cpu_phys_contig (LAYERFIRST + non-sharded) with few segments.
// ============================================================================
template <BackendType Type>
void ce_transfer_segmented_direct(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &ce_config);

// ============================================================================
// STAGED_MERGE: pinned staging buffer + merged-segment memcpy + CPU
//   scatter/gather to a strided destination. Chosen by choose_path when
//   gpu_phys_contig (GPU blocks physically contiguous -> one memcpy per
//   contiguous run of block ids). Ping-pong: D2H layer-level (always enabled
//   for D2H; is_per_block is always false since gpu_phys_contig is true).
//   enable_memcpy2d: when true, D2H/H2D uses cudaMemcpy2DAsync per segment
//   (strided GPU<->CPU directly, bypassing staging + scatter/gather).
// ============================================================================
template <BackendType Type>
void ce_transfer_staged_merge(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &ce_config);

// ============================================================================
// STAGED_BLOCK: pinned staging buffer + per-block memcpy + CPU scatter/gather.
//   Chosen by choose_path when !gpu_phys_contig (sharded D2H: GPU block stride
//   != chunk_size). One memcpy per block (no segment merging possible).
//   Ping-pong: D2H disabled when both sides non-contiguous
//   (!gpu_phys_contig && !cpu_phys_contig -> per-block granularity makes event
//   overhead dominate); enabled when D2H && cpu_phys_contig.
//   enable_memcpy2d: when true, D2H/H2D uses cudaMemcpy2DAsync per segment
//   (strided GPU<->CPU directly, bypassing staging + scatter/gather).
// ============================================================================
template <BackendType Type>
void ce_transfer_staged_block(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &ce_config);

// ============================================================================
// GATHER_SCATTER: GPU index_select/index_copy_ pipeline through a staging
//   buffer, for many scattered segments (> segment_threshold). Uses staging
//   (D2H ping-pong enabled). Requires gpu_phys_contig (GPU block stride == chunk).
//     D2H: GPU index_select gather -> D2H staging -> CPU scatter
//     H2D: CPU gather -> H2D staging -> GPU index_copy_ scatter
// ============================================================================
template <BackendType Type>
void ce_transfer_gather_scatter(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &ce_config);

// ============================================================================
// BF_TRANSPOSE (preprocess entry, NOT a CEPath enum value):
//   BF MLA (rank0_only/all_write) D2H/H2D. Called from transfer.cu BEFORE
//   choose_path() when is_blockfirst && is_mla && !cpu_phys_contig &&
//   gpu_phys_contig. D2D transpose (LAYERFIRST->BLOCKFIRST) via index_select +
//   transpose + contiguous, then per-segment cudaMemcpyAsync
//   (contiguous/segmented/per-block) matching the transposed BLOCKFIRST layout.
//   No CPU scatter needed for contiguous/few_seg; per-block for scattered.
//   D2D SM overhead <1ms (large). Only triggered when is_blockfirst && is_mla
//   (BF MHA has tp-strided layout that D2D transpose cannot fix).
// ============================================================================
template <BackendType Type>
void ce_transfer_bf_transpose(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &ce_config);

// ============================================================================
// scatter_to_cpu: scatter from contiguous staging buffer to strided CPU dst.
//   When cpu_phys_contig (LAYERFIRST), consecutive cpu_block_ids are also
//   physically adjacent, so we merge them into a single memcpy. When
//   !cpu_phys_contig (BLOCKFIRST), consecutive block_ids have a stride gap
//   between them, so each block must be scattered individually.
//   Shared by STAGED_MERGE/STAGED_BLOCK and GATHER_SCATTER.
// ============================================================================
void scatter_to_cpu(const void *staging_buf, int64_t *cpu_ptr_int64,
                    int64_t *cpu_block_ids, int num_blocks,
                    int64_t cpu_block_stride_int64,
                    int64_t cpu_startoff_inside_chunks_int64,
                    int64_t chunk_size_in_bytes, int layer_idx, int kv_idx,
                    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
                    int start_layer_id, bool cpu_phys_contig);

// ============================================================================
// gather_from_cpu: gather from strided CPU positions to contiguous staging buf.
//   H2D symmetric counterpart of scatter_to_cpu. Same parameters and merge
//   optimization, with src/dst swapped and const-ness adjusted.
//   Shared by STAGED_MERGE/STAGED_BLOCK and GATHER_SCATTER.
// ============================================================================
void gather_from_cpu(void *staging_buf, const int64_t *cpu_ptr_int64,
                     const int64_t *cpu_block_ids, int num_blocks,
                     int64_t cpu_block_stride_int64,
                     int64_t cpu_startoff_inside_chunks_int64,
                     int64_t chunk_size_in_bytes, int layer_idx, int kv_idx,
                     int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
                     int start_layer_id, bool cpu_phys_contig);

} // namespace flexkv
