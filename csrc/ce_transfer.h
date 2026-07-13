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
  // always; true = pick BULK_CONTIG / SEGMENTED_DIRECT / STAGED_SCATTER /
  // GATHER_SCATTER based on block-id contiguity + CPU/GPU layout.
  bool path_opt_enabled = true;
  // force_path: test/benchmark only. -1 = auto (choose_path); 0-3 = force a
  // specific CEPath (0=BULK_CONTIG, 1=SEGMENTED_DIRECT, 2=STAGED_SCATTER,
  // 3=GATHER_SCATTER). Production MUST leave this at -1 (never set it).
  // Used by microbenchmark_ce_strategy.py to prove choose_path picks the
  // fastest strategy for each case (runs all viable paths head-to-head).
  int force_path = -1;
};

// ============================================================================
// CE transfer strategy taxonomy
// ============================================================================
//
// Every CE transfer is one of five execution strategies. path_opt_enabled
// selects PER_BLOCK (the baseline) vs the four optimized strategies; among the
// optimized ones choose_path() picks based on the CEAnalysis flags.
//
//   PER_BLOCK       baseline: one cudaMemcpyAsync per block. No merging, no
//                   staging. Correct for every layout; slowest. Only used when
//                   path_opt_enabled == false.
//   BULK_CONTIG     block ids fully contiguous on both sides AND dst is
//                   physically contiguous (LAYERFIRST, non-sharded): a single
//                   large memcpy per (layer, kv). No staging.
//   SEGMENTED_DIRECT few merged runs, dst physically contiguous: one memcpy
//                   per contiguous run, straight CPU<->GPU. No staging, so
//                   ping-pong does not apply.
//   STAGED_SCATTER  dst NOT physically contiguous (BLOCKFIRST, or sharded D2H)
//                   with few segments: copy via a pinned staging buffer, then
//                   CPU scatter/gather to the strided destination. Uses
//                   staging (D2H ping-pong enabled). Two internal GPU-side
//                   variants, selected at runtime by gpu_phys_contig:
//                     STAGED_CONTIG_RUN  GPU blocks contiguous  -> one memcpy
//                                        per merged run between GPU and staging
//                     STAGED_PER_BLOCK   GPU blocks NOT contiguous (sharded
//                                        D2H) -> one memcpy per block
//                   These are NOT top-level strategies: they share the same
//                   staging + scatter + ping-pong machinery and differ only in
//                   the innermost copy loop, so they live inside STAGED_SCATTER.
//   GATHER_SCATTER  many scattered segments (> segment_threshold), GPU blocks
//                   physically contiguous: GPU index_select gather (D2H) /
//                   index_copy_ scatter (H2D) through a staging buffer. Uses
//                   staging (D2H ping-pong enabled).
//
// ping-pong summary: D2H only, applies to STAGED_SCATTER and GATHER_SCATTER.
enum class CEPath : int {
  PER_BLOCK = -1,       // baseline (path_opt_enabled == false)
  BULK_CONTIG = 0,
  SEGMENTED_DIRECT = 1,
  STAGED_SCATTER = 2,
  GATHER_SCATTER = 3,
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
//   Requires: gpu_log_contig && cpu_log_contig && cpu_phys_contig && gpu_phys_contig
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
// STAGED_SCATTER: pinned staging buffer + CPU scatter/gather to a strided
//   destination. Uses staging (D2H ping-pong enabled). Chosen when dst NOT
//   physically contiguous (BLOCKFIRST, or sharded D2H) with few segments.
//   Internally selects a GPU-side variant at runtime by gpu_phys_contig:
//     STAGED_CONTIG_RUN (GPU contiguous) -> merged-run memcpy GPU<->staging
//     STAGED_PER_BLOCK  (sharded D2H)    -> per-block   memcpy GPU<->staging
// ============================================================================
template <BackendType Type>
void ce_transfer_staged_scatter(
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

} // namespace flexkv
