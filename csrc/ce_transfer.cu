/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CE adaptive transfer implementation: host-side analysis + multi-path
 * execution. Extracted from transfer.cu.
 *
 * Five optimized paths selected by choose_path() based on block-id contiguity
 * analysis (see ce_transfer.h CEPath enum):
 *
 *   CONTIG_DIRECT (0):     cpu_phys_contig && gpu_phys_contig && num_segments==1.
 *                         Single large cudaMemcpyAsync per (layer, kv_dim).
 *                         Optimal — O(1) API calls. No staging, no ping-pong.
 *
 *   SEGMENT_DIRECT (1): cpu_phys_contig (LAYERFIRST), few segments.
 *                         Per-segment cudaMemcpyAsync straight to final dst.
 *                         No staging, no ping-pong.
 *
 *   SEGMENT_SCATTER (2):    BLOCKFIRST + GPU contiguous, few segments.
 *                         Pinned staging buffer + merged segment memcpy + CPU
 *                         scatter/gather. Ping-pong: D2H = layer-level.
 *                         Pinned staging buffer + per-block memcpy + CPU
 *                         scatter/gather. Ping-pong disabled when both sides
 *                         non-contiguous (per-block granularity).
 *   ce_transfer_staged_scatter that branched on gpu_phys_contig at runtime;
 *   they are now split into ce_transfer_segment_scatter (merged segment memcpy,
 *   !gpu_phys_contig). Both share get_cached_hugepage_buffer /
 *   scatter_to_cpu / gather_from_cpu / get_cached_event_pair.
 *
 *   GATHER_SCATTER (4):  Many scattered segments (num_segments > threshold).
 *                         GPU index_select gather + single D2H/H2D + CPU
 *                         scatter (D2H) or GPU index_copy_ scatter (H2D).
 *                         Ping-pong: D2H = layer-level, H2D = disabled.
 *
 * Configuration is passed via CETransferConfig struct (from Python
 * GLOBAL_CONFIG_FROM_ENV), NOT read from environment variables directly.
 */
#include "ce_transfer.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <utility>
#include <unordered_map>
#include <array>

#include "monitoring/metrics_manager.h"

// FLEXKV_GPU_CPU_TRANSFER: metrics hook for transfer byte accounting.
// Defined as no-op when monitoring is disabled (FLEXKV_ENABLE_METRICS=0).
#ifndef FLEXKV_GPU_CPU_TRANSFER
#define FLEXKV_GPU_CPU_TRANSFER(is_h2d, size)
#endif

namespace flexkv {

// ---- Segment computation ----

CEAnalysis analyze_ce_transfer(
    const int64_t *gpu_block_ids, const int64_t *cpu_block_ids,
    int num_blocks, int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t gpu_block_stride_in_bytes) {
  CEAnalysis a;
  a.gpu_log_contig = true;
  a.cpu_log_contig = true;
  a.cpu_phys_contig = (cpu_block_stride_in_bytes == chunk_size_in_bytes);
  a.gpu_phys_contig = (gpu_block_stride_in_bytes == 0 ||
                       gpu_block_stride_in_bytes == chunk_size_in_bytes);
  a.num_segments = 0;

  if (num_blocks == 0) return a;

  a.num_segments = 1;
  int seg_start = 0;
  for (int k = 1; k < num_blocks; ++k) {
    bool src_step = (gpu_block_ids[k] == gpu_block_ids[k - 1] + 1);
    bool dst_step = (cpu_block_ids[k] == cpu_block_ids[k - 1] + 1);
    if (!src_step) a.gpu_log_contig = false;
    if (!dst_step) a.cpu_log_contig = false;
    if (!src_step || !dst_step) {
      a.segments.push_back({seg_start, k - seg_start});
      seg_start = k;
      a.num_segments++;
    }
  }
  a.segments.push_back({seg_start, num_blocks - seg_start});
  return a;
}

// ---- Path selection ----

// Select the optimized strategy (PER_BLOCK is handled upstream by the
// path_opt_enabled switch, not here). See the CEPath taxonomy in ce_transfer.h.
//
// This preserves the original numeric decision table exactly; it only splits
// the former "Path 1" into CONTIG_DIRECT (was Path 0) / SEGMENT_DIRECT (dst
// physically contiguous) / SEGMENT_SCATTER (dst strided, GPU contiguous) /
// self-describing name. GATHER_SCATTER (was Path 2) is unchanged.
CEPath choose_path(const CEAnalysis &a, const CETransferConfig &ce_config,
                   int64_t chunk_size_in_bytes) {
  // GATHER_DIRECT: BLOCKFIRST + CPU non-contiguous + GPU physically contiguous
  // (non-sharded). D2D transpose GPU LAYERFIRST -> dev_staging (contiguous),
  // then a direct per-segment memcpy to CPU. The compact-staging trick needs
  // the GPU block to be physically contiguous (full chunk per block) so the
  // staging block stride equals the CPU block stride. Sharded D2H shrinks the
  // GPU chunk to a shard (gpu_phys_contig=false), so a contiguous copy would
  // misplace every (layer,kv)>0 within each block (CPU BF stride is full_chunk,
  // staging stride is shard_size); route those to GATHER_SCATTER instead (it
  // CPU-scatters each shard to its exact offset). Checked before
  // !gpu_phys_contig but only when GPU is also contiguous.
  if (ce_config.is_blockfirst && !a.cpu_phys_contig && a.gpu_phys_contig)
    return CEPath::GATHER_DIRECT;

  // CONTIG_DIRECT: logical + physical contiguity on both sides -> one big memcpy.
  if (a.gpu_log_contig && a.cpu_log_contig && a.cpu_phys_contig && a.gpu_phys_contig)
    return CEPath::CONTIG_DIRECT;

  // Sharded D2H (LF + MLA sharded): !gpu_phys_contig (chunk shrunk to shard).
  // GATHER_SCATTER handles this via GPU index_select gather with proper
  // stride (same technique as GATHER_DIRECT). If chunk not 8-aligned
  // (extremely rare), fall back to SEGMENT_SCATTER.
  if (!a.gpu_phys_contig) {
    if (chunk_size_in_bytes > 0 && chunk_size_in_bytes % sizeof(int64_t) != 0)
      return CEPath::SEGMENT_SCATTER;
    return CEPath::GATHER_SCATTER;
  }

  // LAYERFIRST or BF MHA: remaining paths by contiguity.
  // LAYERFIRST non-MLA (!cpu_phys_contig && !is_blockfirst): SEGMENT_SCATTER
  // (strided is head-dimension, not layer-dimension; D2D transpose can't help).
  if (a.num_segments <= ce_config.segment_threshold) {
    return a.cpu_phys_contig ? CEPath::SEGMENT_DIRECT
                             : CEPath::SEGMENT_SCATTER;
  }
  // Many scattered segments (src contiguous) -> GPU gather/scatter pipeline.
  // GATHER_SCATTER uses index_select/index_copy_ which requires chunk_size
  // to be 8-byte aligned (int64 view). Fall back to SEGMENT_SCATTER if not.
  if (chunk_size_in_bytes > 0 && chunk_size_in_bytes % sizeof(int64_t) != 0)
    return CEPath::SEGMENT_SCATTER;
  return CEPath::GATHER_SCATTER;
}

// ---- Cached hugepage staging buffer (per current CUDA device) ----
//
// The cache is keyed by the *current* CUDA device, not just thread-local.
// TPTransferThreadGroup runs one thread per GPU (thread-local alone would be
// enough there), but LayerwiseTransferGroup iterates all GPUs in a SINGLE
// thread with cudaSetDevice() between them. A plain thread_local buffer/event
// created on device 0 would then be reused on device N, and a CUDA event is
// device-bound -> "invalid resource handle". Keying by current device makes
// both call patterns correct.

struct HostStagingBuf {
  void *buf = nullptr;
  size_t size = 0;

  ~HostStagingBuf() {
    if (buf) {
      cudaFreeHost(buf);
      buf = nullptr;
    }
  }
};

// Device staging buffer cache (same pattern as HostStagingBuf).
// Keyed by current CUDA device to handle LayerwiseTransferGroup's
// single-thread multi-GPU iteration pattern.
struct DeviceStagingBuf {
  void *buf = nullptr;
  size_t size = 0;

  ~DeviceStagingBuf() {
    if (buf) {
      cudaFree(buf);
      buf = nullptr;
    }
  }
};

void *get_cached_hugepage_buffer(size_t size) {
  int dev = 0;
  cudaGetDevice(&dev);
  thread_local std::unordered_map<int, HostStagingBuf> cache;
  HostStagingBuf &b = cache[dev];
  if (size > b.size) {
    if (b.buf) {
      cudaFreeHost(b.buf);
    }
    TORCH_CHECK(cudaSuccess == cudaMallocHost(&b.buf, size, cudaHostAllocDefault),
                "cudaMallocHost failed for cached hugepage buffer");
    b.size = size;
  }
  return b.buf;
}

// Get a cached device buffer of at least `size` bytes on the current device.
// Avoids per-call cudaMalloc/cudaFree overhead (nsys showed ~930ms of 2.6s total
// API time was spent in alloc/free of dev_buf, gpu_ids, dst_ids).
//
// `slot` allows callers to request independent buffers that must not alias
// (e.g. gpu_ids and dst_ids in GATHER_SCATTER are both alive simultaneously
// — using the same buffer would cause the second cudaMemcpyAsync to clobber
// the first). Default slot=0; callers that need a second independent buffer
// pass slot=1.
void *get_cached_device_buffer(size_t size, int slot) {
  int dev = 0;
  cudaGetDevice(&dev);
  thread_local std::unordered_map<int, std::array<DeviceStagingBuf, 3>> cache;
  DeviceStagingBuf &b = cache[dev][slot];
  if (size > b.size) {
    if (b.buf) {
      cudaFree(b.buf);
    }
    TORCH_CHECK(cudaSuccess == cudaMalloc(&b.buf, size),
                "cudaMalloc failed for cached device buffer");
    b.size = size;
  }
  return b.buf;
}

// Cached CUDA event pair (per device, thread_local).
// Events are created once on first use and NEVER destroyed per call.
// This avoids:
// 1. Per-call cudaEventCreate/Destroy overhead (~0.01ms each, adds up over
//    num_layers * kv_dim iterations).
// 2. cudaEventDestroy blocking when the stream hasn't been drained yet
//    (sync=false / polling mode). A cached event is just reused — the old
//    record is overwritten by the next cudaEventRecord.
//
// Safety: within each call, all events are sync'd (cudaEventSynchronize)
// before the staging buffers they guard are reused. At function return,
// the last event is sync'd (for pingpong_on) or the stream is sync'd
// (for pingpong_off / sync=true). So all GPU work is complete before the
// cached events are reused in the next call.
struct CachedEventPair {
  cudaEvent_t ev[2] = {nullptr, nullptr};
  bool created = false;
};

cudaEvent_t *get_cached_event_pair(bool need, bool &created) {
  if (!need) return nullptr;
  thread_local std::unordered_map<int, CachedEventPair> cache;
  int dev = 0;
  cudaGetDevice(&dev);
  CachedEventPair &e = cache[dev];
  if (!e.created) {
    TORCH_CHECK(cudaSuccess == cudaEventCreateWithFlags(&e.ev[0], cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed for cached event[0]");
    TORCH_CHECK(cudaSuccess == cudaEventCreateWithFlags(&e.ev[1], cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed for cached event[1]");
    e.created = true;
  }
  created = true;
  return e.ev;
}


// ============================================================================
// PER_BLOCK (baseline): per-block memcpy (no segment merging, no index_select,
//   no ping-pong). Used as microbenchmark baseline to quantify optimization
//   gains. Correct for all modes (sharded, BF, etc.) but slowest —
//   O(num_blocks) API calls. Selected when path_opt_enabled == false.
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
    cudaStream_t stream, bool is_host_to_device) {
  cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                          : cudaMemcpyDeviceToHost;
  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < kv_dim; j++) {
      int64_t *cpu_base =
          cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
          j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
      for (int b = 0; b < num_blocks; b++) {
        int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                        i + start_layer_id, j,
                                        gpu_block_ids[b]);
        int64_t *gpu_ptr_off =
            reinterpret_cast<int64_t *>(gpu_ptr) +
            gpu_startoff_inside_chunks_int64;
        int64_t *cpu_ptr_b =
            cpu_base + cpu_block_ids[b] * cpu_block_stride_int64;
        void *dst = is_host_to_device ? (void *)gpu_ptr_off : (void *)cpu_ptr_b;
        void *src = is_host_to_device ? (void *)cpu_ptr_b : (void *)gpu_ptr_off;
        cudaMemcpyAsync(dst, src, chunk_size_in_bytes, kind, stream);
        FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, chunk_size_in_bytes);
      }
    }
  }
}

// ============================================================================
// CONTIG_DIRECT: single large memcpy per (layer, kv_dim)
// ============================================================================

template <BackendType Type>
void ce_transfer_contig_direct(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device) {
  int64_t big_size = chunk_size_in_bytes * num_blocks;
  cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                          : cudaMemcpyDeviceToHost;
  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < kv_dim; j++) {
      int64_t *cpu_chunk_ptr =
          cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
          j * cpu_kv_stride_int64 +
          cpu_block_ids[0] * cpu_block_stride_int64 +
          cpu_startoff_inside_chunks_int64;
      int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler, i + start_layer_id,
                                      j, gpu_block_ids[0]);
      int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                               gpu_startoff_inside_chunks_int64;
      void *dst = is_host_to_device ? (void *)gpu_chunk_ptr
                                    : (void *)cpu_chunk_ptr;
      void *src = is_host_to_device ? (void *)cpu_chunk_ptr
                                    : (void *)gpu_chunk_ptr;
      cudaMemcpyAsync(dst, src, big_size, kind, stream);
      FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, big_size);
    }
  }
}

// ============================================================================
// SEGMENT_DIRECT: per-merged-run memcpy straight between CPU and GPU.
//   dst physically contiguous (LAYERFIRST + non-sharded), so no staging is
//   needed and ping-pong does not apply. One cudaMemcpyAsync per contiguous
//   run of blocks, for each (layer, kv).
// ============================================================================

template <BackendType Type>
void ce_transfer_segment_direct(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &ce_config) {
  (void)ce_config;  // ping-pong / staging config unused: this path never stages.
  cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                          : cudaMemcpyDeviceToHost;
  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < kv_dim; j++) {
      for (const auto &seg : analysis.segments) {
        int64_t seg_size = (int64_t)seg.run_len * chunk_size_in_bytes;
        int64_t *cpu_ptr =
            cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
            j * cpu_kv_stride_int64 +
            cpu_block_ids[seg.start_k] * cpu_block_stride_int64 +
            cpu_startoff_inside_chunks_int64;
        int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                        i + start_layer_id, j,
                                        gpu_block_ids[seg.start_k]);
        int64_t *gpu_ptr_off =
            reinterpret_cast<int64_t *>(gpu_ptr) +
            gpu_startoff_inside_chunks_int64;
        void *dst = is_host_to_device ? (void *)gpu_ptr_off
                                      : (void *)cpu_ptr;
        void *src = is_host_to_device ? (void *)cpu_ptr
                                      : (void *)gpu_ptr_off;
        cudaMemcpyAsync(dst, src, seg_size, kind, stream);
        FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, seg_size);
      }
    }
  }
}

// ============================================================================
// Ping-pong scope (D2H double-buffering of the host staging buffer)
// ============================================================================
//
// D2H ping-pong exists ONLY in:
//   - SEGMENT_SCATTER (always enabled for D2H)
//       is_per_block = !gpu_phys_contig && !cpu_phys_contig, disabled then)
//   - GATHER_SCATTER
//       need_pingpong_host = need_host_buf && !is_host_to_device
//
// Ping-pong is NOT in:
//   - CONTIG_DIRECT / SEGMENT_DIRECT (no staging buffer at all)
//   - GATHER_DIRECT (no staging after transpose -- direct per-segment memcpy)
//   - scatter_to_cpu / gather_from_cpu themselves (ping-pong wraps AROUND them
//     in the main per-(layer,kv) loop, not inside the scatter/gather function)

// ============================================================================
// scatter_to_cpu: scatter from contiguous staging buffer to strided CPU dst.
//   When cpu_phys_contig (LAYERFIRST), consecutive cpu_block_ids are also
//   physically adjacent, so we merge them into a single memcpy. When
//   !cpu_phys_contig (BLOCKFIRST), consecutive block_ids have a stride gap
//   between them, so each block must be scattered individually.
// ============================================================================
void scatter_to_cpu(const void *staging_buf, int64_t *cpu_ptr_int64,
                    int64_t *cpu_block_ids, int num_blocks,
                    int64_t cpu_block_stride_int64,
                    int64_t cpu_startoff_inside_chunks_int64,
                    int64_t chunk_size_in_bytes, int layer_idx, int kv_idx,
                    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
                    int start_layer_id, bool cpu_phys_contig) {
  int64_t *cpu_base = cpu_ptr_int64 +
      (layer_idx + start_layer_id) * cpu_layer_stride_int64 +
      kv_idx * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
  int64_t k = 0;
  while (k < num_blocks) {
    int64_t run_start = k;
    if (cpu_phys_contig) {
      // LAYERFIRST: consecutive block_ids are physically adjacent -- merge.
      while (k + 1 < num_blocks &&
             cpu_block_ids[k + 1] == cpu_block_ids[k] + 1) {
        ++k;
      }
      int64_t run_len = k - run_start + 1;
      int64_t cb = cpu_block_ids[run_start];
      int64_t run_bytes = run_len * chunk_size_in_bytes;
      memcpy(cpu_base + cb * cpu_block_stride_int64,
             (const char *)staging_buf +
                 (int64_t)run_start * chunk_size_in_bytes,
             run_bytes);
    } else {
      // BLOCKFIRST: each block is at a strided position -- scatter individually.
      int64_t cb = cpu_block_ids[k];
      memcpy(cpu_base + cb * cpu_block_stride_int64,
             (const char *)staging_buf +
                 (int64_t)k * chunk_size_in_bytes,
             chunk_size_in_bytes);
    }
    ++k;
  }
}

// ============================================================================
// gather_from_cpu: gather from strided CPU positions to contiguous staging buf.
//   H2D symmetric counterpart of scatter_to_cpu (src/dst swapped, const-ness
//   adjusted). When cpu_phys_contig (LAYERFIRST), consecutive cpu_block_ids
//   are also physically adjacent, so we merge them into a single memcpy. When
//   !cpu_phys_contig (BLOCKFIRST), consecutive block_ids have a stride gap
//   between them, so each block must be gathered individually.
// ============================================================================
void gather_from_cpu(void *staging_buf, const int64_t *cpu_ptr_int64,
                     const int64_t *cpu_block_ids, int num_blocks,
                     int64_t cpu_block_stride_int64,
                     int64_t cpu_startoff_inside_chunks_int64,
                     int64_t chunk_size_in_bytes, int layer_idx, int kv_idx,
                     int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
                     int start_layer_id, bool cpu_phys_contig) {
  const int64_t *cpu_base = cpu_ptr_int64 +
      (layer_idx + start_layer_id) * cpu_layer_stride_int64 +
      kv_idx * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
  int64_t k = 0;
  while (k < num_blocks) {
    int64_t run_start = k;
    if (cpu_phys_contig) {
      // LAYERFIRST: consecutive block_ids are physically adjacent -- merge.
      while (k + 1 < num_blocks &&
             cpu_block_ids[k + 1] == cpu_block_ids[k] + 1) {
        ++k;
      }
      int64_t run_len = k - run_start + 1;
      int64_t cb = cpu_block_ids[run_start];
      int64_t run_bytes = run_len * chunk_size_in_bytes;
      memcpy((char *)staging_buf +
                 (int64_t)run_start * chunk_size_in_bytes,
             cpu_base + cb * cpu_block_stride_int64,
             run_bytes);
    } else {
      // BLOCKFIRST: each block is at a strided position -- gather individually.
      int64_t cb = cpu_block_ids[k];
      memcpy((char *)staging_buf +
                 (int64_t)k * chunk_size_in_bytes,
             cpu_base + cb * cpu_block_stride_int64,
             chunk_size_in_bytes);
    }
    ++k;
  }
}

// ============================================================================
// SEGMENT_SCATTER: pinned staging buffer + merged-segment memcpy + CPU
//   scatter/gather to a strided destination (BLOCKFIRST, or any strided CPU
//   layout with GPU contiguous). Requires gpu_phys_contig (GPU block stride ==
//   chunk_size) so consecutive GPU blocks within a segment are physically
//   adjacent -> one cudaMemcpyAsync per contiguous run of block ids.
//   Ping-pong: D2H layer-level (always enabled for D2H; is_per_block is
//   always false since gpu_phys_contig is true).
//   enable_memcpy2d branch (at entry): when true, use cudaMemcpy2DAsync per
//   segment (strided GPU<->CPU directly, bypassing staging + scatter/gather).
//   Fast on NVIDIA (DMA engine handles 2D), very slow on P800/Kunlunxin.
// ============================================================================

template <BackendType Type>
void ce_transfer_segment_scatter(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &ce_config) {

  // ---- memcpy2d branch (bidirectional, NVIDIA-specific optimization) ----
  // When enable_memcpy2d=TRUE, use cudaMemcpy2DAsync per segment to do a
  // strided GPU<->CPU transfer directly, bypassing the staging buffer +
  // sync + CPU scatter/gather. Fast on NVIDIA (DMA engine handles 2D),
  // extremely slow on P800/Kunlunxin. Default off (FLEXKV_ENABLE_MEMCPY2D=0).
  // Direction (is_host_to_device) selects src/dst/pitch/kind:
  //   D2H: GPU src (contiguous within segment) -> CPU dst (strided)
  //   H2D: CPU src (strided) -> GPU dst (contiguous within segment)
  if (ce_config.enable_memcpy2d) {
    cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                            : cudaMemcpyDeviceToHost;
    const int64_t total_iters = (int64_t)num_layers * kv_dim;
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);
      for (const auto &seg : analysis.segments) {
        // GPU pointer: ptr_at for the first block in this segment. Within a
        // segment gpu_block_ids are contiguous (step=1), so the GPU block
        // stride (pitch) is the pointer diff of two adjacent blocks.
        int64_t *gpu_ptr_first = ptr_at<Type>(gpu_tensor_handler,
                                              i + start_layer_id, j,
                                              gpu_block_ids[seg.start_k]);
        int64_t *gpu_ptr_next = ptr_at<Type>(gpu_tensor_handler,
                                             i + start_layer_id, j,
                                             gpu_block_ids[seg.start_k] + 1);
        size_t gpu_pitch = (size_t)((char *)gpu_ptr_next - (char *)gpu_ptr_first);
        void *gpu_ptr = (char *)gpu_ptr_first +
            gpu_startoff_inside_chunks_int64 * sizeof(int64_t);
        // CPU pointer: strided, first block in segment.
        int64_t *cpu_base = cpu_ptr_int64 +
            (i + start_layer_id) * cpu_layer_stride_int64 +
            j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
        void *cpu_ptr = cpu_base + cpu_block_ids[seg.start_k] * cpu_block_stride_int64;
        size_t cpu_pitch = (size_t)cpu_block_stride_int64 * sizeof(int64_t);

        // Select src/dst/pitch by direction.
        void *dst = is_host_to_device ? gpu_ptr : cpu_ptr;
        void *src = is_host_to_device ? cpu_ptr : gpu_ptr;
        size_t dpitch = is_host_to_device ? gpu_pitch : cpu_pitch;
        size_t spitch = is_host_to_device ? cpu_pitch : gpu_pitch;

        cudaMemcpy2DAsync(dst, dpitch, src, spitch,
                          chunk_size_in_bytes, seg.run_len, kind, stream);
        FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, chunk_size_in_bytes * seg.run_len);
      }
    }
    cudaStreamSynchronize(stream);
    return;
  }

  // ---- staging buffer + CPU scatter/gather ----
  // SEGMENT_SCATTER: gpu_phys_contig is true (required by choose_path), so
  // is_per_block is always false. Ping-pong enabled for all D2H.
  size_t layer_buf_size = (size_t)num_blocks * chunk_size_in_bytes;
  bool need_pingpong = !is_host_to_device;

  void *host_base = get_cached_hugepage_buffer(need_pingpong ? layer_buf_size * 2
                                                       : layer_buf_size);
  void *host_bufs[2] = {
      host_base,
      need_pingpong ? (char *)host_base + layer_buf_size : nullptr};
  // Cached ping-pong events (per device, thread_local -- see get_cached_event_pair).
  bool events_created = false;
  cudaEvent_t *pingpong_events = get_cached_event_pair(need_pingpong, events_created);

  const int64_t total_iters = (int64_t)num_layers * kv_dim;
  for (int64_t it = 0; it < total_iters; ++it) {
    int i = (int)(it / kv_dim);
    int j = (int)(it % kv_dim);
    int idx = need_pingpong ? (int)(it & 1) : 0;
    int prev_idx = idx ^ 1;
    void *buf = host_bufs[idx];

    if (!is_host_to_device) {
      // ---- D2H ----
      // D2H all segments into staging (merged segment memcpy: gpu_phys_contig)
      int64_t seg_offset = 0;
      for (const auto &seg : analysis.segments) {
        int64_t seg_size = (int64_t)seg.run_len * chunk_size_in_bytes;
        int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                        i + start_layer_id, j,
                                        gpu_block_ids[seg.start_k]);
        int64_t *gpu_ptr_off =
            reinterpret_cast<int64_t *>(gpu_ptr) +
            gpu_startoff_inside_chunks_int64;
        cudaMemcpyAsync((char *)buf + seg_offset, gpu_ptr_off, seg_size,
                        cudaMemcpyDeviceToHost, stream);
        FLEXKV_GPU_CPU_TRANSFER(false, seg_size);
        seg_offset += seg_size;
      }
      if (need_pingpong) {
        cudaEventRecord(pingpong_events[idx], stream);
        // CPU scatter previous layer
        if (it >= 1) {
          cudaEventSynchronize(pingpong_events[prev_idx]);
          int pi = (int)((it - 1) / kv_dim);
          int pj = (int)((it - 1) % kv_dim);
          // scatter from host_bufs[prev_idx] to strided dst
          scatter_to_cpu(host_bufs[prev_idx], cpu_ptr_int64,
                         cpu_block_ids, num_blocks,
                         cpu_block_stride_int64,
                         cpu_startoff_inside_chunks_int64,
                         chunk_size_in_bytes, pi, pj,
                         cpu_kv_stride_int64, cpu_layer_stride_int64,
                         start_layer_id, analysis.cpu_phys_contig);
        }
      } else {
        cudaStreamSynchronize(stream);
        // scatter current layer
        scatter_to_cpu(buf, cpu_ptr_int64,
                       cpu_block_ids, num_blocks,
                       cpu_block_stride_int64,
                       cpu_startoff_inside_chunks_int64,
                       chunk_size_in_bytes, i, j,
                       cpu_kv_stride_int64, cpu_layer_stride_int64,
                       start_layer_id, analysis.cpu_phys_contig);
      }
    } else {
      // ---- H2D (no ping-pong: CPU gather is too fast to benefit) ----
      // Gather all segments into buf, then H2D all, then drain.
      // `buf` is pinned to host_bufs[0] and reused every iteration.
      gather_from_cpu(buf, cpu_ptr_int64,
                      cpu_block_ids, num_blocks,
                      cpu_block_stride_int64,
                      cpu_startoff_inside_chunks_int64,
                      chunk_size_in_bytes, i, j,
                      cpu_kv_stride_int64, cpu_layer_stride_int64,
                      start_layer_id, analysis.cpu_phys_contig);
      // H2D all segments from staging (merged segment memcpy: gpu_phys_contig)
      int64_t off = 0;
      for (const auto &seg : analysis.segments) {
        int64_t seg_size = (int64_t)seg.run_len * chunk_size_in_bytes;
        int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                        i + start_layer_id, j,
                                        gpu_block_ids[seg.start_k]);
        int64_t *gpu_ptr_off =
            reinterpret_cast<int64_t *>(gpu_ptr) +
            gpu_startoff_inside_chunks_int64;
        cudaMemcpyAsync(gpu_ptr_off, (char *)buf + off, seg_size,
                        cudaMemcpyHostToDevice, stream);
        FLEXKV_GPU_CPU_TRANSFER(true, seg_size);
        off += seg_size;
      }
      // The async H2D memcpy's above are still reading `buf` when the
      // next iteration's CPU gather overwrites it. Drain the stream so
      // `buf` is safe to overwrite next iteration.
      cudaStreamSynchronize(stream);
    }
  }
  // Drain last ping-pong slot (D2H)
  if (!is_host_to_device && need_pingpong && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    cudaEventSynchronize(pingpong_events[last_idx]);
    int li = (int)(last / kv_dim);
    int lj = (int)(last % kv_dim);
    scatter_to_cpu(host_bufs[last_idx], cpu_ptr_int64,
                   cpu_block_ids, num_blocks,
                   cpu_block_stride_int64,
                   cpu_startoff_inside_chunks_int64,
                   chunk_size_in_bytes, li, lj,
                   cpu_kv_stride_int64, cpu_layer_stride_int64,
                   start_layer_id, analysis.cpu_phys_contig);
  }
  // NOTE: ping-pong events are cached (get_cached_event_pair) and NOT
  // destroyed here. All GPU work has been sync'd within the loop (via
  // per-slot cudaEventSynchronize) and at the end (via final flush or
  // cudaStreamSynchronize), so the events are safe to reuse in the next call.
}

// ============================================================================
// GATHER_SCATTER: GPU index_select/index_copy_ pipeline through a staging
//   buffer, for many scattered segments. Uses staging (ping-pong applies).
//    D2H: GPU index_select gather -> D2H staging -> CPU scatter
//    H2D: CPU gather -> H2D staging -> GPU index_copy_ scatter
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
    const CEAnalysis &analysis, const CETransferConfig &ce_config) {
  TORCH_CHECK(chunk_size_in_bytes % sizeof(int64_t) == 0,
              "Path 2 requires chunk_size_in_bytes % 8 == 0");
  const int64_t elems_per_block = chunk_size_in_bytes / sizeof(int64_t);
  // buf_bytes = total bytes for all blocks' staging buffer.
  // elems_per_block * sizeof(int64_t) == chunk_size_in_bytes, so this is
  // equivalent to num_blocks * chunk_size_in_bytes — kept as one variable.
  const size_t buf_bytes = (size_t)num_blocks * (size_t)chunk_size_in_bytes;

  // Bind ATen to our cuda stream
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  c10::cuda::CUDAStream aten_stream =
      c10::cuda::getStreamFromExternal(stream, cur_dev);
  c10::cuda::CUDAStreamGuard stream_guard(aten_stream);

  // Find max GPU block index for tensor views
  int64_t max_gpu_id = 0;
  for (int k = 0; k < num_blocks; ++k) {
    if (gpu_block_ids[k] > max_gpu_id) max_gpu_id = gpu_block_ids[k];
  }

  // IMPORTANT: pin the CUDA TensorOptions to cur_dev explicitly. A bare
  // at::kCUDA uses ATen's "current device", which is unreliable in the
  // LayerwiseTransferGroup path (single thread iterating GPUs via
  // cudaSetDevice). from_blob()/index_select on the wrong device index then
  // touches another device's memory -> segfault. Binding to cur_dev makes the
  // views/gather/scatter target the same device as the raw GPU pointers.
  auto i64_cuda = at::TensorOptions().dtype(at::kLong)
                      .device(at::kCUDA, cur_dev);

  // Transfer block ids to GPU (for index_select / index_copy_).
  // Needed when GPU blocks are non-contiguous — either logically
  // (gpu_log_contig=false, scattered block ids) or physically
  // (gpu_phys_contig=false, sharded D2H with stride gap between blocks).
  // Use cached device buffers (not per-call cudaMalloc/cudaFree) so that
  // in sync=false (async/layerwise polling) mode we can return WITHOUT
  // draining the stream — the GPU may still be reading these buffers
  // asynchronously. Per-call cudaFree would be a use-after-free; a cached
  // buffer survives across calls and is reused next time.
  const size_t ids_bytes = (size_t)num_blocks * sizeof(int64_t);
  void *gpu_ids_raw = nullptr;
  at::Tensor gpu_ids_cuda;
  void *dst_ids_raw = nullptr;
  at::Tensor dst_ids_cuda;
  if (!analysis.gpu_log_contig || !analysis.gpu_phys_contig) {
    gpu_ids_raw = get_cached_device_buffer(ids_bytes);
    cudaMemcpyAsync(gpu_ids_raw, gpu_block_ids, ids_bytes,
                    cudaMemcpyHostToDevice, stream);
    gpu_ids_cuda = at::from_blob(gpu_ids_raw, {num_blocks}, i64_cuda);

    if (is_host_to_device) {
      dst_ids_raw = get_cached_device_buffer(ids_bytes, 1);  // slot=1: independent from gpu_ids
      cudaMemcpyAsync(dst_ids_raw, gpu_block_ids, ids_bytes,
                      cudaMemcpyHostToDevice, stream);
      dst_ids_cuda = at::from_blob(dst_ids_raw, {num_blocks}, i64_cuda);
    }
  }

  // Allocate ping-pong device buffers.
  //
  // CRITICAL: these staging buffers must NOT come from ATen's caching
  // allocators (at::empty on CUDA / pinned CPU). We run every ATen op below
  // under a CUDAStreamGuard wrapping the CALLER's external stream. ATen's
  // caching allocators then record_stream() that external stream onto the
  // buffer's block; when the buffer is later freed (during a subsequent
  // test's GC) ATen replays cudaEventRecord() on that external stream -- but
  // the caller (LayerwiseTransferGroup) has since destroyed it, so the record
  // crashes deep inside ~StorageImpl. (Confirmed via native backtrace:
  // ~StorageImpl -> CachingHostAllocatorImpl::free -> record_stream ->
  // CUDAEvent::record -> cuEventRecordWithFlags on a dead stream.)
  //
  // Use raw cudaMalloc / cudaMallocHost and wrap them in from_blob views
  // (which own no storage, so ATen never record_stream/free them). We free
  // the raw memory ourselves before returning.
  // Device buffer needed for GPU-side gather/scatter when GPU block IDs
  // are non-contiguous (gpu_log_contig=false) OR GPU blocks are physically
  // non-contiguous (gpu_phys_contig=false, sharded D2H). In both cases
  // index_select/index_copy_ is used to gather/scatter through dev_buf.
  // For ping-pong, allocate 2x and split into two halves (each call to
  // get_cached_device_buffer returns the SAME pointer, so calling it twice
  // would give two views of the same memory — breaking ping-pong).
  bool need_dev_buf = !analysis.gpu_log_contig || !analysis.gpu_phys_contig;
  void *dev_raw[2] = {nullptr, nullptr};
  at::Tensor dev_buf[2];
  if (need_dev_buf) {
    bool need_two = !is_host_to_device;  // D2H ping-pong only
    size_t dev_alloc = need_two ? buf_bytes * 2 : buf_bytes;
    void *dev_base = get_cached_device_buffer(dev_alloc, 2);  // slot=2: dev_buf (independent from gpu_ids/dst_ids)
    dev_raw[0] = dev_base;
    dev_raw[1] = need_two ? (char *)dev_base + buf_bytes : dev_base;
    dev_buf[0] = at::from_blob(dev_raw[0], {num_blocks, elems_per_block},
                               i64_cuda);
    dev_buf[1] = at::from_blob(dev_raw[1], {num_blocks, elems_per_block},
                               i64_cuda);
  }

  // Host staging buffer needed:
  // - D2H: always (staging + CPU scatter to strided dst)
  // - H2D: when CPU src is non-contiguous (cpu_log_contig = CPU side)
  bool need_host_buf =
      !is_host_to_device ||  // D2H: always stage then scatter
      (is_host_to_device && !analysis.cpu_log_contig);  // H2D: CPU gather needed

  // D2H ping-pong: CPU scatter overlaps with GPU D2H.
  // H2D has no ping-pong (CPU gather too fast).
  bool need_pingpong_host = need_host_buf && !is_host_to_device;

  void *host_buf[2] = {nullptr, nullptr};
  if (need_host_buf) {
    // For ping-pong, allocate 2x size and split into two halves.
    // For non-pingpong, allocate 1x size.
    size_t host_alloc = need_pingpong_host ? buf_bytes * 2 : buf_bytes;
    void *host_base = get_cached_hugepage_buffer(host_alloc);
    host_buf[0] = host_base;
    if (need_pingpong_host) {
      host_buf[1] = (char *)host_base + buf_bytes;
    }
  }

  // Cached ping-pong events (per device, thread_local -- see get_cached_event_pair).
  bool events_created = false;
  cudaEvent_t *pingpong_events = get_cached_event_pair(need_pingpong_host, events_created);

  const int64_t total_iters = (int64_t)num_layers * kv_dim;

  for (int64_t it = 0; it < total_iters; ++it) {
    int i = (int)(it / kv_dim);
    int j = (int)(it % kv_dim);
    int idx = pingpong_events ? (int)(it & 1) : 0;
    int prev_idx = idx ^ 1;

    // GPU block stride (pitch) in int64 elements. Non-sharded: equals
    // elems_per_block (contiguous). Sharded D2H: full_chunk stride while
    // elems_per_block = shard_size/8, so from_blob needs an explicit stride.
    int64_t *gpu_ptr_block0 =
        ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);
    int64_t *gpu_ptr_block1 =
        ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 1);
    int64_t gpu_block_stride_elems =
        (int64_t)((char *)gpu_ptr_block1 - (char *)gpu_ptr_block0) /
        sizeof(int64_t);
    int64_t *gpu_layer_kv_base =
        gpu_ptr_block0 + gpu_startoff_inside_chunks_int64;

    if (!is_host_to_device) {
      // ============ D2H ============
      // Step 1: GPU gather (if src non-contig — logical or physical)
      const int64_t *d2h_src;
      if (analysis.gpu_log_contig && analysis.gpu_phys_contig) {
        d2h_src = gpu_layer_kv_base +
                  gpu_block_ids[0] * gpu_block_stride_elems;
      } else {
        at::Tensor src_view = at::from_blob(
            gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
            {gpu_block_stride_elems, 1}, i64_cuda);
        at::index_select_out(dev_buf[idx], src_view, 0, gpu_ids_cuda);
        d2h_src = reinterpret_cast<int64_t *>(dev_buf[idx].data_ptr());
      }

      // Step 2: D2H into staging
      void *dst_ptr = need_host_buf ? host_buf[idx]
                                   : (void *)(cpu_ptr_int64 +
                                      (i + start_layer_id) * cpu_layer_stride_int64 +
                                      j * cpu_kv_stride_int64 +
                                      cpu_block_ids[0] * cpu_block_stride_int64 +
                                      cpu_startoff_inside_chunks_int64);
      cudaMemcpyAsync(dst_ptr, d2h_src, buf_bytes,
                      cudaMemcpyDeviceToHost, stream);
      FLEXKV_GPU_CPU_TRANSFER(false, buf_bytes);

      if (pingpong_events) {
        cudaEventRecord(pingpong_events[idx], stream);
        // Step 3: CPU scatter previous slot
        if (it >= 1) {
          cudaEventSynchronize(pingpong_events[prev_idx]);
          int pi = (int)((it - 1) / kv_dim);
          int pj = (int)((it - 1) % kv_dim);
          scatter_to_cpu(host_buf[prev_idx], cpu_ptr_int64,
                         cpu_block_ids, num_blocks,
                         cpu_block_stride_int64,
                         cpu_startoff_inside_chunks_int64,
                         chunk_size_in_bytes, pi, pj,
                         cpu_kv_stride_int64, cpu_layer_stride_int64,
                         start_layer_id, analysis.cpu_phys_contig);
        }
      } else if (need_host_buf) {
        cudaStreamSynchronize(stream);
        // scatter current
        scatter_to_cpu(host_buf[idx], cpu_ptr_int64,
                       cpu_block_ids, num_blocks,
                       cpu_block_stride_int64,
                       cpu_startoff_inside_chunks_int64,
                       chunk_size_in_bytes, i, j,
                       cpu_kv_stride_int64, cpu_layer_stride_int64,
                       start_layer_id, analysis.cpu_phys_contig);
      }
    } else {
      // ============ H2D ============
      // In H2D: actual src = CPU, actual dst = GPU. The CEAnalysis naming is
      // direction-agnostic (gpu_* = GPU side, cpu_* = CPU side), so no mental
      // swap is needed — just use the correct side's flags.
      //
      // Step 1: CPU gather (if CPU src non-contig)
      const void *h2d_src;
      if (analysis.cpu_log_contig && analysis.cpu_phys_contig) {
        // CPU src is contiguous — direct from cpu_ptr, no staging needed.
        h2d_src = cpu_ptr_int64 +
                  (i + start_layer_id) * cpu_layer_stride_int64 +
                  j * cpu_kv_stride_int64 +
                  cpu_block_ids[0] * cpu_block_stride_int64 +
                  cpu_startoff_inside_chunks_int64;
      } else {
        // gather into staging (H2D: no ping-pong, idx always 0)
        gather_from_cpu(host_buf[idx], cpu_ptr_int64,
                        cpu_block_ids, num_blocks,
                        cpu_block_stride_int64,
                        cpu_startoff_inside_chunks_int64,
                        chunk_size_in_bytes, i, j,
                        cpu_kv_stride_int64, cpu_layer_stride_int64,
                        start_layer_id, analysis.cpu_phys_contig);
        h2d_src = host_buf[idx];
      }

      // Step 2: H2D — GPU dst contiguity (logical + physical)
      void *h2d_dst;
      if (analysis.gpu_log_contig && analysis.gpu_phys_contig) {
        h2d_dst = gpu_layer_kv_base +
                  gpu_block_ids[0] * gpu_block_stride_elems;
      } else {
        h2d_dst = dev_buf[idx].data_ptr();
      }
      cudaMemcpyAsync(h2d_dst, h2d_src, buf_bytes,
                      cudaMemcpyHostToDevice, stream);
      FLEXKV_GPU_CPU_TRANSFER(true, buf_bytes);

      // Step 3: GPU scatter (if GPU dst non-contig — logical or physical)
      if (!analysis.gpu_log_contig || !analysis.gpu_phys_contig) {
        at::Tensor dst_view = at::from_blob(
            gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
            {gpu_block_stride_elems, 1}, i64_cuda);
        dst_view.index_copy_(0, dst_ids_cuda, dev_buf[idx]);
      }

      // H2D: no ping-pong. idx is pinned to 0, so host_buf[0]/dev_buf[0] are
      // reused every iteration. Drain the stream so staging buffers are safe
      // to overwrite before the next iteration touches them.
      if (need_host_buf || need_dev_buf) {
        cudaStreamSynchronize(stream);
      }
    }
  }

  // Drain last D2H scatter
  if (!is_host_to_device && pingpong_events && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    cudaEventSynchronize(pingpong_events[last_idx]);
    int li = (int)(last / kv_dim);
    int lj = (int)(last % kv_dim);
    scatter_to_cpu(host_buf[last_idx], cpu_ptr_int64,
                   cpu_block_ids, num_blocks,
                   cpu_block_stride_int64,
                   cpu_startoff_inside_chunks_int64,
                   chunk_size_in_bytes, li, lj,
                   cpu_kv_stride_int64, cpu_layer_stride_int64,
                   start_layer_id, analysis.cpu_phys_contig);
  }

  // Drain last H2D
  if (is_host_to_device && pingpong_events && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    cudaEventSynchronize(pingpong_events[last_idx]);
  }

  // Drain the stream before returning. The staging buffers (dev_buf,
  // host_buf) are cached and survive across calls, but the per-call
  // id tensors (gpu_ids_raw, dst_ids_raw) are freed on return — any
  // in-flight async op referencing them would read/write freed memory.
  // D2H with ping-pong already drained the last scatter above; H2D
  // already drained via per-iteration cudaStreamSynchronize. But we
  // still need a final drain for the D2H non-pingpong case (when
  // pingpong_events is null but need_host_buf is true).
  cudaStreamSynchronize(stream);

  // NOTE: ping-pong events are cached (get_cached_event_pair) and NOT
  // destroyed here. All GPU work has been sync'd within the loop (via
  // per-slot cudaEventSynchronize) and at the end (via final drain).

  // Release the from_blob views. The underlying buffers (dev_buf, host_buf,
  // gpu_ids_raw, dst_ids_raw) are all cached and survive across calls.
  gpu_ids_cuda.reset();
  dst_ids_cuda.reset();
  dev_buf[0].reset();
  dev_buf[1].reset();
}

// ============================================================================
// GATHER_DIRECT: BF non-sharded (any mla mode / MHA) D2H/H2D.
//   D2D transpose (LAYERFIRST→BLOCKFIRST) via index_select + transpose +
//   contiguous, then per-segment cudaMemcpyAsync matching the transposed
//   BLOCKFIRST layout. Works for both kv_dim=1 (MLA) and kv_dim=2 (MHA):
//   staging [num_blocks, total_iters, elems] matches CPU [num_blocks, num_layers,
//   kv_dim, chunk] because total_iters = num_layers * kv_dim interleaves kv
//   within each layer — same as CPU's layer_stride/kv_stride layout.
// ============================================================================

template <BackendType Type>
void ce_transfer_gather_direct(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &ce_config) {
  TORCH_CHECK(chunk_size_in_bytes % sizeof(int64_t) == 0,
              "GATHER_DIRECT requires chunk_size % 8 == 0");
  const int64_t elems_per_block = chunk_size_in_bytes / sizeof(int64_t);
  const size_t buf_bytes = (size_t)num_blocks * (size_t)chunk_size_in_bytes;
  const int64_t total_iters = (int64_t)num_layers * kv_dim;
  // Device staging: [num_blocks, total_iters, elems_per_block] (BLOCKFIRST)
  const size_t total_dev_bytes = buf_bytes * (size_t)total_iters;

  // Bind ATen to our cuda stream
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  c10::cuda::CUDAStream aten_stream =
      c10::cuda::getStreamFromExternal(stream, cur_dev);
  c10::cuda::CUDAStreamGuard stream_guard(aten_stream);
  auto i64_cuda = at::TensorOptions().dtype(at::kLong)
                      .device(at::kCUDA, cur_dev);

  // Find max GPU block index for tensor views
  int64_t max_gpu_id = 0;
  for (int k = 0; k < num_blocks; ++k)
    if (gpu_block_ids[k] > max_gpu_id) max_gpu_id = gpu_block_ids[k];

  // Transfer block ids to GPU (needed for index_select when !gpu_log_contig)
  const size_t ids_bytes = (size_t)num_blocks * sizeof(int64_t);
  void *gpu_ids_raw = nullptr;
  at::Tensor gpu_ids_cuda;
  if (!analysis.gpu_log_contig) {
    gpu_ids_raw = get_cached_device_buffer(ids_bytes);
    cudaMemcpyAsync(gpu_ids_raw, gpu_block_ids, ids_bytes,
                    cudaMemcpyHostToDevice, stream);
    gpu_ids_cuda = at::from_blob(gpu_ids_raw, {num_blocks}, i64_cuda);
  }

  // Device staging buffer: [num_blocks, total_iters, elems_per_block] contiguous
  void *dev_staging = get_cached_device_buffer(total_dev_bytes, 2);
  at::Tensor dev_staging_view = at::from_blob(
      dev_staging, {num_blocks, total_iters, elems_per_block}, i64_cuda);

  if (!is_host_to_device) {
    // ============ D2H ============
    // Step 1: D2D transpose — gather each (layer, kv) into staging
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);
      int64_t *gpu_ptr_block0 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);
      int64_t *gpu_ptr_block1 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 1);
      // GPU block stride (pitch) in int64 elements. Non-sharded: equals
      // elems_per_block (contiguous). Sharded D2H: full_chunk stride while
      // elems_per_block = shard_size/8, so from_blob needs an explicit stride.
      int64_t gpu_block_stride_elems =
          (int64_t)((char *)gpu_ptr_block1 - (char *)gpu_ptr_block0) /
          sizeof(int64_t);
      // Add gpu_startoff to land on this rank's shard (sharded D2H).
      int64_t *gpu_layer_kv_base =
          gpu_ptr_block0 + gpu_startoff_inside_chunks_int64;
      at::Tensor src_view = at::from_blob(
          gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
          {gpu_block_stride_elems, 1}, i64_cuda);
      // Gather directly into staging[:, it, :] via index_select_out
      // (avoids a temporary tensor + copy_).
      auto staging_slice = dev_staging_view.select(1, it);
      if (analysis.gpu_log_contig) {
        staging_slice.copy_(src_view.narrow(0, gpu_block_ids[0], num_blocks));
      } else {
        at::index_select_out(staging_slice, src_view, 0, gpu_ids_cuda);
      }
    }

    // Step 2: D2H per-segment (staging layout matches CPU BLOCKFIRST)
    // staging [num_blocks, total_iters, elems] == CPU [num_blocks, num_layers,
    // kv_dim, chunk] because total_iters interleaves kv within layers.
    // block_stride in staging = total_iters * chunk_size = cpu_block_stride.
    int64_t block_bytes = total_iters * chunk_size_in_bytes;
    // When all layers are transferred at once (total_iters = num_layers*kv_dim
    // and block_bytes == cpu_block_stride), a contiguous per-segment copy of
    // num_blocks*block_bytes is layout-correct. But layerwise_transfer calls
    // this with num_layers=1 per batch, so block_bytes != cpu_block_stride and
    // a contiguous copy would cross block/layer boundaries — and it omits the
    // start_layer_id offset, always touching layer 0. Detect the full-block
    // case to keep the fast path; otherwise scatter per-block per-(layer,kv).
    bool full_block = (block_bytes == cpu_block_stride_int64 * sizeof(int64_t));
    if (full_block) {
      for (const auto &seg : analysis.segments) {
        int64_t seg_start_block = cpu_block_ids[seg.start_k];
        int64_t seg_bytes = (int64_t)seg.run_len * block_bytes;
        int64_t *cpu_dst = cpu_ptr_int64 +
            (seg_start_block * cpu_block_stride_int64) +
            cpu_startoff_inside_chunks_int64;
        void *src = (char *)dev_staging +
            (int64_t)seg.start_k * total_iters * chunk_size_in_bytes;
        cudaMemcpyAsync(cpu_dst, src, seg_bytes,
                        cudaMemcpyDeviceToHost, stream);
        FLEXKV_GPU_CPU_TRANSFER(false, seg_bytes);
      }
    } else {
      // Per-layer batch (layer_parallel): block_bytes != cpu_block_stride.
      // BF layout [block, layer, kv, chunk]: L/N layers are contiguous within
      // each block, so we can transfer total_iters * chunk_size per block
      // (not per-(layer,kv) like the old code).
      //
      // enable_memcpy2d=true: per-segment cudaMemcpy2DAsync (fewest calls,
      //   fastest on NVIDIA; slow on P800).
      // enable_memcpy2d=false: per-block cudaMemcpyAsync (platform-safe,
      //   each block's L/N layers are contiguous in both dev_staging and CPU).
      size_t width = (size_t)total_iters * chunk_size_in_bytes;
      if (ce_config.enable_memcpy2d) {
        size_t spitch = (size_t)cpu_block_stride_int64 * sizeof(int64_t);
        for (const auto &seg : analysis.segments) {
          int64_t seg_start_block = cpu_block_ids[seg.start_k];
          int64_t *cpu_dst = cpu_ptr_int64 +
              seg_start_block * cpu_block_stride_int64 +
              start_layer_id * cpu_layer_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          void *src = (char *)dev_staging +
              (int64_t)seg.start_k * total_iters * chunk_size_in_bytes;
          cudaMemcpy2DAsync(cpu_dst, spitch, src, width,
                            width, (size_t)seg.run_len,
                            cudaMemcpyDeviceToHost, stream);
          FLEXKV_GPU_CPU_TRANSFER(false, width * seg.run_len);
        }
      } else {
        for (int b = 0; b < num_blocks; ++b) {
          int64_t cb = cpu_block_ids[b];
          int64_t *cpu_dst = cpu_ptr_int64 +
              cb * cpu_block_stride_int64 +
              start_layer_id * cpu_layer_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          void *src = (char *)dev_staging +
              (int64_t)b * total_iters * chunk_size_in_bytes;
          cudaMemcpyAsync(cpu_dst, src, width,
                          cudaMemcpyDeviceToHost, stream);
          FLEXKV_GPU_CPU_TRANSFER(false, width);
        }
      }
    }
    cudaStreamSynchronize(stream);

  } else {
    // ============ H2D ============
    // Step 1: H2D per-segment from CPU BLOCKFIRST to dev_staging
    // (reverse of D2H — same contiguous layout)
    int64_t block_bytes = total_iters * chunk_size_in_bytes;
    // See D2H Step 2: keep the fast contiguous path only when block_bytes ==
    // cpu_block_stride (all layers at once); otherwise gather per-block
    // per-(layer,kv) with explicit offsets (mirrors gather_from_cpu).
    bool full_block = (block_bytes == cpu_block_stride_int64 * sizeof(int64_t));
    if (full_block) {
      for (const auto &seg : analysis.segments) {
        int64_t seg_start_block = cpu_block_ids[seg.start_k];
        int64_t seg_bytes = (int64_t)seg.run_len * block_bytes;
        int64_t *cpu_src = cpu_ptr_int64 +
            (seg_start_block * cpu_block_stride_int64) +
            cpu_startoff_inside_chunks_int64;
        void *dst = (char *)dev_staging +
            (int64_t)seg.start_k * total_iters * chunk_size_in_bytes;
        cudaMemcpyAsync(dst, cpu_src, seg_bytes,
                        cudaMemcpyHostToDevice, stream);
        FLEXKV_GPU_CPU_TRANSFER(true, seg_bytes);
      }
    } else {
      // Per-layer batch (layer_parallel): symmetric to D2H.
      // enable_memcpy2d=true: per-segment cudaMemcpy2DAsync.
      // enable_memcpy2d=false: per-block cudaMemcpyAsync.
      size_t width = (size_t)total_iters * chunk_size_in_bytes;
      if (ce_config.enable_memcpy2d) {
        size_t spitch = (size_t)cpu_block_stride_int64 * sizeof(int64_t);
        for (const auto &seg : analysis.segments) {
          int64_t seg_start_block = cpu_block_ids[seg.start_k];
          const int64_t *cpu_src = cpu_ptr_int64 +
              seg_start_block * cpu_block_stride_int64 +
              start_layer_id * cpu_layer_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          void *dst = (char *)dev_staging +
              (int64_t)seg.start_k * total_iters * chunk_size_in_bytes;
          cudaMemcpy2DAsync(dst, width, cpu_src, spitch,
                            width, (size_t)seg.run_len,
                            cudaMemcpyHostToDevice, stream);
          FLEXKV_GPU_CPU_TRANSFER(true, width * seg.run_len);
        }
      } else {
        for (int b = 0; b < num_blocks; ++b) {
          int64_t cb = cpu_block_ids[b];
          const int64_t *cpu_src = cpu_ptr_int64 +
              cb * cpu_block_stride_int64 +
              start_layer_id * cpu_layer_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          void *dst = (char *)dev_staging +
              (int64_t)b * total_iters * chunk_size_in_bytes;
          cudaMemcpyAsync(dst, cpu_src, width,
                          cudaMemcpyHostToDevice, stream);
          FLEXKV_GPU_CPU_TRANSFER(true, width);
        }
      }
    }

    // Step 2: D2D reverse transpose — scatter from staging to GPU LAYERFIRST
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);
      int64_t *gpu_ptr_block0 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);
      int64_t *gpu_ptr_block1 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 1);
      // GPU block stride (pitch) in int64 elements (see D2H branch).
      int64_t gpu_block_stride_elems =
          (int64_t)((char *)gpu_ptr_block1 - (char *)gpu_ptr_block0) /
          sizeof(int64_t);
      // Add gpu_startoff to land on this rank's shard (defensive for sharded).
      int64_t *gpu_layer_kv_base =
          gpu_ptr_block0 + gpu_startoff_inside_chunks_int64;
      at::Tensor dst_view = at::from_blob(
          gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
          {gpu_block_stride_elems, 1}, i64_cuda);
      at::Tensor src_slice = dev_staging_view.select(1, it);
      if (analysis.gpu_log_contig) {
        dst_view.narrow(0, gpu_block_ids[0], num_blocks).copy_(src_slice);
      } else {
        dst_view.index_copy_(0, gpu_ids_cuda, src_slice);
      }
    }
    cudaStreamSynchronize(stream);
  }

  // Release from_blob views (underlying buffers are cached)
  gpu_ids_cuda.reset();
  dev_staging_view.reset();
}

// ---- Explicit template instantiations ----
//
// Signature groups:
//   NOSTG  = (int, int, int, int, int64_t*, GTensorHandler, int64_t,
//             int64_t*, int64_t*, int64_t x5, cudaStream_t, bool)
//   STG    = NOSTG + (const CEAnalysis&, const CETransferConfig&)
// per_block / bulk_contig use NOSTG; segmented_direct / staged_merge /

#define FLEXKV_INST_NOSTG(FN, BK)                                            \
  template void FN<BackendType::BK>(                                         \
      int, int, int, int, int64_t *, GTensorHandler, int64_t,                \
      int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,     \
      cudaStream_t, bool);

#define FLEXKV_INST_STG(FN, BK)                                              \
  template void FN<BackendType::BK>(                                         \
      int, int, int, int, int64_t *, GTensorHandler, int64_t,                \
      int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,     \
      cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &);

#define FLEXKV_INST_STG_SYNC(FN, BK)                                         \
  template void FN<BackendType::BK>(                                         \
      int, int, int, int, int64_t *, GTensorHandler, int64_t,                \
      int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,     \
      cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &, bool);

#define FLEXKV_INST_ALL_BACKENDS(MACRO, FN)                                  \
  MACRO(FN, VLLM) MACRO(FN, TRTLLM) MACRO(FN, SGLANG)

FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_NOSTG, ce_transfer_per_block)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_NOSTG, ce_transfer_contig_direct)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_segment_direct)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_segment_scatter)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_gather_scatter)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_gather_direct)

#undef FLEXKV_INST_NOSTG
#undef FLEXKV_INST_STG
#undef FLEXKV_INST_STG_SYNC
#undef FLEXKV_INST_ALL_BACKENDS

} // namespace flexkv
