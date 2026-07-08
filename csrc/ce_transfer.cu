/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CE adaptive transfer implementation: host-side analysis + multi-path
 * execution. Extracted from transfer.cu.
 *
 * Four paths selected by choose_path() based on block-id contiguity analysis
 * (see ce_transfer.h CEPath enum):
 *
 *   BULK_CONTIG (0):     gpu_log_contig && cpu_log_contig && cpu_phys_contig.
 *                         Single large cudaMemcpyAsync per (layer, kv_dim).
 *                         Optimal — O(1) API calls. No staging, no ping-pong.
 *
 *   SEGMENTED_DIRECT (1): cpu_phys_contig (LAYERFIRST), few segments.
 *                         Per-segment cudaMemcpyAsync straight to final dst.
 *                         No staging, no ping-pong.
 *
 *   STAGED_SCATTER (2):  dst NOT physically contiguous (BLOCKFIRST), or
 *                         sharded D2H (!gpu_phys_contig).
 *                         Pinned staging buffer + CPU scatter/gather.
 *                         Ping-pong: H2D = segment-level, D2H = layer-level.
 *                         Runtime variant: STAGED_CONTIG_RUN (gpu_phys_contig)
 *                         vs STAGED_PER_BLOCK (!gpu_phys_contig, sharded D2H).
 *
 *   GATHER_SCATTER (3):  Many scattered segments (num_segments > threshold).
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
// the former "Path 1" into BULK_CONTIG (was Path 0) / SEGMENTED_DIRECT (dst
// physically contiguous) / STAGED_SCATTER (dst strided) so each strategy has a
// self-describing name. GATHER_SCATTER (was Path 2) is unchanged.
CEPath choose_path(const CEAnalysis &a, const CETransferConfig &ce_config,
                   int64_t chunk_size_in_bytes) {
  // BULK_CONTIG: logical + physical contiguity on both sides -> one big memcpy.
  if (a.gpu_log_contig && a.cpu_log_contig && a.cpu_phys_contig && a.gpu_phys_contig)
    return CEPath::BULK_CONTIG;
  // GATHER_SCATTER (index_select) requires gpu_phys_contig (GPU block stride ==
  // chunk_size). Sharded D2H has chunk=shard, stride=gpu_chunk ->
  // !gpu_phys_contig -> cannot use it, so a strided dst must stage.
  if (!a.gpu_phys_contig) {
    // dst physically contiguous -> direct segment memcpy; else stage + scatter.
    return a.cpu_phys_contig ? CEPath::SEGMENTED_DIRECT
                             : CEPath::STAGED_SCATTER;
  }
  // gpu_phys_contig below. Few segments -> segment-level strategy.
  if (a.num_segments <= ce_config.segment_threshold) {
    return a.cpu_phys_contig ? CEPath::SEGMENTED_DIRECT
                             : CEPath::STAGED_SCATTER;
  }
  // Many scattered segments (src contiguous) -> GPU gather/scatter pipeline.
  // GATHER_SCATTER uses index_select/index_copy_ which requires chunk_size
  // to be 8-byte aligned (int64 view). Fall back to STAGED_SCATTER if not.
  if (chunk_size_in_bytes > 0 && chunk_size_in_bytes % sizeof(int64_t) != 0)
    return CEPath::STAGED_SCATTER;
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
void *get_cached_device_buffer(size_t size) {
  int dev = 0;
  cudaGetDevice(&dev);
  thread_local std::unordered_map<int, DeviceStagingBuf> cache;
  DeviceStagingBuf &b = cache[dev];
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

// Persistent ping-pong CUDA event pair (per current CUDA device, thread_local).
//
// WHY cached, not per-call: a per-call cudaEventCreate + cudaEventRecord +
// cudaEventDestroy sequence DEADLOCKS the async/polling batch loop. The
// LayerwiseTransferGroup submits ALL H2D batches (sync=false) and only THEN
// cudaStreamSynchronize()s the stream. If a CE call records an event on that
// stream and then cudaEventDestroy()s it before the stream has finished the
// recorded work, cudaEventDestroy blocks the batch-loop thread -> the next
// batch is never submitted -> the polling thread never fires all eventfds ->
// SGLang read(eventfd) times out -> HANG. (This was the original production
// deadlock that motivated guarding event creation behind `sync`, which in
// turn left `pingpong_events == nullptr` and caused the STAGED_SCATTER
// H2D NULL-deref crash.)
//
// Caching the pair once per device and reusing it removes the per-call
// destroy entirely. The caller ALWAYS drains the stream (layerwise:
// cudaStreamSynchronize after the batch loop; sync=true path: transfer.cu
// end-of-call sync) before the events could be reused in a stale/incomplete
// state, so the cached pair is safe for the process lifetime.
//
// `outstanding[s]` marks that staging-buffer slot s has a copy in flight on
// the stream that the CPU must not overwrite yet. It lives in the cached
// (process-lifetime) struct, so it correctly gates the CPU when a slot is
// reused by a LATER batch — not just within one call. This is what closes
// the async-H2D data-corruption race (the CPU gather overwriting a buffer
// the in-flight cudaMemcpyAsync is still reading).
struct PingPongEvents {
  cudaEvent_t ev[2] = {nullptr, nullptr};
  bool created = false;
  bool outstanding[2] = {false, false};
};

// Returns the cached ping-pong event pair for the current device (creating it
// on first use) and, via `out`, the owning struct so callers can read/reset
// the `outstanding` flags. The events are NEVER destroyed per call — they
// persist until thread/process exit. Safe under both call patterns:
// LayerwiseTransferGroup (single thread, cudaSetDevice per GPU) and
// TPTransferThreadGroup (one thread per GPU) — the cache is thread_local and
// device-keyed, just like the staging-buffer caches above.
cudaEvent_t *get_cached_pingpong_events(PingPongEvents *&out) {
  int dev = 0;
  cudaGetDevice(&dev);
  thread_local std::unordered_map<int, PingPongEvents> cache;
  PingPongEvents &e = cache[dev];
  if (!e.created) {
    TORCH_CHECK(cudaSuccess == cudaEventCreateWithFlags(&e.ev[0], cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed for cached ping-pong event[0]");
    TORCH_CHECK(cudaSuccess == cudaEventCreateWithFlags(&e.ev[1], cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed for cached ping-pong event[1]");
    e.created = true;
  }
  out = &e;
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
// BULK_CONTIG: single large memcpy per (layer, kv_dim)
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
// SEGMENTED_DIRECT: per-merged-run memcpy straight between CPU and GPU.
//   dst physically contiguous (LAYERFIRST + non-sharded), so no staging is
//   needed and ping-pong does not apply. One cudaMemcpyAsync per contiguous
//   run of blocks, for each (layer, kv).
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
// STAGED_SCATTER: pinned staging buffer + CPU scatter/gather to a strided
//   destination (BLOCKFIRST, or sharded D2H). Uses staging, so ping-pong
//   (ce_config.use_pingpong) applies here. Runtime GPU-side variant on
//   analysis.gpu_phys_contig:
//     STAGED_CONTIG_RUN  (gpu_phys_contig)  -> merged-run memcpy GPU<->staging
//     STAGED_PER_BLOCK   (!gpu_phys_contig) -> per-block   memcpy GPU<->staging
//   (sharded D2H is the only !gpu_phys_contig case.)
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
    const CEAnalysis &analysis, const CETransferConfig &ce_config,
    bool sync) {
  bool use_pingpong = ce_config.use_pingpong;
  // GPU-side variant: contiguous GPU blocks let us merge each segment into one
  // memcpy (STAGED_CONTIG_RUN); sharded D2H (!gpu_phys_contig) forces one
  // memcpy per block (STAGED_PER_BLOCK). Both share the staging buffer, CPU
  // scatter/gather and ping-pong machinery below. The inline
  // if (analysis.gpu_phys_contig) branches below select the variant.
  {
    // ---- staging buffer + CPU scatter/gather ----
    size_t layer_buf_size = (size_t)num_blocks * chunk_size_in_bytes;
    bool need_pingpong = use_pingpong && (!is_host_to_device || !sync);

    void *host_base = get_cached_hugepage_buffer(need_pingpong ? layer_buf_size * 2
                                                         : layer_buf_size);
    void *host_bufs[2] = {
        host_base,
        need_pingpong ? (char *)host_base + layer_buf_size : nullptr};
    // Persistent ping-pong events (cached per device, never destroyed per
    // call — see get_cached_pingpong_events). `pingpong_events` is ALWAYS
    // valid now; the `need_pingpong` flag (below) still decides whether we
    // actually use the 2-buffer overlap machinery. Caching avoids the
    // per-call cudaEventDestroy that deadlocked the async/polling batch loop,
    // and `pp->outstanding[]` lets us gate the CPU across calls (see below).
    PingPongEvents *pp = nullptr;
    cudaEvent_t *pingpong_events = get_cached_pingpong_events(pp);

    const int64_t total_iters = (int64_t)num_layers * kv_dim;
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);
      int idx = need_pingpong ? (int)(it & 1) : 0;
      int prev_idx = idx ^ 1;
      void *buf = host_bufs[idx];

      if (!is_host_to_device) {
        // ---- D2H ----
        // D2H all segments into staging
        int64_t seg_offset = 0;
        if (analysis.gpu_phys_contig) {
          // GPU blocks contiguous: continuous segment memcpy
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
        } else {
          // GPU blocks not contiguous (sharded D2H): per-block memcpy
          for (const auto &seg : analysis.segments) {
            for (int b = 0; b < seg.run_len; ++b) {
              int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                              i + start_layer_id, j,
                                              gpu_block_ids[seg.start_k + b]);
              int64_t *gpu_ptr_off =
                  reinterpret_cast<int64_t *>(gpu_ptr) +
                  gpu_startoff_inside_chunks_int64;
              cudaMemcpyAsync((char *)buf + seg_offset, gpu_ptr_off,
                              chunk_size_in_bytes,
                              cudaMemcpyDeviceToHost, stream);
              FLEXKV_GPU_CPU_TRANSFER(false, chunk_size_in_bytes);
              seg_offset += chunk_size_in_bytes;
            }
          }
        }
        if (need_pingpong) {
          cudaEventRecord(pingpong_events[idx], stream);
          pp->outstanding[idx] = true;
          // CPU scatter previous layer
          if (it >= 1) {
            if (pp->outstanding[prev_idx]) {
              cudaEventSynchronize(pingpong_events[prev_idx]);
              pp->outstanding[prev_idx] = false;
            }
            int pi = (int)((it - 1) / kv_dim);
            int pj = (int)((it - 1) % kv_dim);
            // scatter from host_bufs[prev_idx] to strided dst
            int64_t *cpu_base =
                cpu_ptr_int64 + (pi + start_layer_id) * cpu_layer_stride_int64 +
                pj * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
            int64_t off = 0;
            for (const auto &seg : analysis.segments) {
              for (int b = 0; b < seg.run_len; ++b) {
                int64_t cb = cpu_block_ids[seg.start_k + b];
                int64_t *dst = cpu_base + cb * cpu_block_stride_int64;
                memcpy(dst, (char *)host_bufs[prev_idx] + off,
                       chunk_size_in_bytes);
                off += chunk_size_in_bytes;
              }
            }
          }
        } else {
          cudaStreamSynchronize(stream);
          // scatter current layer
          int64_t *cpu_base =
              cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
              j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
          int64_t off = 0;
          for (const auto &seg : analysis.segments) {
            for (int b = 0; b < seg.run_len; ++b) {
              int64_t cb = cpu_block_ids[seg.start_k + b];
              int64_t *dst = cpu_base + cb * cpu_block_stride_int64;
              memcpy(dst, (char *)buf + off, chunk_size_in_bytes);
              off += chunk_size_in_bytes;
            }
          }
        }
      } else {
        // ---- H2D ----
        if (need_pingpong) {
          // Segment-level ping-pong: CPU gather of seg K+1 overlaps with
          // H2D of seg K. Mirrors SGLang hicache (p800 transfer.cu).
          // Each segment alternates between host_bufs[0] and host_bufs[1].
          int seg_idx = 0;
          int64_t *cpu_base =
              cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
              j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
          for (const auto &seg : analysis.segments) {
            int sidx = seg_idx & 1;
            int64_t seg_size = (int64_t)seg.run_len * chunk_size_in_bytes;

            // Wait for the previous use of this slot (2 segments ago, or a
            // prior batch) to finish BEFORE the CPU overwrites it. The CUDA
            // stream is in-order, but the CPU gather (host memcpy) is NOT
            // covered by stream ordering, so we must gate on the slot's
            // event. `pp->outstanding[sidx]` survives across calls, so this
            // also closes the cross-batch staging-buffer reuse race (the
            // real cause of the silent data corruption under polling).
            // In async mode this blocks only until THIS slot's prior copy
            // completes — never a full-stream sync — so it never deadlocks
            // the batch loop.
            if (pp->outstanding[sidx]) {
              cudaEventSynchronize(pingpong_events[sidx]);
              pp->outstanding[sidx] = false;
            }

            // CPU gather: copy from strided CPU into staging slot.
            int64_t off = 0;
            for (int b = 0; b < seg.run_len; ++b) {
              int64_t cb = cpu_block_ids[seg.start_k + b];
              int64_t *src = cpu_base + cb * cpu_block_stride_int64;
              memcpy((char *)host_bufs[sidx] + off, src, chunk_size_in_bytes);
              off += chunk_size_in_bytes;
            }

            // H2D from staging to GPU.
            if (analysis.gpu_phys_contig) {
              int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                              i + start_layer_id, j,
                                              gpu_block_ids[seg.start_k]);
              int64_t *gpu_ptr_off =
                  reinterpret_cast<int64_t *>(gpu_ptr) +
                  gpu_startoff_inside_chunks_int64;
              cudaMemcpyAsync(gpu_ptr_off, host_bufs[sidx], seg_size,
                              cudaMemcpyHostToDevice, stream);
            } else {
              // Per-block H2D (!gpu_phys_contig, rare for H2D).
              int64_t off2 = 0;
              for (int b = 0; b < seg.run_len; ++b) {
                int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                                i + start_layer_id, j,
                                                gpu_block_ids[seg.start_k + b]);
                int64_t *gpu_ptr_off =
                    reinterpret_cast<int64_t *>(gpu_ptr) +
                    gpu_startoff_inside_chunks_int64;
                cudaMemcpyAsync(gpu_ptr_off,
                                (char *)host_bufs[sidx] + off2,
                                chunk_size_in_bytes,
                                cudaMemcpyHostToDevice, stream);
                off2 += chunk_size_in_bytes;
              }
            }
            FLEXKV_GPU_CPU_TRANSFER(true, seg_size);
            cudaEventRecord(pingpong_events[sidx], stream);
            pp->outstanding[sidx] = true;
            seg_idx++;
          }
          // Flush remaining in-flight segments. In sync=true mode, wait for
          // both slots' last copies so this call returns fully complete.
          // In sync=false (async/layerwise) mode, do NOT sync here — the
          // batch loop's caller (LayerwiseTransferGroup) cudaStreamSynchronize
          // after all batches. The `outstanding[]` flags stay set so the next
          // batch gates on these copies before reusing the buffers.
          if (sync) {
            for (int s = 0; s < 2; ++s) {
              if (pp->outstanding[s]) {
                cudaEventSynchronize(pingpong_events[s]);
                pp->outstanding[s] = false;
              }
            }
          }
        } else {
          // No ping-pong: a single staging buffer (host_bufs[0]) reused every
          // iteration. In sync=true mode we drain the stream between iters so
          // the next CPU gather never races the in-flight H2D. In sync=false
          // (async/polling) mode we must NOT block the batch loop, so we gate
          // the CPU gather behind a cached event marking the previous
          // iteration's H2D — it waits for ONLY that copy (not the whole
          // stream), keeping correctness without deadlocking.
          int64_t *cpu_base =
              cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
              j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
          // Gate the CPU gather behind the previous iteration's in-flight
          // H2D (async mode only; sync mode drains via cudaStreamSynchronize).
          if (!sync && pp->outstanding[0]) {
            cudaEventSynchronize(pingpong_events[0]);
            pp->outstanding[0] = false;
          }
          int64_t off = 0;
          for (const auto &seg : analysis.segments) {
            for (int b = 0; b < seg.run_len; ++b) {
              int64_t cb = cpu_block_ids[seg.start_k + b];
              int64_t *src = cpu_base + cb * cpu_block_stride_int64;
              memcpy((char *)buf + off, src, chunk_size_in_bytes);
              off += chunk_size_in_bytes;
            }
          }
          // H2D all segments from staging
          off = 0;
          if (analysis.gpu_phys_contig) {
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
          } else {
            for (const auto &seg : analysis.segments) {
              for (int b = 0; b < seg.run_len; ++b) {
                int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                                i + start_layer_id, j,
                                                gpu_block_ids[seg.start_k + b]);
                int64_t *gpu_ptr_off =
                    reinterpret_cast<int64_t *>(gpu_ptr) +
                    gpu_startoff_inside_chunks_int64;
                cudaMemcpyAsync(gpu_ptr_off, (char *)buf + off,
                                chunk_size_in_bytes,
                                cudaMemcpyHostToDevice, stream);
                FLEXKV_GPU_CPU_TRANSFER(true, chunk_size_in_bytes);
                off += chunk_size_in_bytes;
              }
            }
          }
          // Mark this iteration's H2D in flight so the next iteration (or the
          // next batch) gates the CPU before reusing `buf`. sync=false relies
          // on the caller's cudaStreamSynchronize; sync=true also drains below.
          if (sync) {
            cudaStreamSynchronize(stream);
          } else {
            cudaEventRecord(pingpong_events[0], stream);
            pp->outstanding[0] = true;
          }
        }
      }
    }
    // Drain last ping-pong slot (D2H)
    if (!is_host_to_device && need_pingpong && total_iters >= 1) {
      int64_t last = total_iters - 1;
      int last_idx = (int)(last & 1);
      if (pp->outstanding[last_idx]) {
        cudaEventSynchronize(pingpong_events[last_idx]);
        pp->outstanding[last_idx] = false;
      }
      int li = (int)(last / kv_dim);
      int lj = (int)(last % kv_dim);
      int64_t *cpu_base = cpu_ptr_int64 + (li + start_layer_id) * cpu_layer_stride_int64 +
          lj * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
      int64_t off = 0;
      for (const auto &seg : analysis.segments) {
        for (int b = 0; b < seg.run_len; ++b) {
          int64_t cb = cpu_block_ids[seg.start_k + b];
          int64_t *dst = cpu_base + cb * cpu_block_stride_int64;
          memcpy(dst, (char *)host_bufs[last_idx] + off,
                 chunk_size_in_bytes);
          off += chunk_size_in_bytes;
        }
      }
    }
    // NOTE: ping-pong events are cached (get_cached_pingpong_events) and
    // intentionally NOT destroyed here. A per-call cudaEventDestroy blocks if
    // the event was recorded on a stream the caller has not yet drained, which
    // deadlocked the async/polling batch loop. The caller always drains the
    // stream before the events could be reused in a stale state.
  }
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
    const CEAnalysis &analysis, const CETransferConfig &ce_config,
    bool sync) {
  TORCH_CHECK(chunk_size_in_bytes % sizeof(int64_t) == 0,
              "Path 2 requires chunk_size_in_bytes % 8 == 0");
  const int64_t elems_per_block = chunk_size_in_bytes / sizeof(int64_t);
  // buf_bytes = total bytes for all blocks' staging buffer.
  // elems_per_block * sizeof(int64_t) == chunk_size_in_bytes, so this is
  // equivalent to num_blocks * chunk_size_in_bytes — kept as one variable.
  const size_t buf_bytes = (size_t)num_blocks * (size_t)chunk_size_in_bytes;
  bool use_pingpong = ce_config.use_pingpong;

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
  // Only needed when GPU blocks are non-contiguous (GATHER_SCATTER path).
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
  if (!analysis.gpu_log_contig) {
    // gpu_ids_cuda: for D2H index_select gather (GPU source non-contiguous)
    gpu_ids_raw = get_cached_device_buffer(ids_bytes);
    cudaMemcpyAsync(gpu_ids_raw, gpu_block_ids, ids_bytes,
                    cudaMemcpyHostToDevice, stream);
    gpu_ids_cuda = at::from_blob(gpu_ids_raw, {num_blocks}, i64_cuda);

    // dst_ids_cuda: for H2D index_copy_ scatter (GPU dst non-contiguous)
    if (is_host_to_device) {
      dst_ids_raw = get_cached_device_buffer(ids_bytes);
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
  // are non-contiguous. gpu_log_contig = GPU side (direction-agnostic).
  // For ping-pong, allocate 2x and split into two halves (each call to
  // get_cached_device_buffer returns the SAME pointer, so calling it twice
  // would give two views of the same memory — breaking ping-pong).
  bool need_dev_buf = !analysis.gpu_log_contig;
  void *dev_raw[2] = {nullptr, nullptr};
  at::Tensor dev_buf[2];
  if (need_dev_buf) {
    bool need_two = use_pingpong && (!is_host_to_device || !sync);  // D2H or async H2D
    size_t dev_alloc = need_two ? buf_bytes * 2 : buf_bytes;
    void *dev_base = get_cached_device_buffer(dev_alloc);
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

  // Ping-pong: D2H always (CPU scatter overlap); H2D only in async mode
  // (sync=false) where per-iteration stream sync would deadlock the caller.
  // In sync=true mode, H2D ping-pong is disabled (event overhead > overlap gain).
  bool need_pingpong_host = need_host_buf && use_pingpong && (!is_host_to_device || !sync);

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

  // Persistent ping-pong events (cached per device, never destroyed per call
  // — see get_cached_pingpong_events). `pp_ev` is ALWAYS valid; `pingpong_events`
  // is non-null only when we use the 2-buffer overlap machinery
  // (need_pingpong_host). The `pp->outstanding[]` flags gate the CPU when a
  // staging slot is reused — within a call and across batches — which closes
  // the async-H2D data-corruption race (CPU gather overwriting a buffer the
  // in-flight H2D/index_copy_ is still reading).
  PingPongEvents *pp = nullptr;
  cudaEvent_t *pp_ev = get_cached_pingpong_events(pp);
  cudaEvent_t *pingpong_events = need_pingpong_host ? pp_ev : nullptr;

  const int64_t total_iters = (int64_t)num_layers * kv_dim;
  for (int64_t it = 0; it < total_iters; ++it) {
    int i = (int)(it / kv_dim);
    int j = (int)(it % kv_dim);
    int idx = need_pingpong_host ? (int)(it & 1) : 0;
    int prev_idx = idx ^ 1;

    int64_t *gpu_layer_kv_base =
        ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);

    if (!is_host_to_device) {
      // ============ D2H ============
      // Step 1: GPU gather (if src non-contig)
      const int64_t *d2h_src;
      if (analysis.gpu_log_contig) {
        d2h_src = reinterpret_cast<int64_t *>(gpu_layer_kv_base) +
                  gpu_startoff_inside_chunks_int64 +
                  gpu_block_ids[0] * (chunk_size_in_bytes / sizeof(int64_t));
      } else {
        at::Tensor src_view = at::from_blob(
            gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block}, i64_cuda);
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
        cudaEventRecord(pp_ev[idx], stream);
        pp->outstanding[idx] = true;
        // Step 3: CPU scatter previous slot
        if (it >= 1) {
          if (pp->outstanding[prev_idx]) {
            cudaEventSynchronize(pp_ev[prev_idx]);
            pp->outstanding[prev_idx] = false;
          }
          int pi = (int)((it - 1) / kv_dim);
          int pj = (int)((it - 1) % kv_dim);
          int64_t *cpu_base = cpu_ptr_int64 + (pi + start_layer_id) * cpu_layer_stride_int64 +
              pj * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
          for (int k = 0; k < num_blocks; ++k) {
            int64_t cb = cpu_block_ids[k];
            memcpy(cpu_base + cb * cpu_block_stride_int64,
                   (char *)host_buf[prev_idx] +
                       (int64_t)k * chunk_size_in_bytes,
                   chunk_size_in_bytes);
          }
        }
      } else if (need_host_buf) {
        cudaStreamSynchronize(stream);
        // scatter current
        int64_t *cpu_base = cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
            j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
        for (int k = 0; k < num_blocks; ++k) {
          int64_t cb = cpu_block_ids[k];
          memcpy(cpu_base + cb * cpu_block_stride_int64,
                 (char *)host_buf[idx] +
                     (int64_t)k * chunk_size_in_bytes,
                 chunk_size_in_bytes);
        }
      }
    } else {
      // ============ H2D ============
      // GATE: wait for the previous use of this staging slot to finish before
      // we overwrite it. Host staging (host_buf[idx]) is written by the CPU
      // gather (Step 1) when CPU src is non-contiguous; device staging
      // (dev_buf[idx]) is written by the GPU index_copy_ scatter (Step 3) when
      // GPU dst is non-contiguous. Both are keyed by `idx` and reused every
      // iteration AND across batches, so the prior use must complete first.
      // `pp->outstanding[idx]` survives across calls, closing the cross-batch
      // async-H2D data-corruption race. In async mode this blocks only until
      // THIS slot's prior copy finishes (not a full-stream sync), so it never
      // deadlocks the batch loop.
      if (need_host_buf || need_dev_buf) {
        if (pp->outstanding[idx]) {
          cudaEventSynchronize(pp_ev[idx]);
          pp->outstanding[idx] = false;
        }
      }

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
        // gather into staging
        int64_t *cpu_base = cpu_ptr_int64 +
            (i + start_layer_id) * cpu_layer_stride_int64 +
            j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
        for (int k = 0; k < num_blocks; ++k) {
          int64_t cb = cpu_block_ids[k];
          memcpy((char *)host_buf[idx] +
                     (int64_t)k * chunk_size_in_bytes,
                 cpu_base + cb * cpu_block_stride_int64,
                 chunk_size_in_bytes);
        }
        h2d_src = host_buf[idx];
      }

      // Step 2: H2D — GPU dst contiguity = gpu_log_contig (GPU side)
      void *h2d_dst;
      if (analysis.gpu_log_contig) {
        h2d_dst = reinterpret_cast<int64_t *>(gpu_layer_kv_base) +
                  gpu_startoff_inside_chunks_int64 +
                  gpu_block_ids[0] * (chunk_size_in_bytes / sizeof(int64_t));
      } else {
        h2d_dst = dev_buf[idx].data_ptr();
      }
      cudaMemcpyAsync(h2d_dst, h2d_src, buf_bytes,
                      cudaMemcpyHostToDevice, stream);
      FLEXKV_GPU_CPU_TRANSFER(true, buf_bytes);

      // Step 3: GPU scatter (if GPU dst non-contig = !gpu_log_contig)
      if (!analysis.gpu_log_contig) {
        at::Tensor dst_view = at::from_blob(
            gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block}, i64_cuda);
        dst_view.index_copy_(0, dst_ids_cuda, dev_buf[idx]);
      }

      // Mark this slot's copy in flight so the next reuse (iteration or
      // batch) gates on it. The event covers the full H2D + index_copy_
      // chain. In sync mode the caller (transfer.cu end-of-call sync) drains
      // the stream; in async mode the LayerwiseTransferGroup drains after all
      // batches — so the event is always complete before the slot is reused.
      if (need_host_buf || need_dev_buf) {
        cudaEventRecord(pp_ev[idx], stream);
        pp->outstanding[idx] = true;
      }
    }
  }

  // Drain last D2H scatter (CPU must finish before returning).
  if (!is_host_to_device && need_pingpong_host && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    if (pp->outstanding[last_idx]) {
      cudaEventSynchronize(pp_ev[last_idx]);
      pp->outstanding[last_idx] = false;
    }
    int li = (int)(last / kv_dim);
    int lj = (int)(last % kv_dim);
    int64_t *cpu_base = cpu_ptr_int64 + (li + start_layer_id) * cpu_layer_stride_int64 +
        lj * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
    for (int k = 0; k < num_blocks; ++k) {
      int64_t cb = cpu_block_ids[k];
      memcpy(cpu_base + cb * cpu_block_stride_int64,
             (char *)host_buf[last_idx] +
                 (int64_t)k * chunk_size_in_bytes,
             chunk_size_in_bytes);
    }
  }

  // Drain the stream before returning ONLY in sync=true mode. In async mode
  // (sync=false, polling) the caller (layerwise.cpp) cudaStreamSynchronize's
  // after all batches are submitted — so we must NOT block here. All per-call
  // GPU resources (gpu_ids, dst_ids, dev_buf, host_buf) are cached and stream
  // ordering guarantees the previous call's async ops complete before the next
  // call's on the same stream; the `pp->outstanding[]` flags gate the CPU so
  // the staging buffers are never overwritten while an in-flight copy reads
  // them.
  if (sync) {
    cudaStreamSynchronize(stream);
  }

  // NOTE: ping-pong events are cached (get_cached_pingpong_events) and
  // intentionally NOT destroyed here. A per-call cudaEventDestroy blocks if
  // the event was recorded on a stream the caller has not yet drained, which
  // deadlocked the async/polling batch loop. The caller always drains the
  // stream before the events could be reused in a stale state.

  // Release the from_blob views. The large staging buffers (dev_buf, host_buf)
  // are cached and freed by RAII on thread exit. The id tensors
  // (gpu_ids_raw, dst_ids_raw) are also cached (get_cached_device_buffer),
  // so we just reset the views — no cudaFree, no drain needed.
  // In sync=false mode the GPU may still be reading these buffers
  // asynchronously; since they are cached (not per-call freed), this is safe.
  gpu_ids_cuda.reset();
  dst_ids_cuda.reset();
  dev_buf[0].reset();
  dev_buf[1].reset();
}

// ---- Explicit template instantiations ----
//
// Signature groups:
//   NOSTG  = (int, int, int, int, int64_t*, GTensorHandler, int64_t,
//             int64_t*, int64_t*, int64_t x5, cudaStream_t, bool)
//   STG    = NOSTG + (const CEAnalysis&, const CETransferConfig&)
// per_block / bulk_contig use NOSTG; segmented_direct / staged_scatter /
// gather_scatter use STG.

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
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_NOSTG, ce_transfer_bulk_contig)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_segmented_direct)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG_SYNC, ce_transfer_staged_scatter)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG_SYNC, ce_transfer_gather_scatter)

#undef FLEXKV_INST_NOSTG
#undef FLEXKV_INST_STG
#undef FLEXKV_INST_STG_SYNC
#undef FLEXKV_INST_ALL_BACKENDS

} // namespace flexkv
