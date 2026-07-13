#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sharded BF D2H benchmark: 3 approaches on NVIDIA & P800.

Pure ctypes, no torch. Auto-detects CUDA runtime library.

Layout (MLA sharded, BLOCKFIRST CPU interleave):
  GPU:  [num_layers, num_blocks, full_chunk]   (LAYERFIRST)
  CPU:  [num_blocks, num_layers, num_gpus, shard_size]  (BLOCKFIRST interleave)

  full_chunk = tokens_per_block * head_dim * dtype_size
  shard_size = full_chunk / num_gpus

Approaches:
  1. memcpy2d   : per-(layer,seg) cudaMemcpy2DAsync, strided D2H, no D2D
  2. contig+merge: per-GPU contiguous cudaMemcpyAsync + CPU interleave merge
                  (D2D transpose cost measured separately on H20, estimated on P800)
  3. baseline   : per-(layer,block) cudaMemcpyAsync + CPU scatter

Usage:
  python benchmarks/microbenchmark_sharded_bf_d2h.py
  NUM_BLOCKS=512 NUM_LAYERS=80 python benchmarks/microbenchmark_sharded_bf_d2h.py
"""
import ctypes, time, random, os, functools
print = functools.partial(print, flush=True)

# ---- auto-detect CUDA runtime ----
LIB_PATHS = [
    "/usr/local/xpu-5.18.1.0/so/libcudart.so.12",   # P800 Kunlunxin
    "libcudart.so",                                    # NVIDIA (in LD_LIBRARY_PATH)
    "libcudart.so.12",
    "/usr/local/cuda/lib64/libcudart.so",
]
cuda = None
for p in LIB_PATHS:
    try:
        cuda = ctypes.CDLL(p)
        print("Loaded CUDA runtime: %s" % p)
        break
    except OSError:
        continue
if cuda is None:
    raise RuntimeError("Cannot find libcudart. Set LD_LIBRARY_PATH or edit LIB_PATHS.")

HTD = 1; DTH = 2; D2D = 3
PORTABLE = 1
c_void_p_p = ctypes.POINTER(ctypes.c_void_p)

def decl(name, restype, argtypes):
    fn = getattr(cuda, name); fn.restype = restype; fn.argtypes = argtypes

decl("cudaSetDevice", ctypes.c_int, [ctypes.c_int])
decl("cudaMalloc", ctypes.c_int, [c_void_p_p, ctypes.c_size_t])
decl("cudaFree", ctypes.c_int, [ctypes.c_void_p])
decl("cudaHostAlloc", ctypes.c_int, [c_void_p_p, ctypes.c_size_t, ctypes.c_uint])
decl("cudaFreeHost", ctypes.c_int, [ctypes.c_void_p])
decl("cudaStreamCreate", ctypes.c_int, [c_void_p_p])
decl("cudaStreamDestroy", ctypes.c_int, [ctypes.c_void_p])
decl("cudaStreamSynchronize", ctypes.c_int, [ctypes.c_void_p])
decl("cudaMemcpyAsync", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p])
decl("cudaMemcpy2DAsync", ctypes.c_int, [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p])
decl("cudaGetErrorString", ctypes.c_char_p, [ctypes.c_int])
decl("cudaMemset", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t])

def check(err, tag):
    if err != 0:
        s = cuda.cudaGetErrorString(err)
        raise RuntimeError("CUDA error %d at %s: %s" % (err, tag, s.decode() if s else "?"))

# ---- config ----
NUM_BLOCKS  = int(os.environ.get("NUM_BLOCKS", "256"))
NUM_LAYERS  = int(os.environ.get("NUM_LAYERS", "80"))
TOKENS      = int(os.environ.get("TOKENS", "64"))
HEAD_DIM    = int(os.environ.get("HEAD_DIM", "576"))
DTYPE       = 2  # fp16
NUM_GPUS    = int(os.environ.get("NUM_GPUS", "8"))
REPEAT      = int(os.environ.get("REPEAT", "3"))
N_SELECT    = int(os.environ.get("N_SELECT", str(NUM_BLOCKS)))  # blocks transferred

FULL_CHUNK  = TOKENS * HEAD_DIM * DTYPE        # bytes per block per layer (full)
SHARD_SIZE  = FULL_CHUNK // NUM_GPUS            # bytes per shard
BLOCK_STRIDE = NUM_LAYERS * FULL_CHUNK          # CPU BF: stride between blocks
LAYER_STRIDE = FULL_CHUNK                        # CPU BF: stride between layers
TOTAL_PER_GPU = N_SELECT * NUM_LAYERS * SHARD_SIZE  # bytes per GPU
TOTAL_ALL    = N_SELECT * NUM_LAYERS * FULL_CHUNK    # total bytes (all GPUs)

GiB = 1024**3
MiB = 1024**2

check(cuda.cudaSetDevice(0), "setDevice")

# ---- alloc ----
# GPU: [NUM_LAYERS, NUM_BLOCKS, FULL_CHUNK] (LAYERFIRST)
d_kv = ctypes.c_void_p()
check(cuda.cudaMalloc(c_void_p_p(d_kv), NUM_BLOCKS * NUM_LAYERS * FULL_CHUNK), "malloc gpu")

# CPU interleave target: [N_SELECT, NUM_LAYERS, NUM_GPUS, SHARD_SIZE]
h_interleave = ctypes.c_void_p()
check(cuda.cudaHostAlloc(c_void_p_p(h_interleave), N_SELECT * NUM_LAYERS * FULL_CHUNK, PORTABLE), "malloc cpu interleave")

# CPU separate regions (for contiguous D2H + merge): NUM_GPUS regions
h_regions = []
for i in range(NUM_GPUS):
    r = ctypes.c_void_p()
    check(cuda.cudaHostAlloc(c_void_p_p(r), TOTAL_PER_GPU, PORTABLE), "malloc cpu region %d" % i)
    h_regions.append(r)

stream = ctypes.c_void_p()
check(cuda.cudaStreamCreate(c_void_p_p(stream)), "streamCreate")

# fill GPU with pattern
pattern = (ctypes.c_uint8 * FULL_CHUNK)()
for i in range(FULL_CHUNK):
    pattern[i] = i & 0xFF
for l in range(NUM_LAYERS):
    for b in range(NUM_BLOCKS):
        off = ctypes.c_void_p(d_kv.value + (l * NUM_BLOCKS + b) * FULL_CHUNK)
        check(cuda.cudaMemcpyAsync(off, ctypes.cast(pattern, ctypes.c_void_p), FULL_CHUNK, HTD, ctypes.c_void_p(0)), "fill")
check(cuda.cudaStreamSynchronize(ctypes.c_void_p(0)), "fill sync")

# block ids: few_seg (4 segments)
random.seed(42)
q = N_SELECT // 4
base = list(range(N_SELECT))
# few_seg pattern: [0..q-1, 2q..3q-1, q..2q-1, 3q..4q-1]
selected = base[0:q] + base[2*q:3*q] + base[q:2*q] + base[3*q:4*q]
segments = []
k = 0
while k < N_SELECT:
    start = k
    while k + 1 < N_SELECT and selected[k + 1] == selected[k] + 1:
        k += 1
    segments.append((selected[start], k - start + 1))  # (block_id, run_len)
    k += 1

print("=" * 90)
print("Sharded BF D2H Benchmark")
print("layout: MLA sharded, [num_layers=%d, num_blocks=%d, tokens=%d, hd=%d] fp16, %d GPUs" %
      (NUM_LAYERS, NUM_BLOCKS, TOKENS, HEAD_DIM, NUM_GPUS))
print("full_chunk=%d  shard_size=%d  block_stride=%d  layer_stride=%d" %
      (FULL_CHUNK, SHARD_SIZE, BLOCK_STRIDE, LAYER_STRIDE))
print("selected=%d blocks, %d segments, total=%.1f MiB, per-GPU=%.1f MiB" %
      (N_SELECT, len(segments), TOTAL_ALL / MiB, TOTAL_PER_GPU / MiB))
print("=" * 90)

def time_fn(fn, nbytes):
    best = 1e9
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        check(cuda.cudaStreamSynchronize(stream), "sync")
        t1 = time.perf_counter()
        best = min(best, (t1 - t0) * 1000.0)
    bw = (nbytes / GiB) / (best / 1000.0) if best > 0 else 0
    return best, bw

def memset_cpu(ptr, size):
    ctypes.memset(ctypes.c_void_p(ptr), 0, size)

# ============================================================
# 1. memcpy2d: per-(layer,segment) cudaMemcpy2DAsync, strided
# ============================================================
def op_memcpy2d():
    for gi in range(NUM_GPUS):
        shard_off = gi * SHARD_SIZE
        for layer in range(NUM_LAYERS):
            for seg_bid, run_len in segments:
                src = ctypes.c_void_p(d_kv.value + layer * NUM_BLOCKS * FULL_CHUNK + seg_bid * FULL_CHUNK + shard_off)
                dst = ctypes.c_void_p(h_interleave.value + seg_bid * BLOCK_STRIDE + layer * LAYER_STRIDE + shard_off)
                check(cuda.cudaMemcpy2DAsync(
                    dst, BLOCK_STRIDE,      # dpitch (CPU BF block stride, large)
                    src, FULL_CHUNK,         # spitch (GPU LF block stride, small)
                    SHARD_SIZE, run_len,     # width=shard, height=run_len
                    DTH, stream), "memcpy2d")

t, bw = time_fn(op_memcpy2d, TOTAL_ALL)
print("\n1. memcpy2d (strided cudaMemcpy2DAsync):")
print("   time=%8.1f ms  bw=%6.2f GiB/s  calls=%d (L%d×S%d×G%d)" %
      (t, bw, NUM_LAYERS * len(segments) * NUM_GPUS, NUM_LAYERS, len(segments), NUM_GPUS))

# ============================================================
# 2. contig+merge: per-GPU contiguous D2H + CPU interleave merge
# ============================================================
def op_contig_d2h():
    # Each GPU: contiguous D2H of its shard data to its own region
    # Source is LAYERFIRST strided, so we can't do ONE contiguous D2H per GPU.
    # Instead: per (layer, segment) contiguous D2H to region (blocks contiguous within a layer segment)
    for gi in range(NUM_GPUS):
        shard_off = gi * SHARD_SIZE
        region = h_regions[gi]
        off = 0
        for layer in range(NUM_LAYERS):
            for seg_bid, run_len in segments:
                src = ctypes.c_void_p(d_kv.value + layer * NUM_BLOCKS * FULL_CHUNK + seg_bid * FULL_CHUNK + shard_off)
                dst = ctypes.c_void_p(region.value + off)
                check(cuda.cudaMemcpyAsync(dst, src, run_len * SHARD_SIZE, DTH, stream), "contig_d2h")
                off += run_len * SHARD_SIZE

def op_contig_d2h_oneshot():
    # Ideal ceiling: all data contiguous (simulate D2D transpose result)
    # One big cudaMemcpyAsync per GPU (if data were contiguous after D2D)
    for gi in range(NUM_GPUS):
        region = h_regions[gi]
        # Use first part of d_kv as source (just for timing, data doesn't matter)
        src = ctypes.c_void_p(d_kv.value)
        dst = ctypes.c_void_p(region.value)
        check(cuda.cudaMemcpyAsync(dst, src, TOTAL_PER_GPU, DTH, stream), "oneshot")

def op_cpu_merge():
    # CPU interleave merge: from NUM_GPUS regions to interleave layout
    # Region gi: [N_SELECT, NUM_LAYERS, SHARD_SIZE] contiguous
    # Interleave: [N_SELECT, NUM_LAYERS, NUM_GPUS, SHARD_SIZE]
    for b_idx in range(N_SELECT):
        for layer in range(NUM_LAYERS):
            for gi in range(NUM_GPUS):
                src = ctypes.c_void_p(h_regions[gi].value + (b_idx * NUM_LAYERS + layer) * SHARD_SIZE)
                dst = ctypes.c_void_p(h_interleave.value + b_idx * BLOCK_STRIDE + layer * LAYER_STRIDE + gi * SHARD_SIZE)
                ctypes.memmove(dst, src, SHARD_SIZE)

def op_cpu_merge_optimized():
    # Optimized: for each (block, gpu), strided copy of all layers
    # src: region[gi] + b * NL * SS (contiguous, SS stride)
    # dst: interleave + b * BS + gi * SS (strided, LS stride)
    for b_idx in range(N_SELECT):
        for gi in range(NUM_GPUS):
            src_base = h_regions[gi].value + b_idx * NUM_LAYERS * SHARD_SIZE
            dst_base = h_interleave.value + b_idx * BLOCK_STRIDE + gi * SHARD_SIZE
            for layer in range(NUM_LAYERS):
                ctypes.memmove(
                    ctypes.c_void_p(dst_base + layer * LAYER_STRIDE),
                    ctypes.c_void_p(src_base + layer * SHARD_SIZE),
                    SHARD_SIZE)

t_d2h, bw_d2h = time_fn(op_contig_d2h, TOTAL_ALL)
print("\n2. contig+merge (D2D + contiguous D2H + CPU merge):")
print("   D2H (per-layer-seg contiguous):  time=%8.1f ms  bw=%6.2f GiB/s  calls=%d" %
      (t_d2h, bw_d2h, NUM_LAYERS * len(segments) * NUM_GPUS))

t_d2h_1shot, bw_1shot = time_fn(op_contig_d2h_oneshot, TOTAL_ALL)
print("   D2H (oneshot ceiling, post-D2D): time=%8.1f ms  bw=%6.2f GiB/s  calls=%d" %
      (t_d2h_1shot, bw_1shot, NUM_GPUS))

# CPU merge (only time the memcpy, not D2H)
t_merge = 1e9
for _ in range(REPEAT):
    t0 = time.perf_counter()
    op_cpu_merge_optimized()
    t1 = time.perf_counter()
    t_merge = min(t_merge, (t1 - t0) * 1000.0)
print("   CPU interleave merge:            time=%8.1f ms  (memcpy only)" % t_merge)
print("   Total (D2H oneshot + merge):     time=%8.1f ms  (+ D2D cost separately)" % (t_d2h_1shot + t_merge))
print("   Total (D2H per-seg + merge):     time=%8.1f ms  (no D2D needed)" % (t_d2h + t_merge))

# ============================================================
# 3. baseline: per-(layer,block) cudaMemcpyAsync + CPU scatter
# ============================================================
def op_baseline():
    for gi in range(NUM_GPUS):
        shard_off = gi * SHARD_SIZE
        for layer in range(NUM_LAYERS):
            for b_idx in range(N_SELECT):
                bid = selected[b_idx]
                src = ctypes.c_void_p(d_kv.value + layer * NUM_BLOCKS * FULL_CHUNK + bid * FULL_CHUNK + shard_off)
                dst = ctypes.c_void_p(h_interleave.value + bid * BLOCK_STRIDE + layer * LAYER_STRIDE + shard_off)
                check(cuda.cudaMemcpyAsync(dst, src, SHARD_SIZE, DTH, stream), "baseline")

t_base, bw_base = time_fn(op_baseline, TOTAL_ALL)
print("\n3. baseline (per-block cudaMemcpyAsync):")
print("   time=%8.1f ms  bw=%6.2f GiB/s  calls=%d (L%d×B%d×G%d)" %
      (t_base, bw_base, NUM_LAYERS * N_SELECT * NUM_GPUS, NUM_LAYERS, N_SELECT, NUM_GPUS))

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 90)
print("Summary (lower is better):")
print("-" * 90)
print("  %-45s %10s %10s %10s" % ("Approach", "Time(ms)", "BW(GiB/s)", "Calls"))
print("-" * 90)
print("  %-45s %10.1f %10.2f %10d" % ("1. memcpy2d (strided 2D)", t, bw, NUM_LAYERS * len(segments) * NUM_GPUS))
print("  %-45s %10.1f %10.2f %10d" % ("2a. contig D2H (per-layer-seg) + merge", t_d2h + t_merge, TOTAL_ALL/(t_d2h+t_merge)/GiB*1000 if t_d2h+t_merge>0 else 0, NUM_LAYERS * len(segments) * NUM_GPUS))
print("  %-45s %10.1f %10.2f %10d" % ("2b. contig D2H (oneshot) + merge + D2D", t_d2h_1shot + t_merge + 1, 0, NUM_GPUS + 1))
print("  %-45s %10.1f %10.2f %10d" % ("3. baseline (per-block)", t_base, bw_base, NUM_LAYERS * N_SELECT * NUM_GPUS))
print("-" * 90)
print("  Note: 2b adds ~1ms D2D (H20 measured), P800 D2D cost TBD")
print("  memcpy2d vs baseline: %.1fx" % (t_base / t if t > 0 else 0))
print("  contig+merge vs baseline: %.1fx" % (t_base / (t_d2h + t_merge) if t_d2h + t_merge > 0 else 0))
print("=" * 90)

# cleanup
check(cuda.cudaStreamDestroy(stream), "stream destroy")
check(cuda.cudaFree(d_kv), "free gpu")
check(cuda.cudaFreeHost(h_interleave), "free cpu interleave")
for r in h_regions:
    check(cuda.cudaFreeHost(r), "free cpu region")
print("\nDONE")
