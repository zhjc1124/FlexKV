"""
Microbenchmark: does enabling Transparent Huge Pages (THP) on the CPU KV
buffer improve host<->device (P800 / Kunlun XPU, exposed as "cuda" via the
CUDA shim) KV transfer bandwidth?

This script mirrors the FlexKV CPU KV buffer allocation path
(`flexkv/storage/allocator.py`):

    CPUAllocator.allocate (FLEXKV_USE_HUGEPAGE_CPU_BUFFER=0, default THP path):
        mmap(... PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS)
        + madvise(MADV_HUGEPAGE)      # 2MB transparent huge pages
        + cudaHostRegister(...)       # pin the pages

We compare two *both-pinned* configurations (the only thing that differs is
the page size hint passed to madvise):

    THP   : mmap -> madvise(MADV_HUGEPAGE)      -> cudaHostRegister
    PLAIN : mmap -> madvise(MADV_NOHUGEPAGE)    -> cudaHostRegister

Both buffers are then wrapped as a torch tensor living on the registered host
memory, and we measure H2D (host->device) and D2H (device->host) copy
bandwidth.

------------------------------------------------------------------------------
CRITICAL P800 CONSTRAINTS (verified on the pod):
------------------------------------------------------------------------------
1. TIMING: we use ONLY `time.perf_counter()` wrapped around the copy call plus
   `torch.cuda.synchronize()`. We NEVER use `torch.cuda.Event` /
   `elapsed_time` -- on P800 the CUDA-event shim always returns 0.0, which
   would make every bandwidth number meaningless.
2. DEVICE: `torch.cuda` is the shim over XPU. `torch.cuda.is_available()` is
   expected to be True. We use `torch.cuda.current_device()` /
   `torch.cuda.synchronize()` only.
3. REGISTRATION: `cudaHostRegister` / `cudaHostUnregister` are called via
   ctypes against libcudart.so.12. We prefer
   `/usr/local/xpu-5.18.1.0/so/libcudart.so.12`, then
   `ctypes.util.find_library('cudart')`, then standard search paths. On
   failure we degrade to `tensor.pin_memory()` and emit a clear caveat.
4. ALLOCATION: the backing memory is OUR OWN `mmap`. We mmap first, then
   `madvise`, THEN `cudaHostRegister`. The mmap pointer is wrapped as a
   `numpy.ndarray` -> `torch.from_numpy(...)` so the tensor lives on the
   registered host memory.

Usage:
    python benchmarks/microbenchmark_thp_cpu_buffer.py \
        --sizes 64M,256M,1G --rounds 20 --warmup 3

Only numpy + torch are required (no external dependencies).
"""

import argparse
import ctypes
import ctypes.util
import mmap
import os
import statistics
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception as exc:  # pragma: no cover - environment dependent
    print("ERROR: failed to import torch: %s" % exc, file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Low-level constants
# ---------------------------------------------------------------------------

# madvise advice values for Linux x86_64 / aarch64 (MADV_HUGEPAGE=14,
# MADV_NOHUGEPAGE=15). P800 is x86_64.
MADV_HUGEPAGE = 14
MADV_NOHUGEPAGE = 15

# MAP_HUGETLB (Linux) -- used only in the best-effort hugetlbfs path.
# Value is 0x40000 on x86_64/aarch64.
MAP_HUGETLB = 0x40000

# Candidate locations for the CUDA-runtime shim (P800 exposes XPU as cuda).
CUDART_CANDIDATES = [
    "/usr/local/xpu-5.18.1.0/so/libcudart.so.12",
    "libcudart.so.12",
    "libcudart.so",
]

# Dtype used for the KV buffer (matches KV-cache element width).
DTYPE = np.float16
TORCH_DTYPE = torch.float16
ELEM_BYTES = np.dtype(DTYPE).itemsize  # 2 for float16

# Default result file (pod path that survives `exec` exit).
DEFAULT_OUT_PATH = "/workspace/zittozhang/thp_bench_result.txt"


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def load_libc() -> ctypes.CDLL:
    """Load libc and configure the `madvise` prototype.

    Returns a CDLL handle. `madvise` is advisory, so a failure there is
    non-fatal (we only warn).
    """
    libname = ctypes.util.find_library("c") or "libc.so.6"
    try:
        libc = ctypes.CDLL(libname, use_errno=True)
    except OSError as exc:
        raise RuntimeError("failed to load libc (%s): %s" % (libname, exc))
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    return libc


def load_cudart() -> Optional[ctypes.CDLL]:
    """Locate libcudart.so.12 (the P800 CUDA-shim runtime).

    Tries the explicit XPU path first, then find_library, then standard
    names. Returns None if nothing can be loaded (registration will then be
    degraded to tensor.pin_memory() with a caveat).
    """
    searched: List[str] = list(CUDART_CANDIDATES)
    found = ctypes.util.find_library("cudart")
    if found and found not in searched:
        searched.append(found)

    for path in searched:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        # cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
        lib.cudaHostRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint
        ]
        lib.cudaHostRegister.restype = ctypes.c_int
        # cudaError_t cudaHostUnregister(void* ptr);
        lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        lib.cudaHostUnregister.restype = ctypes.c_int
        return lib

    print(
        "[warn] libcudart not found (searched: %s). Host registration will be "
        "skipped and results may NOT reflect true pinned bandwidth."
        % ", ".join(searched),
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# Allocation: mmap -> madvise -> cudaHostRegister -> torch tensor
# ---------------------------------------------------------------------------

def _advise(libc: ctypes.CDLL, addr: int, length: int, advice: int) -> None:
    """Call madvise(addr, length, advice). Advisory only -- warn on failure."""
    ret = libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length),
                       ctypes.c_int(advice))
    if ret != 0:
        errno = ctypes.get_errno()
        name = "MADV_HUGEPAGE" if advice == MADV_HUGEPAGE else "MADV_NOHUGEPAGE"
        print(
            "  [warn] madvise(%s) failed (ret=%d, errno=%d) -- continuing "
            "without the hint" % (name, ret, errno),
            file=sys.stderr,
        )


def allocate_pinned(
    nbytes: int,
    use_thp: bool,
    cudart: Optional[ctypes.CDLL],
    libc: ctypes.CDLL,
) -> Tuple[mmap.mmap, "torch.Tensor", int, bool, int]:
    """Allocate a pinned host tensor of `nbytes` bytes on our own mmap.

    Order (mirrors FlexKV CPUAllocator):
        mmap  ->  madvise(HUGEPAGE|NOHUGEPAGE)  ->  cudaHostRegister.

    Returns:
        buf          : the mmap object (kept alive to back the tensor)
        tensor       : torch tensor sharing the mmap memory (host, pinned)
        addr         : integer base address (for unregister)
        registered   : True if cudaHostRegister succeeded
        actual_bytes : real byte count (nbytes rounded to an element)
    """
    n_elements = nbytes // ELEM_BYTES
    if n_elements <= 0:
        raise ValueError("requested size %d bytes is smaller than one %s element"
                         % (nbytes, DTYPE.__name__))
    actual_bytes = n_elements * ELEM_BYTES

    # 1) mmap anonymous, read/write.
    buf = mmap.mmap(
        -1,
        actual_bytes,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
    )

    # Wrap as a writable numpy view over the mmap, then grab the base address.
    arr = np.ndarray((n_elements,), dtype=DTYPE, buffer=buf)
    addr = int(arr.ctypes.data)

    # 2) madvise: request (or forbid) transparent huge pages.
    advice = MADV_HUGEPAGE if use_thp else MADV_NOHUGEPAGE
    _advise(libc, addr, actual_bytes, advice)

    # 3) cudaHostRegister the mmap memory (pin it for DMA).
    registered = False
    if cudart is not None:
        try:
            err = cudart.cudaHostRegister(
                ctypes.c_void_p(addr),
                ctypes.c_size_t(actual_bytes),
                ctypes.c_uint(0),  # cudaHostRegisterDefault
            )
            if isinstance(err, tuple):  # some ctypes bindings return tuples
                err = err[0]
            if err != 0:
                print(
                    "  [warn] cudaHostRegister returned %s (addr=0x%x, "
                    "size=%d); degrading to pin_memory()" % (err, addr, actual_bytes),
                    file=sys.stderr,
                )
            else:
                registered = True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(
                "  [warn] cudaHostRegister raised %s; degrading to pin_memory()"
                % exc,
                file=sys.stderr,
            )

    # Wrap the mmap-backed numpy array as a torch tensor (shares memory).
    tensor = torch.from_numpy(arr)

    # 4) Degraded fallback: if driver registration failed, still pin a tensor
    #    (this copies into a fresh pinned buffer, so the mmap is no longer the
    #    backing store -- noted as a caveat by the caller via `registered`).
    if not registered:
        try:
            tensor = tensor.pin_memory()
        except Exception as exc:  # pragma: no cover
            print("  [warn] pin_memory() fallback failed: %s" % exc, file=sys.stderr)

    return buf, tensor, addr, registered, actual_bytes


def free_pinned(
    buf: mmap.mmap,
    addr: int,
    registered: bool,
    cudart: Optional[ctypes.CDLL],
) -> None:
    """Unregister (if registered) and close the mmap."""
    if registered and cudart is not None:
        try:
            err = cudart.cudaHostUnregister(ctypes.c_void_p(addr))
            if isinstance(err, tuple):
                err = err[0]
            if err != 0:
                print(
                    "  [warn] cudaHostUnregister returned %s (addr=0x%x)"
                    % (err, addr),
                    file=sys.stderr,
                )
        except Exception as exc:  # pragma: no cover
            print("  [warn] cudaHostUnregister raised %s" % exc, file=sys.stderr)
    try:
        buf.close()
    except Exception as exc:  # pragma: no cover
        print("  [warn] mmap close failed: %s" % exc, file=sys.stderr)


# ---------------------------------------------------------------------------
# Bandwidth measurement (perf_counter + synchronize ONLY)
# ---------------------------------------------------------------------------

def _measure_direction(
    src: "torch.Tensor",
    dst: "torch.Tensor",
    rounds: int,
    warmup: int,
    direction: str,
) -> float:
    """Measure copy latency (seconds) for one direction, return mean latency.

    `direction` is "h2d" (src=host, dst=device) or "d2h" (src=device,
    dst=host). Timing uses perf_counter around the non-blocking copy plus a
    synchronize. CUDA events are deliberately NOT used.
    """
    for _ in range(warmup):
        if direction == "h2d":
            dst.copy_(src, non_blocking=True)
        else:
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

    latencies: List[float] = []
    for _ in range(rounds):
        torch.cuda.synchronize()  # ensure no overlap with the previous copy
        t0 = time.perf_counter()
        if direction == "h2d":
            dst.copy_(src, non_blocking=True)
        else:
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    return statistics.fmean(latencies)


def bench_pair(
    host_tensor: "torch.Tensor",
    size_bytes: int,
    rounds: int,
    warmup: int,
) -> Tuple[float, float]:
    """Measure H2D and D2H bandwidth (GB/s) for a pinned host tensor.

    Returns (h2d_gbps, d2h_gbps).
    """
    dev_tensor = torch.empty_like(host_tensor, device="cuda")
    dev_tensor.fill_(1.0)  # give the device buffer real content

    # H2D: host(pinned) -> device.
    dst_dev = torch.empty_like(host_tensor, device="cuda")
    h2d_lat = _measure_direction(host_tensor, dst_dev, rounds, warmup, "h2d")

    # D2H: device -> host(pinned).
    d2h_lat = _measure_direction(dev_tensor, host_tensor, rounds, warmup, "d2h")

    h2d_gbps = (size_bytes / h2d_lat) / 1e9 if h2d_lat > 0 else 0.0
    d2h_gbps = (size_bytes / d2h_lat) / 1e9 if d2h_lat > 0 else 0.0
    return h2d_gbps, d2h_gbps


# ---------------------------------------------------------------------------
# Optional best-effort hugetlbfs path
# ---------------------------------------------------------------------------

def _has_reserved_hugepages() -> bool:
    try:
        with open("/proc/sys/vm/nr_hugepages", "r") as fh:
            return int(fh.read().strip()) > 0
    except OSError:
        return False


def _has_hugetlbfs_mount() -> bool:
    try:
        with open("/proc/mounts", "r") as fh:
            return any("hugetlbfs" in line for line in fh)
    except OSError:
        return False


def bench_hugetlb(
    nbytes: int,
    rounds: int,
    warmup: int,
    cudart: Optional[ctypes.CDLL],
    libc: ctypes.CDLL,
) -> Optional[Tuple[float, float]]:
    """Best-effort explicit hugetlbfs (MAP_HUGETLB) measurement.

    Only attempted when reserved hugepages exist and hugetlbfs is mounted;
    otherwise returns None so the caller can print a 'skipped' note.
    """
    if not (_has_reserved_hugepages() and _has_hugetlbfs_mount()):
        return None

    n_elements = nbytes // ELEM_BYTES
    actual_bytes = n_elements * ELEM_BYTES
    try:
        buf = mmap.mmap(
            -1,
            actual_bytes,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_HUGETLB,
        )
        arr = np.ndarray((n_elements,), dtype=DTYPE, buffer=buf)
        addr = int(arr.ctypes.data)
        registered = False
        if cudart is not None:
            err = cudart.cudaHostRegister(
                ctypes.c_void_p(addr),
                ctypes.c_size_t(actual_bytes),
                ctypes.c_uint(0),
            )
            if isinstance(err, tuple):
                err = err[0]
            registered = (err == 0)
        tensor = torch.from_numpy(arr)
        if not registered and cudart is not None:
            tensor = tensor.pin_memory()
        result = bench_pair(tensor, actual_bytes, rounds, warmup)
        free_pinned(buf, addr, registered, cudart)
        return result
    except Exception as exc:  # pragma: no cover - environment dependent
        print("  [warn] hugetlbfs path failed: %s" % exc, file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Size parsing / output helpers
# ---------------------------------------------------------------------------

def parse_size(token: str) -> int:
    """Parse a human size token (e.g. '64M', '256M', '1G', '1048576') -> bytes."""
    token = token.strip().upper()
    if not token:
        raise ValueError("empty size token")
    units = {
        "B": 1,
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
        "T": 1024 ** 4,
    }
    if token[-1] in units:
        return int(float(token[:-1]) * units[token[-1]])
    return int(token)


def _fmt_size(nbytes: int) -> str:
    for unit, factor in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if nbytes >= factor:
            return "%.2f %s" % (nbytes / factor, unit)
    return "%d B" % nbytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure H2D/D2H KV-transfer bandwidth: THP-pinned vs "
                    "plain 4KB-pinned CPU buffer (P800 / XPU-as-cuda)."
    )
    parser.add_argument(
        "--sizes",
        default="64M,256M,1G",
        help="Comma-separated sizes in bytes or human form (e.g. 64M,256M,1G). "
             "Default: 64M,256M,1G",
    )
    parser.add_argument("--rounds", type=int, default=20,
                        help="Timed rounds per measurement (default 20).")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup rounds before timing (default 3).")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH,
                        help="Path to also write the result table.")
    args = parser.parse_args()

    if not CUDA_AVAILABLE:
        print("ERROR: torch.cuda.is_available() is False -- no XPU/cuda device.",
              file=sys.stderr)
        sys.exit(1)

    torch.cuda.init()
    device_id = torch.cuda.current_device()

    try:
        libc = load_libc()
    except RuntimeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(1)
    cudart = load_cudart()

    sizes = [parse_size(tok) for tok in args.sizes.split(",") if tok.strip()]

    print("=" * 78)
    print("THP vs PLAIN CPU KV buffer bandwidth benchmark (P800 / XPU-as-cuda)")
    print("=" * 78)
    print("Device        : %s (id=%d)" % (torch.cuda.get_device_name(device_id), device_id))
    print("Rounds/Warmup : %d / %d" % (args.rounds, args.warmup))
    print("Sizes         : %s" % ", ".join(_fmt_size(s) for s in sizes))
    print("Dtype         : %s (%d B/element)" % (DTYPE.__name__, ELEM_BYTES))
    print("Timing        : time.perf_counter() + torch.cuda.synchronize() "
          "(NO cuda.Event)")
    print("Registration  : %s" % ("libcudart present" if cudart is not None
                                  else "libcudart MISSING (degraded)"))
    print("=" * 78)

    # Records: list of (size_bytes, mode, h2d_gbps, d2h_gbps, registered)
    records: List[Tuple[int, str, float, float, bool]] = []
    caveats: List[str] = []

    for size in sizes:
        print("\n--- size %s (%d bytes) ---" % (_fmt_size(size), size))
        for mode, use_thp in (("THP", True), ("PLAIN", False)):
            buf, tensor, addr, registered, actual = allocate_pinned(
                size, use_thp, cudart, libc
            )
            if not registered:
                caveats.append(
                    "%s @ %s: not driver-registered (degraded to pin_memory)"
                    % (mode, _fmt_size(size))
                )
            try:
                h2d, d2h = bench_pair(tensor, actual, args.rounds, args.warmup)
            finally:
                del tensor
                free_pinned(buf, addr, registered, cudart)
            records.append((size, mode, h2d, d2h, registered))
            print("  %-6s H2D=%8.2f GB/s  D2H=%8.2f GB/s%s"
                  % (mode, h2d, d2h, "" if registered else "  [!not-registered]"))

        # Optional best-effort hugetlbfs path.
        hugetlb = bench_hugetlb(size, args.rounds, args.warmup, cudart, libc)
        if hugetlb is None:
            print("  HUGETLB skipped (no reserved hugepages / no hugetlbfs mount)")
        else:
            h2d_h, d2h_h = hugetlb
            records.append((size, "HUGETLB", h2d_h, d2h_h, True))
            print("  HUGETLB H2D=%8.2f GB/s  D2H=%8.2f GB/s" % (h2d_h, d2h_h))

    # ---- Summary table (stdout + file) -------------------------------------
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("THP vs PLAIN CPU KV buffer bandwidth benchmark (P800 / XPU-as-cuda)")
    lines.append("=" * 78)
    lines.append("Device: %s (id=%d)"
                 % (torch.cuda.get_device_name(device_id), device_id))
    lines.append("Rounds/Warmup: %d / %d   Dtype: %s"
                 % (args.rounds, args.warmup, DTYPE.__name__))
    lines.append("Timing: time.perf_counter() + torch.cuda.synchronize() "
                 "(NO cuda.Event)")
    lines.append("")
    header = "%-12s %-8s %12s %12s" % ("size", "mode", "H2D_GBps", "D2H_GBps")
    lines.append(header)
    lines.append("-" * len(header))
    for size, mode, h2d, d2h, registered in records:
        lines.append("%-12s %-8s %12.2f %12.2f%s"
                     % (_fmt_size(size), mode, h2d, d2h,
                        "" if registered else "  [!not-registered]"))

    # Speedup summary THP vs PLAIN (same size, paired).
    lines.append("")
    lines.append("Speedup THP vs PLAIN (bandwidth ratio):")
    lines.append("-" * 50)
    by_size = {}
    for size, mode, h2d, d2h, _ in records:
        by_size.setdefault(size, {})[mode] = (h2d, d2h)
    for size in sizes:
        entry = by_size.get(size, {})
        if "THP" in entry and "PLAIN" in entry:
            thp_h, thp_d = entry["THP"]
            pln_h, pln_d = entry["PLAIN"]
            h2d_x = (thp_h / pln_h) if pln_h > 0 else float("nan")
            d2h_x = (thp_d / pln_d) if pln_d > 0 else float("nan")
            lines.append(
                "%-12s H2D x%.3f   D2H x%.3f"
                % (_fmt_size(size), h2d_x, d2h_x)
            )

    if caveats:
        lines.append("")
        lines.append("CAVEATS:")
        for c in caveats:
            lines.append("  - " + c)

    report = "\n".join(lines) + "\n"
    print("\n" + report)

    # Write to the pod result file (best-effort, survives exec exit).
    try:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(report)
        print("[ok] result written to %s" % args.out)
    except OSError as exc:
        print("[warn] could not write result file %s: %s"
              % (args.out, exc), file=sys.stderr)


if __name__ == "__main__":
    main()
