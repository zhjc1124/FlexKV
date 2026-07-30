// Diagnostic reproducer for cudaMemcpyBatchAsync (CUDA 12.x).
// Build: nvcc -o /tmp/t benchmarks/test_cuda_batch_api.cu -lcudart && /tmp/t
// Tries several attribute/buffer configurations to isolate the "invalid argument" cause.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

static void dump_enums() {
  printf("=== enum values ===\n");
  printf("cudaMemcpySrcAccessOrderAny    = %d\n", (int)cudaMemcpySrcAccessOrderAny);
  printf("cudaMemcpySrcAccessOrderStream = %d\n", (int)cudaMemcpySrcAccessOrderStream);
  printf("cudaMemcpyFlagDefault          = %d\n", (int)cudaMemcpyFlagDefault);
  printf("cudaMemLocationTypeDevice      = %d\n", (int)cudaMemLocationTypeDevice);
  printf("cudaMemLocationTypeHost        = %d\n", (int)cudaMemLocationTypeHost);
  printf("=====================\n");
}

// buf_kind: 0 = pinned host (cudaMallocHost), 1 = managed (cudaMallocManaged)
// h2d: false = D2H (src dev, dst host), true = H2D (src host, dst dev)
// sao: srcAccessOrder value
// flags: attribute flags
static bool run_batch(int buf_kind, bool h2d, int sao, unsigned flags) {
  const int N = 8;
  const size_t SZ = 4096;
  char *hostA = nullptr, *hostB = nullptr;
  if (buf_kind == 0) {
    if (cudaMallocHost(&hostA, N*SZ) != cudaSuccess) return false;
    if (cudaMallocHost(&hostB, N*SZ) != cudaSuccess) return false;
  } else {
    if (cudaMallocManaged(&hostA, N*SZ) != cudaSuccess) return false;
    if (cudaMallocManaged(&hostB, N*SZ) != cudaSuccess) return false;
  }
  char *devA = nullptr, *devB = nullptr;
  if (cudaMalloc(&devA, N*SZ) != cudaSuccess) return false;
  if (cudaMalloc(&devB, N*SZ) != cudaSuccess) return false;

  char *src, *dst;
  if (!h2d) { src = devA; dst = hostB; } else { src = hostA; dst = devB; }

  std::vector<char> pat(N*SZ);
  for (size_t i=0;i<(size_t)N*SZ;++i) pat[i]=(char)i;
  cudaMemcpy(src, pat.data(), N*SZ, h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);

  std::vector<void*> b_dst, b_src;
  std::vector<size_t> b_cnt;
  for (int i=0;i<N;++i){ b_src.push_back(src+(size_t)i*SZ); b_dst.push_back(dst+(size_t)i*SZ); b_cnt.push_back(SZ); }

  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = (enum cudaMemcpySrcAccessOrder)sao;
  if (!h2d) { attr.srcLocHint.type = cudaMemLocationTypeDevice; attr.dstLocHint.type = cudaMemLocationTypeHost; }
  else      { attr.srcLocHint.type = cudaMemLocationTypeHost;   attr.dstLocHint.type = cudaMemLocationTypeDevice; }
  attr.srcLocHint.id = 0; attr.dstLocHint.id = 0;
  attr.flags = flags;

  cudaMemcpyAttributes attrs[1] = {attr};
  std::vector<size_t> attrsIdxs(b_dst.size(), 0);
  size_t failIdx = 0;
  cudaError_t e = cudaMemcpyBatchAsync(b_dst.data(), b_src.data(), b_cnt.data(),
                                       (size_t)N, attrs, attrsIdxs.data(), 1, &failIdx, cudaStreamDefault);
  bool ok = (e == cudaSuccess);
  if (ok) {
    cudaStreamSynchronize(cudaStreamDefault);
    std::vector<char> got(N*SZ);
    cudaMemcpy(got.data(), dst, N*SZ, h2d ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice);
    for (size_t i=0;i<(size_t)N*SZ;++i) if (got[i]!=(char)i) { ok=false; break; }
  }
  const char* bk = buf_kind==0 ? "pinned" : "managed";
  const char* dir = h2d ? "H2D" : "D2H";
  const char* saon = sao==(int)cudaMemcpySrcAccessOrderAny ? "Any" : "Stream";
  if (ok) printf("[PASS] %s %s sao=%s flags=%u\n", bk, dir, saon, flags);
  else    printf("[FAIL] %s %s sao=%s flags=%u : %s\n", bk, dir, saon, flags, cudaGetErrorString(e));
  cudaFree(devA); cudaFree(devB);
  if (buf_kind==0) { cudaFreeHost(hostA); cudaFreeHost(hostB); }
  else { cudaFree(hostA); cudaFree(hostB); }
  return ok;
}

int main() {
  cudaSetDevice(0);
  dump_enums();
  int any = (int)cudaMemcpySrcAccessOrderAny;
  int stream = (int)cudaMemcpySrcAccessOrderStream;
  unsigned fd = (unsigned)cudaMemcpyFlagDefault;
  run_batch(0, false, any, fd);    // A: pinned D2H, Any    (current repro, known FAIL)
  run_batch(1, false, any, fd);    // B: managed D2H, Any
  run_batch(1, false, stream, fd); // C: managed D2H, Stream
  run_batch(0, true,  any, fd);    // D: pinned H2D, Any
  run_batch(0, false, any, 0u);    // E: pinned D2H, Any, flags=0
  return 0;
}
