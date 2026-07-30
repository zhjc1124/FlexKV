// Minimal reproducer for cudaMemcpyBatchAsync attribute setup (CUDA 12.x).
// Build: nvcc -o /tmp/t benchmarks/test_cuda_batch_api.cu -lcudart && /tmp/t
// Prints [OK]/[FAIL] for a D2H batch copy.
// Key lessons (CUDA 12.9):
//  - host pointers MUST be pinned (cudaMallocHost); pageable host -> invalid argument
//  - attrsIdxs must have `count` entries (one index per copy), each < numAttrs
//  - direction must be explicit via srcLocHint/dstLocHint (cudaMemLocationType*)
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

int main() {
  cudaSetDevice(0);
  const int N = 8;
  const size_t SZ = 4096;
  char *hsrc = nullptr, *hdst = nullptr;
  if (cudaMallocHost(&hsrc, N * SZ) != cudaSuccess) { printf("cudaMallocHost hsrc fail\n"); return 1; }
  if (cudaMallocHost(&hdst, N * SZ) != cudaSuccess) { printf("cudaMallocHost hdst fail\n"); return 1; }
  for (size_t i = 0; i < (size_t)N * SZ; ++i) hsrc[i] = (char)i;
  for (size_t i = 0; i < (size_t)N * SZ; ++i) hdst[i] = 0;
  char *dsrc = nullptr;
  if (cudaMalloc(&dsrc, N * SZ) != cudaSuccess) { printf("cudaMalloc fail\n"); return 1; }
  cudaMemcpy(dsrc, hsrc, N * SZ, cudaMemcpyHostToDevice);

  std::vector<void *> b_dst, b_src;
  std::vector<size_t> b_cnt;
  for (int i = 0; i < N; ++i) {
    b_dst.push_back(hdst + (size_t)i * SZ);
    b_src.push_back(dsrc + (size_t)i * SZ);
    b_cnt.push_back(SZ);
  }

  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
  attr.srcLocHint.type = cudaMemLocationTypeDevice;  // D2H: source is device
  attr.srcLocHint.id = 0;
  attr.dstLocHint.type = cudaMemLocationTypeHost;    // D2H: dest is host
  attr.dstLocHint.id = 0;
  attr.flags = cudaMemcpyDefault;
  cudaMemcpyAttributes attrs[1] = {attr};
  // attrsIdxs MUST have `count` entries (one per copy), each a valid attr index.
  std::vector<size_t> attrsIdxs(b_dst.size(), 0);
  size_t failIdx = 0;
  cudaError_t e = cudaMemcpyBatchAsync(b_dst.data(), b_src.data(), b_cnt.data(),
                                       (size_t)N, attrs, attrsIdxs.data(), 1, &failIdx,
                                       cudaStreamDefault);
  if (e != cudaSuccess) {
    printf("[FAIL] cudaMemcpyBatchAsync: %s\n", cudaGetErrorString(e));
    cudaFree(dsrc); cudaFreeHost(hsrc); cudaFreeHost(hdst);
    return 1;
  }
  cudaStreamSynchronize(cudaStreamDefault);
  bool ok = true;
  for (size_t i = 0; i < (size_t)N * SZ; ++i) if (hdst[i] != (char)i) { ok = false; break; }
  printf("[%s] cudaMemcpyBatchAsync D2H with pinned host + explicit locHints + count-sized attrsIdxs\n", ok ? "OK" : "MISMATCH");
  cudaFree(dsrc); cudaFreeHost(hsrc); cudaFreeHost(hdst);
  return ok ? 0 : 1;
}
