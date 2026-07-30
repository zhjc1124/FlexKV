// Minimal reproducer for cudaMemcpyBatchAsync attribute setup (CUDA 12.x).
// Build: nvcc -o /tmp/t benchmarks/test_cuda_batch_api.cu -lcudart && /tmp/t
// Prints [OK]/[FAIL] for a D2H batch copy. If it fails to COMPILE on
// cudaMemLocationType*, that enum name is wrong (try cudaMemcpyLocationType).
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

int main() {
  cudaSetDevice(0);
  const int N = 8;
  const size_t SZ = 4096;
  std::vector<char> hsrc(N * SZ);
  for (size_t i = 0; i < hsrc.size(); ++i) hsrc[i] = (char)i;
  std::vector<char> hdst(N * SZ, 0);
  char *dsrc = nullptr;
  if (cudaMalloc(&dsrc, N * SZ) != cudaSuccess) { printf("cudaMalloc fail\n"); return 1; }
  cudaMemcpy(dsrc, hsrc.data(), N * SZ, cudaMemcpyHostToDevice);

  std::vector<void *> b_dst, b_src;
  std::vector<size_t> b_cnt;
  for (int i = 0; i < N; ++i) {
    b_dst.push_back(hdst.data() + (size_t)i * SZ);
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
  size_t attrsIdxs[1] = {0};
  size_t failIdx = 0;
  cudaError_t e = cudaMemcpyBatchAsync(b_dst.data(), b_src.data(), b_cnt.data(),
                                       (size_t)N, attrs, attrsIdxs, 1, &failIdx,
                                       cudaStreamDefault);
  if (e != cudaSuccess) {
    printf("[FAIL] cudaMemcpyBatchAsync: %s\n", cudaGetErrorString(e));
    return 1;
  }
  cudaStreamSynchronize(cudaStreamDefault);
  bool ok = true;
  for (size_t i = 0; i < hdst.size(); ++i) if (hdst[i] != (char)i) { ok = false; break; }
  printf("[%s] cudaMemcpyBatchAsync D2H with explicit locHints\n", ok ? "OK" : "MISMATCH");
  cudaFree(dsrc);
  return ok ? 0 : 1;
}
