// toolcheck.cu -- deliberately broken kernels, used to prove that
// compute-sanitizer actually detects faults (not merely that it starts up).
//
//   --mode clean  : correct kernels (sanitizer must report 0 errors)
//   --mode oob    : out-of-bounds global write (memcheck must catch)
//   --mode race   : shared-memory race (racecheck must catch)
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                   cudaGetErrorString(err), __FILE__, __LINE__);               \
      return 2;                                                                \
    }                                                                          \
  } while (0)

__global__ void cleanKernel(float* a, float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i] * 2.0f;
}

// Writes 4096 elements past the end of b. Note: a one-element overflow is NOT
// reliably reported, because cudaMalloc allocations are page-granular - the
// sanitizer only flags accesses outside the allocation's pages.
__global__ void oobKernel(float* a, float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  b[i + 4096] = a[i] * 2.0f;
}

// Shared-memory race: thread t writes slot t, then reads slot t+1 without any
// __syncthreads() in between, so the read may observe an unwritten slot.
__global__ void raceKernel(float* out) {
  __shared__ float s[32];
  int t = threadIdx.x;
  s[t] = (float)t;
  float v = s[(t + 1) % 32];
  out[t] = v;
}

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "clean";
  const int n = 1024;
  const size_t bytes = n * sizeof(float);

  float* dA = nullptr;
  float* dB = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMemset(dA, 0, bytes));
  CUDA_CHECK(cudaMemset(dB, 0, bytes));

  std::printf("mode=%s\n", mode);

  if (std::strcmp(mode, "oob") == 0) {
    oobKernel<<<n / 256, 256>>>(dA, dB, n);  // one thread writes dB[n]
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } else if (std::strcmp(mode, "race") == 0) {
    raceKernel<<<1, 32>>>(dB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    cleanKernel<<<n / 256, 256>>>(dA, dB, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  std::printf("done\n");
  return 0;
}
