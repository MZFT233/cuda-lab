// End-to-end toolchain smoke test: nvcc + MSVC + CMake + Ninja + CTest.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                   cudaGetErrorString(err), __FILE__, __LINE__);               \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void saxpyKernel(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  const int n = 1 << 20;                              // 1,048,576 elements
  const size_t bytes = n * sizeof(float);
  const float alpha = 2.5f;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("GPU                : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("elements           : %d\n", n);

  std::vector<float> hA(n), hB(n), hC(n), hRef(n);
  for (int i = 0; i < n; ++i) {
    hA[i] = static_cast<float>(i % 1000) * 0.5f;
    hB[i] = static_cast<float>(i % 100) * 0.25f;
    hRef[i] = alpha * hA[i] + hB[i];
  }

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  // Pass 1: c = a + b
  vectorAdd<<<blocks, threads>>>(dA, dB, dC, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

  float maxErrAdd = 0.f;
  for (int i = 0; i < n; ++i) {
    maxErrAdd = std::fmax(maxErrAdd, std::fabs(hC[i] - (hA[i] + hB[i])));
  }
  std::printf("vectorAdd max error: %.3e\n", maxErrAdd);

  // Pass 2: y = alpha*x + y  (saxpy)
  CUDA_CHECK(cudaMemcpy(dC, hA.data(), bytes, cudaMemcpyHostToDevice));
  saxpyKernel<<<blocks, threads>>>(alpha, dC, dB, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(), dB, bytes, cudaMemcpyDeviceToHost));

  float maxErrSaxpy = 0.f;
  for (int i = 0; i < n; ++i) {
    maxErrSaxpy = std::fmax(maxErrSaxpy, std::fabs(hC[i] - hRef[i]));
  }
  std::printf("saxpy     max error: %.3e\n", maxErrSaxpy);

  // Timed run (event-based)
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  const int iters = 50;
  CUDA_CHECK(cudaEventRecord(t0));
  for (int k = 0; k < iters; ++k) vectorAdd<<<blocks, threads>>>(dA, dB, dC, n);
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  const double gb = (3.0 * bytes * iters) / 1e9;
  std::printf("vectorAdd %d iters : %.3f ms total, %.1f GB/s effective\n",
              iters, ms, gb / (ms / 1000.0));

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));

  const float tol = 1e-3f;
  if (maxErrAdd <= tol && maxErrSaxpy <= tol) {
    std::printf("ALLOK\n");
    return 0;
  }
  std::printf("FAILED: errors above tolerance\n");
  return 1;
}
