#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cudavec {

/**
 * \brief Empty kernel for lazy loading
 **/
inline __global__ void kernelWarmup() {}

template <typename Ty_>
__global__ void addKernel(Ty_ *c, const Ty_ *a, const Ty_ *b, uint32_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

template <typename Ty_>
__global__ void mulKernel(Ty_ *c, const Ty_ *a, const Ty_ *b, uint32_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] * b[i];
  }
}

template <typename Ty_>
__global__ void divKernel(Ty_ *c, const Ty_ *a, const Ty_ *b, uint32_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] / b[i];
  }
}

template <typename Ty_>
__global__ void addEqualsKernel(Ty_ *c, const Ty_ *a, const Ty_ &b,
                                uint32_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b;
  }
}

template <typename Ty_>
__global__ void mulEqualsKernel(Ty_ *c, const Ty_ *a, const Ty_ &b,
                                uint32_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] * b;
  }
}

template <typename Ty_>
__global__ void divEqualsKernel(Ty_ *c, const Ty_ *a, const Ty_ &b,
                                uint32_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] / b;
  }
}

/**
 * \brief Matrix multiplication kernel (no optimization for equal dim matrices)
 **/
template <typename Ty_>
__global__ void matmul_kernel(const Ty_ *A, const Ty_ *B, Ty_ *C, uint32_t M,
                              uint32_t N, uint32_t K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    Ty_ sum = 0;
    for (int i = 0; i < K; ++i) {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

} // namespace cudavec