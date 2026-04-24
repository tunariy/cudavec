#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <kernel.cuh>
#include <matrix.cuh>

#include <cmath>
#include <cstring>
#include <iostream>

namespace cudavec {

template <typename Ty_>
__host__ std::vector<Ty_> matmul_cuda_VRAM(const Ty_ *host_a, const Ty_ *host_b,
                                           uint32_t M, uint32_t N, uint32_t K) {
  cudaError_t errorStatus;
  cudaStream_t cudaStream;
  cudaDeviceProp deviceProps;
  cudaGetDeviceProperties(&deviceProps, 0);

  size_t size_a = M * K;
  size_t size_b = K * N;
  size_t size_c = M * N;

  std::vector<Ty_> host_c;
  Ty_ *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;

  host_c.resize(size_c);
  memset(host_c.data(), 0, size_c);

  errorStatus = cudaStreamCreate(&cudaStream);
  if (errorStatus) {
    std::cerr << "Failed to create cudaStream! (" << errorStatus << ")"
              << std::endl;
  }

  cudaMalloc(&dev_a, size_a * sizeof(Ty_));
  cudaMalloc(&dev_b, size_b * sizeof(Ty_));

  errorStatus =
      cudaMemcpy(dev_a, host_a, sizeof(Ty_) * size_a, cudaMemcpyHostToDevice);
  if (errorStatus) {
    std::cerr << "Failed to copy from host to device for device pointer a! ("
              << errorStatus << ")" << std::endl;
  }
  errorStatus =
      cudaMemcpy(dev_b, host_b, sizeof(Ty_) * size_b, cudaMemcpyHostToDevice);
  if (errorStatus) {
    std::cerr << "Failed to copy from host to device for device pointer b! ("
              << errorStatus << ")" << std::endl;
  }

  cudaMalloc(&dev_c, size_c * sizeof(Ty_));

  uint32_t threadsPerBlock = deviceProps.maxThreadsPerBlock;
  dim3 threads(sqrt(threadsPerBlock), sqrt(threadsPerBlock));
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
  matmul_kernel<<<blocks, threads, 0, cudaStream>>>(dev_a, dev_b, dev_c, M, N,
                                                    K);
  cudaStreamSynchronize(cudaStream);

  errorStatus = cudaMemcpy(host_c.data(), dev_c, size_c * sizeof(Ty_),
                           cudaMemcpyDeviceToHost);
  if (errorStatus) {
    std::cerr << "Failed to copy from device to host for host pointer c! ("
              << errorStatus << ")" << std::endl;
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaStreamDestroy(cudaStream);
  return host_c;
}

template <typename Ty_>
__host__ std::vector<Ty_> matmul_cuda_SHARED(const Ty_ *host_a,
                                             const Ty_ *host_b, uint32_t M,
                                             uint32_t N, uint32_t K) {
  cudaError_t errorStatus;
  cudaStream_t cudaStream;
  cudaDeviceProp deviceProps;
  cudaGetDeviceProperties(&deviceProps, 0);

  size_t size_a = M * K;
  size_t size_b = K * N;
  size_t size_c = M * N;

  Ty_ *dev_a = nullptr, *dev_b = nullptr, *host_c = nullptr;

  errorStatus = cudaStreamCreate(&cudaStream);
  if (errorStatus) {
    std::cerr << "Failed to create cudaStream! (" << errorStatus << ")"
              << std::endl;
  }

  cudaMalloc(&dev_a, size_a * sizeof(Ty_));
  cudaMalloc(&dev_b, size_b * sizeof(Ty_));
  cudaMallocHost(&host_c, size_c * sizeof(Ty_));

  errorStatus =
      cudaMemcpy(dev_a, host_a, sizeof(Ty_) * size_a, cudaMemcpyHostToDevice);
  if (errorStatus) {
    std::cerr << "Failed to copy from host to device for device pointer a! ("
              << errorStatus << ")" << std::endl;
  }
  errorStatus =
      cudaMemcpy(dev_b, host_b, sizeof(Ty_) * size_b, cudaMemcpyHostToDevice);
  if (errorStatus) {
    std::cerr << "Failed to copy from host to device for device pointer b! ("
              << errorStatus << ")" << std::endl;
  }

  uint32_t threadsPerBlock = deviceProps.maxThreadsPerBlock;
  dim3 threads(sqrt(threadsPerBlock), sqrt(threadsPerBlock));
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

  matmul_kernel<<<blocks, threads, 0, cudaStream>>>(dev_a, dev_b, host_c, M, N,
                                                    K);
  cudaStreamSynchronize(cudaStream);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaStreamDestroy(cudaStream);
  return std::vector<Ty_>(host_c, host_c + size_c);
}

template <typename Ty_>
__host__ std::vector<Ty_> matmul_cuda(const Ty_ *host_a, const Ty_ *host_b,
                                      uint32_t M, uint32_t N, uint32_t K) {
  DeviceMemoryStatus memoryStatus{};
  size_t operandAllocSize = (M * K * sizeof(Ty_)) / (1024 * 1024) +
                            (K * N * sizeof(Ty_)) / (1024 * 1024);
  size_t resultAllocSize = (M * N * sizeof(Ty_)) / (1024 * 1024);
  uint64_t hostMemorySize = getTotalSystemMemory();

  if (operandAllocSize + resultAllocSize >= memoryStatus.mFreeAmount) {
    std::cerr << "Not enough VRAM for the process!\n";
    return {};
  } else if (hostMemorySize > operandAllocSize + resultAllocSize &&
             memoryStatus.mFreeAmount > operandAllocSize + resultAllocSize) {
    return matmul_cuda_SHARED(host_a, host_b, M, N, K);
  } else if (hostMemorySize > operandAllocSize + resultAllocSize) {
    std::clog << "only VRAM is enough\n";
    return matmul_cuda_VRAM(host_a, host_b, M, N, K);
  } else {
    return {};
  }
}

template <typename Ty_>
__host__ std::vector<Ty_> matmul_cublas(const Ty_ *A, const Ty_ *B, uint32_t M,
                                        uint32_t N, uint32_t K) {
  cudaError_t cudaStatus = cudaSuccess;
  cudaStream_t stream;
  cudaStatus = cudaStreamCreate(&stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to create stream!" << std::endl;
    cudaStreamDestroy(stream);
    return {};
  }

  Ty_ *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  cudaMallocAsync(&dev_a, M * K * sizeof(Ty_), stream);
  cudaMallocAsync(&dev_b, K * N * sizeof(Ty_), stream);

  cudaMallocAsync(&dev_c, M * N * sizeof(Ty_), stream);

  cudaStatus = cudaMemcpyAsync(dev_a, A, M * K * sizeof(Ty_),
                               cudaMemcpyHostToDevice, stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed memcpy!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(dev_c);
    cudaStreamDestroy(stream);

    return {};
  }
  cudaStatus = cudaMemcpyAsync(dev_b, B, K * N * sizeof(Ty_),
                               cudaMemcpyHostToDevice, stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed memcpy!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(dev_c);
    cudaStreamDestroy(stream);

    return {};
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  Ty_ alpha = 1.0, beta = 0.0;

  if (std::is_same<Ty_, float>::value) {

    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dev_a, M,
                   dev_b, K, &beta, dev_c, M);

  } else {

    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dev_a, M,
                   dev_b, K, &beta, dev_c, M);
  }
  std::vector<Ty_> res = std::vector<Ty_>(M * N);
  cudaMemcpyAsync(res.data(), dev_c, M * N * sizeof(Ty_),
                  cudaMemcpyDeviceToHost, stream);
  cudaDeviceSynchronize();

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaStreamDestroy(stream);

  return res;
}

template <typename Ty_>
__host__ std::vector<Ty_> matmul_flat(const Ty_ *A, const Ty_ *B, uint32_t M,
                                      uint32_t N, uint32_t K) {
  std::vector<Ty_> C(M * N, 0);

  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t k = 0; k < K; k++) {
      Ty_ a_ik = A[i * K + k];
      for (uint32_t j = 0; j < N; j++) {
        C[i * N + j] += a_ik * B[k * N + j];
      }
    }
  }

  return C;
}

// Explicit instantiations for float and double
template std::vector<float> matmul_cuda_VRAM<float>(float const *,
                                                    float const *, uint32_t,
                                                    uint32_t, uint32_t);
template std::vector<double> matmul_cuda_VRAM<double>(double const *,
                                                      double const *, uint32_t,
                                                      uint32_t, uint32_t);

template std::vector<float> matmul_cuda_SHARED<float>(float const *,
                                                      float const *, uint32_t,
                                                      uint32_t, uint32_t);
template std::vector<double> matmul_cuda_SHARED<double>(double const *,
                                                        double const *,
                                                        uint32_t, uint32_t,
                                                        uint32_t);

template std::vector<float> matmul_cuda<float>(float const *, float const *,
                                               uint32_t, uint32_t, uint32_t);
template std::vector<double> matmul_cuda<double>(double const *, double const *,
                                                 uint32_t, uint32_t, uint32_t);

template std::vector<float> matmul_cublas<float>(float const *, float const *,
                                                 uint32_t, uint32_t, uint32_t);

template std::vector<float> matmul_flat<float>(float const *, float const *,
                                               uint32_t, uint32_t, uint32_t);
template std::vector<double> matmul_flat<double>(double const*, double const *,
                                                 uint32_t, uint32_t, uint32_t);

} // namespace cudavec