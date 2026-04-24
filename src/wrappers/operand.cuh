#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <kernel.cuh>

#include <iostream>
#include <vector>

namespace cudavec {

/**
 * \brief Perform an operator on two vectors
 * \param a first operand vector
 * \param b second operand vector
 * \param kernelFunction kernel operator function to be called
 **/
template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_> &a,
                                          const std::vector<Ty_> &b,
                                          KernelFunc kernelFunction) {
  cudaError_t cudaStatus = cudaSuccess;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
    return {};
  }

  Ty_ *dev_a = nullptr, *dev_b = nullptr;
  Ty_ *c = nullptr;

  size_t size = a.size() > b.size() ? b.size() : a.size();

  cudaStream_t stream;
  cudaStatus = cudaStreamCreate(&stream);

  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to create stream!" << std::endl;
    cudaStreamDestroy(stream);
    return {};
  }

  cudaMallocHost(&c, size * sizeof(Ty_));
  cudaMalloc(&dev_a, size * sizeof(Ty_));
  cudaMalloc(&dev_b, size * sizeof(Ty_));

  cudaStatus = cudaMemcpyAsync(dev_a, a.data(), size * sizeof(Ty_),
                               cudaMemcpyHostToDevice, stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed memcpy!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(c);
    cudaStreamDestroy(stream);

    return {};
  }

  cudaStatus = cudaMemcpyAsync(dev_b, b.data(), size * sizeof(Ty_),
                               cudaMemcpyHostToDevice, stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed memcpy!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(c);
    cudaStreamDestroy(stream);

    return {};
  }

  // Kernel launch configuration
  dim3 blocksPerGrid(1024);
  dim3 threadsPerBlock(size / 1024);
  kernelFunction<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(c, dev_a, dev_b,
                                                                size);

  cudaStatus = cudaStreamSynchronize(stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to synchronize streams!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(c);
    cudaStreamDestroy(stream);

    return {};
  }

  std::vector<Ty_> res(c, c + size);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFreeHost(c);
  cudaStatus = cudaStreamDestroy(stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to destroy stream!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(c);
    cudaStreamDestroy(stream);

    return {};
  }

  return res;
}

/**
 * \brief Perform an operator on a vector with an constant operand value
 * \param a first operand vector
 * \param b operand value
 * \param kernelFunction kernel operator function to be called
 **/
template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_> &a,
                                          const Ty_ &b,
                                          KernelFunc kernelFunction) {
  cudaError_t cudaStatus = cudaSuccess;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
    return {};
  }

  Ty_ *dev_a = nullptr, *dev_b = nullptr;

  size_t size = a.size();

  Ty_ *c = nullptr;

  cudaStream_t stream;
  cudaStatus = cudaStreamCreate(&stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to create stream!" << std::endl;
    cudaStreamDestroy(stream);
    return {};
  }

  cudaMallocHost(&c, size * sizeof(Ty_));
  cudaMalloc(&dev_a, size * sizeof(Ty_));
  cudaMalloc(&dev_b, sizeof(Ty_));

  cudaStatus = cudaMemcpyAsync(dev_a, a.data(), size * sizeof(Ty_),
                               cudaMemcpyHostToDevice, stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed memcpy!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(c);
    cudaStreamDestroy(stream);

    return {};
  }

  cudaStatus =
      cudaMemcpyAsync(dev_b, &b, sizeof(Ty_), cudaMemcpyHostToDevice, stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed memcpy!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(c);
    cudaStreamDestroy(stream);

    return {};
  }

  // Kernel launch configuration
  dim3 blocksPerGrid(1024);
  dim3 threadsPerBlock(size / 1024);
  kernelFunction<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(c, dev_a, dev_b,
                                                                size);

  cudaStatus = cudaStreamSynchronize(stream);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to synchronize streams!" << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(c);
    cudaStreamDestroy(stream);

    return {};
  }

  std::vector<Ty_> res(c, c + size);

  // Cleanup
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFreeHost(c);
  cudaStatus = cudaStreamDestroy(stream);

  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to destroy stream!" << std::endl;
    return {};
  }

  return res;
}

}; // namespace cudavec