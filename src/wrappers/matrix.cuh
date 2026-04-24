#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <core.cuh>
#include <kernel.cuh>

#include <cstdint>
#include <vector>

namespace cudavec {

template <typename Ty_>
__host__ std::vector<Ty_> matmul_cuda_VRAM(const Ty_ *host_a, const Ty_ *host_b,
                                           uint32_t M, uint32_t N, uint32_t K);

template <typename Ty_>
__host__ std::vector<Ty_> matmul_cuda_SHARED(const Ty_ *host_a, const Ty_ *host_b,
                                             uint32_t M, uint32_t N, uint32_t K);

template <typename Ty_>
__host__ std::vector<Ty_> matmul_cuda(const Ty_ *host_a, const Ty_ *host_b,
                                      uint32_t M, uint32_t N, uint32_t K);

template <typename Ty_>
__host__ std::vector<Ty_> matmul_cublas(const Ty_ *A, const Ty_ *B, uint32_t M,
                                        uint32_t N, uint32_t K);

template <typename Ty_>
__host__ std::vector<Ty_> matmul_flat(const Ty_ *A, const Ty_ *B, uint32_t M,
                                      uint32_t N, uint32_t K);

}