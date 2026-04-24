#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <kernel.cuh>

#include <vector>

namespace cudavec {

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_> &a,
                                          const std::vector<Ty_> &b,
                                          KernelFunc kernelFunction);

template <typename Ty_, typename KernelFunc>
__host__ std::vector<Ty_> performOperator(const std::vector<Ty_> &a,
                                          Ty_ &&b,
                                          KernelFunc kernelFunction);

} // namespace cudavec