#include <core.cuh>
#include <kernel.cuh>

#include <iostream>

namespace cudavec {

DeviceMemoryStatus::DeviceMemoryStatus() {
  cudaError_t cudaStatus;

  cudaStatus = cudaMemGetInfo(&mFreeAmount, &mTotalAmount);
  if (cudaStatus) {
    std::clog
        << "Failed to retrieve memory info from the current context device\n"
        << std::flush;
  }
  mUsedAmount = mTotalAmount - mFreeAmount;
}

__host__ std::ostream &operator<<(std::ostream &stream,
                                  const DeviceMemoryStatus &memStatus) {
  stream << "Free amount: " << (memStatus.mFreeAmount / c_Megabyte) << "MB"
         << std::endl
         << "Used amount: " << (memStatus.mUsedAmount / c_Megabyte) << "MB"
         << std::endl
         << "Total available amount: " << (memStatus.mTotalAmount / c_Megabyte)
         << "MB" << std::flush;
  return stream;
}

__host__ std::ostream &operator<<(std::ostream &stream,
                                  const cudaDeviceProp &devProps) {
  stream
      << "Device Properties:\n"
      << "--------------------------------\n"
      << "Device name: " << devProps.name << "\n"
      << "totalGlobalMem: " << (devProps.totalGlobalMem / c_Megabyte)
      << " megabytes" << "\n"
      << "sharedMemPerBlock: " << devProps.sharedMemPerBlock << " bytes"
      << "\n"
      // << "regsPerBlock: " << devProps.regsPerBlock << "\n"
      << "maxThreadsPerBlock: " << devProps.maxThreadsPerBlock << " threads"
      << "\n"
      << "maxThreadsDim(x): " << devProps.maxThreadsDim[0] << " threads"
      << "\n"
      // << "maxThreadsDim(y): " << devProps.maxThreadsDim[1] << " threads" <<
      // "\n"
      // << "maxThreadsDim(z): " << devProps.maxThreadsDim[2] << " threads" <<
      // "\n"
      << "maxGridSize(x): " << devProps.maxGridSize[0] << " grids"
      << "\n"
      // << "maxGridSize(y): " << devProps.maxGridSize[1] << " grids" << "\n"
      // << "maxGridSize(z): " << devProps.maxGridSize[2] << " grids" << "\n"
      << "major CUDA compute capability: " << devProps.major << "\n"
      << "minor CUDA compute capability: " << devProps.minor << "\n"
      << "multiProcessorCount: " << devProps.multiProcessorCount
      << " processors" << "\n"
      << "memoryBusWidth: " << devProps.memoryBusWidth << " bits"
      << "\n"
      // << "l2CacheSize: " << (devProps.l2CacheSize / 1000000) << "MB" << "\n"
      << "maxThreadsPerMultiProcessor: " << devProps.maxThreadsPerMultiProcessor
      << " threads" << "\n"
      << "--------------------------------";
  return stream;
}

__host__ void CUDAContextInit(int device) {
  cudaError_t cudaStatus = cudaSuccess;

  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to set device! (incompatible GPU?)" << std::endl;
    return;
  }

  cudaDeviceProp deviceProps;
  if (cudaStatus != cudaSuccess) {
    std::cerr << "Failed to retrieve device properties! (incompatible GPU?)"
              << std::endl;
    return;
  }

  std::clog << "Initializing context for device: " << std::endl;

  kernelWarmup<<<1, 1>>>();
  cudaDeviceSynchronize();

  cudaStatus = cudaGetDeviceProperties(&deviceProps, 0);
  std::clog << deviceProps << std::endl;
}
} // namespace cudavec