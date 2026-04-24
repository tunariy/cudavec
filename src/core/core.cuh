#pragma once
// cudart
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// cuBLAS
#include <cublasLt.h>
#include <cublas_v2.h>

#include <unistd.h>

namespace cudavec {

/**
 * Constants for size conversion
 **/
constexpr size_t c_Kilobyte = 1024;
constexpr size_t c_Megabyte = 1024 * 1024;
constexpr size_t c_Gigabyte = 1024 * 1024 * 1024;

/**
 * \brief Struct for storing memory statistics of current device
 **/
struct DeviceMemoryStatus {
  DeviceMemoryStatus();
  size_t mFreeAmount;
  size_t mTotalAmount;
  size_t mUsedAmount;
};

#if defined(_WIN32)
inline uint64_t getTotalSystemMemory() {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
}
#elif defined(__unix__)
inline uint64_t getTotalSystemMemory() {
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}
#endif

/**
 * \brief Host function for initializing CUDA context
 **/
__host__ void CUDAContextInit(int device);

} // namespace cudavec
