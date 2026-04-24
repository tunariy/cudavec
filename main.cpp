#include "benchtools/Core/Time.hpp"
#include <core.cuh>
#include <matrix.cuh>

#include <benchtools/Logger/Logger.hpp>
#include <benchtools/Timers/ScopedTimer.hpp>
#include <benchtools/Timers/WallTimer.hpp>


#include <cstdint>
#include <vector>

int main() {
  benchtools::WallTimer timer;
  cudavec::CUDAContextInit(0);
  size_t sample_size = 1, size_limit = 13;
  for (uint32_t k = 1; k <= size_limit; ++k) {
    const uint32_t size = static_cast<uint32_t>(1) << k * 2;
    const uint32_t dim = static_cast<uint32_t>(1) << k;

    std::vector<float> A(size);
    std::vector<float> B(size);
    for (uint32_t i = 0; i < size; ++i) {
      A[i] = i;
      B[i] = i;
    }
    std::vector<float> res;
    for (uint32_t i = 0; i < sample_size; i++) {
      INFO("Dim: {} x {}", dim, dim);
      {
        benchtools::ScopedTimer scopeTimer{timer};
        res = cudavec::matmul_cublas(A.data(), B.data(), dim, dim, dim);
      }
      std::clog << timer.duration(benchtools::time_unit::nanoseconds).str()
                << std::endl;
      /*
      {
        benchtools::ScopedTimer scopeTimer{timer};
        res1 = cudavec::matmul_cuda(A.data(), B.data(), dim, dim, dim);
      }
      */
      {
        benchtools::ScopedTimer scopeTimer{timer};
        res = cudavec::matmul_cuda_SHARED(A.data(), B.data(), dim, dim, dim);
      }
      std::clog << timer.duration(benchtools::time_unit::nanoseconds).str()
                << std::endl;
      /*
      {
        benchtools::ScopedTimer scopeTimer{timer};
        res1 = cudavec::matmul_flat(A.data(), B.data(), dim, dim, dim);
      }
      std::clog << timer.duration(benchtools::time_unit::nanoseconds).str()
      << std::endl;
      */

#if defined(_WIN32)
      {
        res1 = matmul_avx(A.data(), B.data(), dim, dim, dim);
      }
#endif
    }
  }
}
