#include <cudavec.cuh>

// Custom library
#include <benchmark.h>

using namespace benchtools;
using namespace cudavec;

int main() {
#if 1
	Logger logger{ "log.txt", OVERWRITE };
	CUDAContextInit(0);
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
		std::clog << "Testing: " << dim << "x" << dim << ".\n";
		logger.log(std::to_string(size));
		std::chrono::duration<double> dur1, dur2, dur3, dur4;
		for (uint32_t i = 0; i < sample_size; i++) {
			std::vector<float> res1;
			{
				benchtools::Timer timer;
				res1 = matmul_cublas(A.data(), B.data(), dim, dim, dim);
			} dur1 = LAST_DURATION;

			{
				benchtools::Timer timer;
				res1 = matmul_cuda(A.data(), B.data(), dim, dim, dim);
			} dur2 = LAST_DURATION;
			{
				benchtools::Timer timer;
				res1 = matmul_flat(A.data(), B.data(), dim, dim, dim);
			} dur3 = LAST_DURATION;

#if OS_WINDOWS
			{
				benchtools::Timer timer;
				res1 = matmul_avx(A.data(), B.data(), dim, dim, dim);
#endif
			} dur4 = LAST_DURATION;
		}

		logger.log(std::to_string(durationCast(dur1, timeunit::nanosecond).count()));
		logger.log(std::to_string(durationCast(dur2, timeunit::nanosecond).count()));
		logger.log(std::to_string(durationCast(dur3, timeunit::nanosecond).count()));
		logger.log(std::to_string(durationCast(dur4, timeunit::nanosecond).count()));
	}
#elif 0
	Logger logger{ "log.txt", OVERWRITE };
	CUDAContextInit(0);
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
		std::clog << "Testing: " << dim << "x" << dim << ".\n";
		logger.log(std::to_string(size));
		std::chrono::duration<double> dur1, dur2, dur3, dur4;
		for (uint32_t i = 0; i < sample_size; i++) {
			std::vector<float> res1;
			{
				// benchtools::Timer timer;
				res1 = matmul_cublas(A.data(), B.data(), dim, dim, dim);
			} dur1 = LAST_DURATION;

			{
				// benchtools::Timer timer;
				res1 = matmul_cuda(A.data(), B.data(), dim, dim, dim);
			} dur2 = LAST_DURATION;
		}

		logger.log(std::to_string(durationCast(dur1, timeunit::nanosecond).count()));
		logger.log(std::to_string(durationCast(dur2, timeunit::nanosecond).count()));
		logger.log("0");
		logger.log("0");
	}
#else
#endif
}
