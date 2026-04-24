# cudavec

- Implementation of matrix multipication _(and few other operators)_ with CUDA
- All kernel functions are wrapped/can be wrapped in a helper function
- A lazy loading function `CudaContextInit()`, which speeds up the initial kernel call 4x at worst

---

## Menu

- [Quick Start](#quick-start)
- [CUDA API Approach](#cuda-api-approach)
- [Benchmarks](#benchmarks)
  - [CUDA Implementation vs Others](#cuda-vs-others-pre-cuda-129)
  - [CUDA Kernel vs Others](#kernel-benchmarking)
  - [Lazy Loading Improvements](#lazy-loading-improvement)

---

## Quick start

### Base Requirements

- CUDA ```CUDA Runtime 12.0``` or higher
- GPU NVIDIA® GPU Geforce® 1000 series+ or NVIDIA® Workstation GPU series

### Setup

Clone:

```bash
git clone https://github.com/tunariy/cudavec.git
```

Build:

```bash
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -S . -B build
cmake --build build 
```

Run:

```bash
# on UNIX
./build/cudavec
# or on Windows
./build/cudavec.exe
```

---

### Guide on usage

- Call the lazyloading function for better performance for the initial CUDA calls

```cpp
  CUDAContextInit();
```

- Call the cuda matrix multiplication functions

```cpp
    std::vector<float> A...;
    std::vector<float> B...;
    auto res1 = matmul_cuda(A.data(), B.data(), dim, dim, dim);
// which decays to either
    auto res1 = matmul_cuda_SHARED(A.data(), B.data(), dim, dim, dim);
// or
    auto res1 = matmul_cuda_VRAM(A.data(), B.data(), dim, dim, dim);
```

- or any other cuda arithmetic wrapper functions

```cpp
    auto res1 = performOperator(A, B, addKernel);
```

```cpp
    auto res1 = performOperator(A, 5, addKernel);
```

---

## CUDA API Approach

```cpp
 // Streams for async allocation, copy etc.
 cudaStream_t stream;
 cudaStatus = cudaStreamCreate(&stream);
 
 cudaMallocAsync(&dev_a, size_a * sizeof(Ty_), stream);
 cudaMallocAsync(&dev_b, size_b * sizeof(Ty_), stream);

 // a section RAM is allocated for shared usage between CPU & GPU
 cudaMallocHost(&c, M * N * sizeof(Ty_));
```

- Pinned memory usage will of course consume a lot of RAM and the memory availability depends on the system. A switch will be implemented.

```cpp
 // VRAM allocation
 cudaStatus = cudaMemcpyAsync(dev_a, a, size_a * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
 cudaStatus = cudaMemcpyAsync(dev_b, b, size_b * sizeof(Ty_), cudaMemcpyHostToDevice, stream);
```

### Kernel launch configuration

- Kernel launch configuration is calculated at runtime using the device properties.

```cpp
  uint32_t threadsPerBlock = deviceProps.maxThreadsPerBlock;
  dim3 threads(sqrt(threadsPerBlock), sqrt(threadsPerBlock));
  dim3 blocks((N + threads.x - 1) / threads.x,
   (M + threads.y - 1) / threads.y);
 matmul_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (dev_a, dev_b, c, M, N, K);
```

---

## Benchmarks

### Test Methodology

- Even matrices of varying size are multiplied
- Each calculation is timed with it's wrapper function

### Specs

- CPU: Intel I9-14900HX
- GPU: RTX 4060 Mobile
- RAM: 32GB DDR5 5600mHz

### Configuration

- CUDA Toolkit Version 12.9
- Compiler: MSVC + nvcc
- Launch configuration: Release mode
- ```/O2``` and ```-use_fast_math``` enabled
- CUDA Wrapper used: `matmul_cuda_SHARED`

### CUDA vs Others (Pre CUDA 12.9)

![graph smh](.github/benchgraph.png)

- 80x speed up on GPU compared to CPU and 18x compared to AVX Instructions
- Comparable performance with cuBLAS
- However it's important to note that cuBLAS has an overhead of streaming the results back to the CPU

### CUDA vs Others (CUDA 13.0)

![icantcompetewitha4trilliondollarcompany](.github/benchgraph1.png)

### Kernel Benchmarking

TODO

---

### Lazy Loading Improvement

- `CudaContextInit()` is a lazy loading function, which speeds up the initial kernel call 4x at worst

- Call this at the start before anything to speed up the initial kernel call.

#### Setup for the Test

- With lazy loading

```bash
CUDA:
Duration(ms): 7ms
Duration(ns): 7033400ns
CUDA:
Duration(ms): 6ms
Duration(ns): 6937900ns
```

- Without lazy loading

```bash
CUDA:
Duration(ms): 77ms
Duration(ns): 77046496ns
CUDA:
Duration(ms): 7ms
Duration(ns): 7368700ns
```

---
