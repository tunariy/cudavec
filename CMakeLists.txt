cmake_minimum_required(VERSION 3.21)
project(cudavec LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

list(APPEND CMAKE_PREFIX_PATH "/opt/cuda")
find_package(CUDAToolkit REQUIRED COMPONENTS cublas)


list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/benchtools/lib/cmake/Benchtools")
find_package(Benchtools REQUIRED)

add_subdirectory(src)
add_executable(cudavec main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(cudavec
    PRIVATE
    cudavec_core             
    CUDA::cublas             
    CUDA::cudart             
    Benchtools::Core         
    Benchtools::Logger
    Benchtools::Timer
    spdlog::spdlog
)