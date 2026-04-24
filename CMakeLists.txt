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

include(FetchContent)
FetchContent_Declare(
    Benchtools
    GIT_REPOSITORY https://github.com/tunariy/benchmark.git
    GIT_TAG v1.0.0
)

FetchContent_MakeAvailable(Benchtools)

add_subdirectory(src)
add_executable(cudavec main.cpp)

# set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(cudavec
    PRIVATE
    cudavec_core             
    CUDA::cublas             
    CUDA::cudart             
    Core         
    Logger
    Timer
    spdlog::spdlog
)