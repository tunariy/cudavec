add_library(cudavec_core STATIC
    core/core.cu
    wrappers/matrix.cu
    wrappers/operand.cu
)

target_link_libraries(cudavec_core
    PUBLIC
    CUDA::cublas
    CUDA::cudart
    Core
    Logger
    Timer
)

target_include_directories(cudavec_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/wrappers
    PRIVATE
    ${Benchtools_INCLUDE_DIRS}
)