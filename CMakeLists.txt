cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(pfac LANGUAGES CXX CUDA)

add_compile_options(-G)
#add_compile_options(-arch compute_61)
add_compile_options(--expt-relaxed-constexpr)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) # $TODO - remove
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS -pthread)

set(SOURCES main.cu src/util.cu)

add_executable(pfac ${SOURCES})

target_include_directories(pfac PRIVATE "include")
target_compile_features(pfac PUBLIC cxx_std_11)

set_target_properties(pfac
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )


