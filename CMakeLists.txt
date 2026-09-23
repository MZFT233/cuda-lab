cmake_minimum_required(VERSION 3.24)
project(cuda_lab LANGUAGES CXX CUDA)

# CUDA 12.6 + MSVC 14.44 (Build Tools) + Ninja
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)   # Ada / RTX 4060 Laptop
endif()

add_executable(cuda_lab src/main.cu)
target_compile_options(cuda_lab PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
)

enable_testing()
add_test(NAME cuda_lab_runs COMMAND cuda_lab)
set_tests_properties(cuda_lab_runs PROPERTIES PASS_REGULAR_EXPRESSION "ALLOK")
