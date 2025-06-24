cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NeuralNetwork LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(app/main.cpp)

target_link_libraries(
  CUDAToolkit::cudart
  CUDAToolkit::cublas
  CUDAToolkit::cudnn
)

target_include_directories(
  ${PROJECT_SOURCE_DIR}/header
  ${PROJECT_SOURCE_DIR}/chessData
)