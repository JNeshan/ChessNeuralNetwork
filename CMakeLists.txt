cmake_minimum_required(VERSION 3.18)
project(ChessNetwork LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(app/main.cpp)

target_link_libraries(
  CUDAToolkit::cudart
  CUDAToolkit::cublas
  CUDAToolkit::cudnn
)

target_include_directories(
  ${PROJECT_SOURCE_DIR}/header
)