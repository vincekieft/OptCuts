cmake_minimum_required(VERSION 3.16)
project(xapack LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(xapack
  xapack.cpp
  ext/xatlas/xatlas.cpp
  ext/fast_obj/fast_obj.c
)
target_include_directories(xapack PRIVATE ext/xatlas ext/fast_obj)
