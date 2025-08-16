cmake_minimum_required(VERSION 3.15)
project(sdf120 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If installed via pip, point CMake to pybind11:
# cmake -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" ..
find_package(pybind11 CONFIG REQUIRED)

# ✅ No “COMPONENTS Core” here
find_package(CGAL REQUIRED)

pybind11_add_module(sdf120 sdf120.cpp)
target_link_libraries(sdf120 PRIVATE CGAL::CGAL)

# Optional: if your CGAL was built with TBB and you want parallelism
# find_package(TBB)
# if(TBB_FOUND)
#   target_link_libraries(sdf120 PRIVATE TBB::tbb)
#   target_compile_definitions(sdf120 PRIVATE CGAL_LINKED_WITH_TBB)
# endif()
