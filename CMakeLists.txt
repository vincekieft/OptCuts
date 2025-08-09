cmake_minimum_required(VERSION 3.9)
project(OptCuts)

list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/tbb-static")
find_package(TBB REQUIRED tbb)

set(OPTCUTS_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/ext")

# --- libigl locations (prebuilt) ---
set(LIBIGL_ROOT   "${OPTCUTS_EXTERNAL}/libigl")
set(LIBIGL_BUILD  "${LIBIGL_ROOT}/build")
set(LIBIGL_LIBDIR "${LIBIGL_BUILD}/lib")
set(LIBIGL_INCDIR "${LIBIGL_ROOT}/include")            # provides igl/*

# Shewchuk Triangle headers (downloaded by libigl into the build tree)
set(LIBIGL_TRIHDR "${LIBIGL_BUILD}/_deps/triangle-src") # contains triangle.h
if(NOT EXISTS "${LIBIGL_TRIHDR}/triangle.h")
  message(FATAL_ERROR "triangle.h not found at ${LIBIGL_TRIHDR}. Rebuild libigl with -DLIBIGL_WITH_TRIANGLE=ON.")
endif()

# vendored Eigen (header-only)
set(EIGEN_INC "${OPTCUTS_EXTERNAL}/eigen")

# headless: make sure nothing OpenGL-related sneaks in
add_compile_definitions(IGL_NO_OPENGL IGL_NO_OPENGL_GLFW IGL_NO_OPENGL_GLFW_IMGUI)

# --- import libigl static archives as CMake targets ---
add_library(igl::core STATIC IMPORTED GLOBAL)
set_target_properties(igl::core PROPERTIES
  IMPORTED_LOCATION             "${LIBIGL_LIBDIR}/libigl.a"
  INTERFACE_INCLUDE_DIRECTORIES "${LIBIGL_INCDIR};${EIGEN_INC}"
)

# libigl triangle wrapper + upstream triangle.a + triangle.h include dir
add_library(igl::triangle STATIC IMPORTED GLOBAL)
# prefer an explicit find; falls back to the expected path
find_library(TRIANGLE_LIB NAMES triangle libtriangle
  PATHS "${LIBIGL_LIBDIR}" NO_DEFAULT_PATH)
if(NOT TRIANGLE_LIB)
  set(TRIANGLE_LIB "${LIBIGL_LIBDIR}/libtriangle.a")
endif()
set_target_properties(igl::triangle PROPERTIES
  IMPORTED_LOCATION             "${LIBIGL_LIBDIR}/libtriangle.a"
  INTERFACE_INCLUDE_DIRECTORIES "${LIBIGL_INCDIR};${LIBIGL_TRIHDR}"
  INTERFACE_LINK_LIBRARIES      "${TRIANGLE_LIB}"
)

# libigl png wrapper + libpng + zlib (archive names vary)
# file(GLOB LIBPNG_A "${LIBIGL_LIBDIR}/libpng*.a")
# file(GLOB  ZLIB_A  "${LIBIGL_LIBDIR}/libz*.a")
# add_library(igl::png STATIC IMPORTED GLOBAL)
# set_target_properties(igl::png PROPERTIES
#   IMPORTED_LOCATION             "${LIBIGL_LIBDIR}/libigl_png.a"
#   INTERFACE_INCLUDE_DIRECTORIES "${LIBIGL_INCDIR}"
#   INTERFACE_LINK_LIBRARIES      "${LIBPNG_A};${ZLIB_A}"
# )

# --- sources ---
file(GLOB SRCFILES
  src/main.cpp
  src/Optimizer.cpp
  src/TriMesh.cpp
  src/Scaffold.cpp
  src/Energy/Energy.cpp
  src/Energy/ARAPEnergy.cpp
  src/Energy/SymDirichletEnergy.cpp
  src/Utils/IglUtils.cpp
  src/LinSysSolver/EigenLibSolver.cpp
)

add_executable(${PROJECT_NAME}_bin ${SRCFILES})

# Ensure correct headers are found (Triangle first to avoid CGAL/Embree homonyms)
target_include_directories(${PROJECT_NAME}_bin
  BEFORE PRIVATE
  ${LIBIGL_TRIHDR}
  ${LIBIGL_INCDIR}
  ${EIGEN_INC}
  src src/Energy src/Utils src/LinSysSolver src/UnitTest
)

# Link ONLY headless static bits
target_link_libraries(${PROJECT_NAME}_bin
  PRIVATE
    igl::core
    igl::triangle
    # igl::png
    TBB::tbb
)

target_compile_definitions(${PROJECT_NAME}_bin PRIVATE USE_TBB=1)

# Mostly-static runtime (no need to fully static-link glibc)
target_link_options(${PROJECT_NAME}_bin PRIVATE -static-libstdc++ -static-libgcc)
