# Local vendored utils library
# Copied from https://github.com/SuperOptimizer/utils
include(FetchContent)

# c-blosc vendored here has cmake_minimum_required < 3.5; newer CMake rejects it.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
endif()

# ── Compression libraries ──────────────────────────────────────────────────

# zlib
FetchContent_Declare(
    zlib
    URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
)
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# zstd (CMakeLists.txt is in build/cmake)
FetchContent_Declare(
    zstd
    URL https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
    URL_HASH SHA256=8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1
    SOURCE_SUBDIR build/cmake
)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)

# lz4 (CMakeLists.txt is in build/cmake)
FetchContent_Declare(
    lz4
    URL https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz
    URL_HASH SHA256=537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b
    SOURCE_SUBDIR build/cmake
)
set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(zlib zstd lz4)

# bzip2 (1.0.8 has no CMakeLists.txt — we build it manually)
FetchContent_Declare(
    bzip2
    URL https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
    URL_HASH SHA256=ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269
)

# xz (liblzma)
FetchContent_Declare(
    xz
    URL https://github.com/tukaani-project/xz/releases/download/v5.6.3/xz-5.6.3.tar.gz
    URL_HASH SHA256=b1d45295d3f71f25a4c9101bd7c8d16cb56348bbef3bbc738da0351e17c73317
)

# c-blosc (v1) — reuse the zlib/zstd/lz4 we already fetched
FetchContent_Declare(
    c-blosc
    URL https://github.com/Blosc/c-blosc/archive/refs/tags/v1.21.6.tar.gz
    URL_HASH SHA256=9fcd60301aae28f97f1301b735f966cc19e7c49b6b4321b839b4579a0c156f38
)
set(PREFER_EXTERNAL_ZLIB ON CACHE BOOL "" FORCE)
set(PREFER_EXTERNAL_ZSTD ON CACHE BOOL "" FORCE)
set(PREFER_EXTERNAL_LZ4 ON CACHE BOOL "" FORCE)
set(BLOSC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BLOSC_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(bzip2 xz)

# Build bzip2 manually (no CMakeLists.txt in 1.0.8)
if(NOT TARGET bz2_static)
    add_library(bz2_static STATIC
        ${bzip2_SOURCE_DIR}/blocksort.c
        ${bzip2_SOURCE_DIR}/huffman.c
        ${bzip2_SOURCE_DIR}/crctable.c
        ${bzip2_SOURCE_DIR}/randtable.c
        ${bzip2_SOURCE_DIR}/compress.c
        ${bzip2_SOURCE_DIR}/decompress.c
        ${bzip2_SOURCE_DIR}/bzlib.c
    )
    target_include_directories(bz2_static PUBLIC ${bzip2_SOURCE_DIR})
    set_target_properties(bz2_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

FetchContent_MakeAvailable(c-blosc)

# Suppress warnings from all vendored subproject targets
get_property(_all_targets DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _all_targets)
    get_property(_type TARGET "${_t}" PROPERTY TYPE)
    if(_type MATCHES "STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY")
        target_compile_options("${_t}" PRIVATE -w)
    endif()
endforeach()

# ── Install helper: add vendored utils to VC's export set ─────────────────
# vc_core links publicly to vc_utils_{json,tensor,zarr}, so they must be
# in the same export set for install(EXPORT) to succeed.
# Add utils targets to the install export set so that
# install(EXPORT VC3DTargets) doesn't fail (vc_core links to these publicly).
macro(_vc_utils_install_export target)
    if(VC_INSTALL_LIBS)
        install(TARGETS ${target} EXPORT "${targets_export_name}"
            ARCHIVE DESTINATION lib COMPONENT Libraries)
    endif()
endmacro()

# ── json ───────────────────────────────────────────────────────────────────
add_library(vc_utils_json STATIC src/json.cpp)
target_include_directories(vc_utils_json PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(vc_utils_json PRIVATE -w)
_vc_utils_install_export(vc_utils_json)

# ── tensor ─────────────────────────────────────────────────────────────────
add_library(vc_utils_tensor STATIC src/tensor.cpp)
target_include_directories(vc_utils_tensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(vc_utils_tensor PRIVATE -w)
_vc_utils_install_export(vc_utils_tensor)

# ── zarr ───────────────────────────────────────────────────────────────────
add_library(vc_utils_zarr STATIC src/zarr.cpp)
target_include_directories(vc_utils_zarr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_zarr PUBLIC vc_utils_tensor vc_utils_json)
target_include_directories(vc_utils_zarr PRIVATE
    ${zlib_SOURCE_DIR}
    ${zlib_BINARY_DIR}
    ${zstd_SOURCE_DIR}/lib
    ${lz4_SOURCE_DIR}/lib
    ${bzip2_SOURCE_DIR}
    ${xz_SOURCE_DIR}/src/liblzma/api
    ${c-blosc_SOURCE_DIR}/blosc
    ${c-blosc_BINARY_DIR}/blosc
)
target_link_libraries(vc_utils_zarr PRIVATE
    zlibstatic libzstd_static lz4_static bz2_static liblzma blosc_static
)
target_compile_options(vc_utils_zarr PRIVATE -w)
_vc_utils_install_export(vc_utils_zarr)

# ── logging ───────────────────────────────────────────────────────────────
add_library(vc_utils_logging STATIC src/logging.cpp)
target_include_directories(vc_utils_logging PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(vc_utils_logging PRIVATE -w)

# ── timer ─────────────────────────────────────────────────────────────────
add_library(vc_utils_timer STATIC src/timer.cpp)
target_include_directories(vc_utils_timer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_timer PUBLIC vc_utils_logging)
target_compile_options(vc_utils_timer PRIVATE -w)

# ── argparse ──────────────────────────────────────────────────────────────
add_library(vc_utils_argparse STATIC src/argparse.cpp)
target_include_directories(vc_utils_argparse PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(vc_utils_argparse PRIVATE -w)

# ── kdtree ────────────────────────────────────────────────────────────────
add_library(vc_utils_kdtree STATIC src/kdtree.cpp)
target_include_directories(vc_utils_kdtree PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(vc_utils_kdtree PRIVATE -w)

# ── rtree ─────────────────────────────────────────────────────────────────
add_library(vc_utils_rtree STATIC src/rtree.cpp)
target_include_directories(vc_utils_rtree PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(vc_utils_rtree PRIVATE -w)

# ── interpolation ─────────────────────────────────────────────────────────
add_library(vc_utils_interpolation STATIC src/interpolation.cpp)
target_include_directories(vc_utils_interpolation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_interpolation PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_interpolation PRIVATE -w)

# ── linalg ────────────────────────────────────────────────────────────────
add_library(vc_utils_linalg STATIC src/linalg.cpp)
target_include_directories(vc_utils_linalg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_linalg PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_linalg PRIVATE -w)

# ── tiff ──────────────────────────────────────────────────────────────────
add_library(vc_utils_tiff STATIC src/tiff.cpp)
target_include_directories(vc_utils_tiff PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_tiff PUBLIC vc_utils_tensor)
target_include_directories(vc_utils_tiff PRIVATE
    ${zlib_SOURCE_DIR}
    ${zlib_BINARY_DIR}
    ${zstd_SOURCE_DIR}/lib
    ${lz4_SOURCE_DIR}/lib
)
target_link_libraries(vc_utils_tiff PRIVATE zlibstatic libzstd_static lz4_static)
target_compile_options(vc_utils_tiff PRIVATE -w)

# ── connected_components ──────────────────────────────────────────────────
add_library(vc_utils_connected_components STATIC src/connected_components.cpp)
target_include_directories(vc_utils_connected_components PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_connected_components PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_connected_components PRIVATE -w)

# ── distance_transform ────────────────────────────────────────────────────
add_library(vc_utils_distance_transform STATIC src/distance_transform.cpp)
target_include_directories(vc_utils_distance_transform PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_distance_transform PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_distance_transform PRIVATE -w)

# ── thinning ──────────────────────────────────────────────────────────────
add_library(vc_utils_thinning STATIC src/thinning.cpp)
target_include_directories(vc_utils_thinning PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_thinning PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_thinning PRIVATE -w)

# ── compositing ───────────────────────────────────────────────────────────
add_library(vc_utils_compositing STATIC src/compositing.cpp)
target_include_directories(vc_utils_compositing PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_compositing PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_compositing PRIVATE -w)

# ── convolution ───────────────────────────────────────────────────────────
add_library(vc_utils_convolution STATIC src/convolution.cpp)
target_include_directories(vc_utils_convolution PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_convolution PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_convolution PRIVATE -w)

# ── morphology ────────────────────────────────────────────────────────────
add_library(vc_utils_morphology STATIC src/morphology.cpp)
target_include_directories(vc_utils_morphology PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_morphology PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_morphology PRIVATE -w)

# ── histogram ────────────────────────────────────────────────────────────
add_library(vc_utils_histogram STATIC src/histogram.cpp)
target_include_directories(vc_utils_histogram PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_histogram PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_histogram PRIVATE -w)

# ── resize ───────────────────────────────────────────────────────────────
add_library(vc_utils_resize STATIC src/resize.cpp)
target_include_directories(vc_utils_resize PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(vc_utils_resize PUBLIC vc_utils_tensor)
target_compile_options(vc_utils_resize PRIVATE -w)
