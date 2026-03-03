# utils — header-only C++23 utility library (vendored)
# Source: https://github.com/SuperOptimizer/utils

add_library(utils INTERFACE)
target_include_directories(utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(utils INTERFACE cxx_std_23)

# Individual header targets (for granular dependency management)
set(UTILS_HEADERS
    thread_pool
    compression
)

foreach(header ${UTILS_HEADERS})
    add_library(utils_${header} INTERFACE)
    target_include_directories(utils_${header} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(utils_${header} INTERFACE cxx_std_23)
endforeach()

# Optional: find compression libraries for compression.hpp
find_library(ZSTD_LIB zstd QUIET)
find_library(LZ4_LIB lz4 QUIET)
find_library(ZLIB_LIB z QUIET)
set(UTILS_COMPRESSION_LIBS "")
if(ZSTD_LIB)
    list(APPEND UTILS_COMPRESSION_LIBS ${ZSTD_LIB})
endif()
if(LZ4_LIB)
    list(APPEND UTILS_COMPRESSION_LIBS ${LZ4_LIB})
endif()
if(ZLIB_LIB)
    list(APPEND UTILS_COMPRESSION_LIBS ${ZLIB_LIB})
endif()
if(UTILS_COMPRESSION_LIBS)
    target_link_libraries(utils INTERFACE ${UTILS_COMPRESSION_LIBS})
    target_link_libraries(utils_compression INTERFACE ${UTILS_COMPRESSION_LIBS})
endif()

# Tests
if(VC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
