# utils — header-only C++26 utility library (vendored)
# Source: https://github.com/SuperOptimizer/utils

find_package(nlohmann_json REQUIRED)

add_library(utils INTERFACE)
target_include_directories(utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(utils INTERFACE cxx_std_26)
target_link_libraries(utils INTERFACE nlohmann_json::nlohmann_json)

# Optional: find curl for http_fetch.hpp
find_package(CURL QUIET)
if(CURL_FOUND)
    target_link_libraries(utils INTERFACE CURL::libcurl)
    target_compile_definitions(utils INTERFACE UTILS_HAS_CURL=1)
endif()

# Individual header targets (for granular dependency management)
set(UTILS_HEADERS
    hash
    lru_cache
    lock_pool
    priority_queue
    compositing
    thread_pool
    zarr
    http_fetch
    compression
)

foreach(header ${UTILS_HEADERS})
    add_library(utils_${header} INTERFACE)
    target_include_directories(utils_${header} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(utils_${header} INTERFACE cxx_std_26)
endforeach()

# Link curl for http_fetch and zarr
if(CURL_FOUND)
    target_link_libraries(utils_http_fetch INTERFACE CURL::libcurl)
    target_link_libraries(utils_zarr INTERFACE CURL::libcurl)
endif()

# zarr needs nlohmann_json
target_link_libraries(utils_zarr INTERFACE nlohmann_json::nlohmann_json)

# Optional: find compression libraries for compression.hpp / zarr.hpp
# NOTE: blosc2 is not used. The project uses the bundled c-blosc (blosc1).
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
    target_link_libraries(utils_zarr INTERFACE ${UTILS_COMPRESSION_LIBS})
endif()

# Tests
if(UTILS_BUILD_TESTS OR VC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
