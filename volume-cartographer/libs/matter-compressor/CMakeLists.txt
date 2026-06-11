cmake_minimum_required(VERSION 3.16)
project(matter_compressor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Standalone (top-level) vs vendored (add_subdirectory in a parent project).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(MC_STANDALONE ON)
else()
  set(MC_STANDALONE OFF)
endif()

# Only set fast/native flags + tests when built standalone. When vendored, the parent
# owns the build type/flags and may build shared libs (so we force PIC below).
if(MC_STANDALONE)
  set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math")
  add_compile_options(-Wall -Wno-stringop-overflow)
  if(NOT DEFINED MC_NO_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# Single merged TU (upstream folded mc_volume/mc_zarr/mc_s3 + codec + archive +
# sample/render into matter_compressor.{c,h}). Bundles the vendored c3d decoder;
# links VC's libs3 target + zstd for the remote-zarr streaming path.
add_library(matter_compressor STATIC
  src/matter_compressor.c src/c3d.c)
target_include_directories(matter_compressor PUBLIC src)
set_target_properties(matter_compressor PROPERTIES
  POSITION_INDEPENDENT_CODE ON C_STANDARD 23 C_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
target_link_libraries(matter_compressor PUBLIC m Threads::Threads libs3 zstd::libzstd)

# Archive determinism (byte-identical across ISAs): no -ffast-math, explicit
# -ffp-contract=off. Hot paths are integer; neither flag costs performance.
# -DMC_NO_NATIVE=ON for portable binaries (distribution / cross-build).
target_compile_options(matter_compressor PRIVATE -O3 -ffp-contract=off -Wno-stringop-overflow)
if(NOT MC_NO_NATIVE)
  include(CheckCCompilerFlag)
  check_c_compiler_flag(-march=native MC_HAS_MARCH_NATIVE)
  if(MC_HAS_MARCH_NATIVE)
    target_compile_options(matter_compressor PRIVATE -march=native)
  endif()
endif()
# c3d: NO -fopenmp (single-threaded; mc_volume owns its own pthread pools, and an
# OpenMP team per decode would multiply threads by nproc). c3d's omp_ calls are
# all _OPENMP-guarded with serial fallbacks, so no OpenMP = the serial path.
set_source_files_properties(src/c3d.c PROPERTIES COMPILE_OPTIONS "-O3;-ffp-contract=off")

# Back-compat alias: consumers that linked the old split lib still resolve.
add_library(matter_compressor_volume ALIAS matter_compressor)

# Tests + native flags only when built standalone (vendoring consumers don't want our
# test exes added to their build).
if(MC_STANDALONE)
  enable_testing()
  add_executable(mc_roundtrip tests/mc_roundtrip.c)
  target_link_libraries(mc_roundtrip matter_compressor)
  add_test(NAME roundtrip COMMAND mc_roundtrip)

  # appendable round-trip: open -> append -> close -> reopen -> append -> decode + verify.
  add_executable(mc_append_roundtrip tests/mc_append_roundtrip.c)
  target_link_libraries(mc_append_roundtrip matter_compressor)
  add_test(NAME append_roundtrip COMMAND mc_append_roundtrip)

  add_executable(mc_stream_partial tests/mc_stream_partial.c)
  target_link_libraries(mc_stream_partial matter_compressor)
  add_test(NAME stream_partial COMMAND mc_stream_partial)

  add_executable(mc_v6_test tests/mc_v6_test.c)
  target_link_libraries(mc_v6_test matter_compressor)
  add_test(NAME v6 COMMAND mc_v6_test)

  add_executable(mc_cache_test tests/mc_cache_test.c)
  target_link_libraries(mc_cache_test matter_compressor)
  add_test(NAME cache COMMAND mc_cache_test)

  add_executable(mc_stream_volume_test tests/mc_stream_volume_test.c)
  target_link_libraries(mc_stream_volume_test matter_compressor)
  add_test(NAME stream_volume COMMAND mc_stream_volume_test)

  # archive repacker: Morton order + clustered index for streaming + offline
  add_executable(mc_export tools/mc_export.c)
  target_include_directories(mc_export PRIVATE src)
  target_link_libraries(mc_export m Threads::Threads)

  # tools: benchmark harness (no extra deps) + zarr fetcher (libcurl + libzstd).
  add_executable(mc_bench tools/mc_bench.c)
  target_link_libraries(mc_bench matter_compressor)

  # fysics-style aggressive air masking preprocessor (no deps)
  add_executable(mc_mask tools/mc_mask.c)

  add_executable(mc_verify tools/mc_verify.c)
  target_link_libraries(mc_verify matter_compressor)

  find_package(CURL QUIET)
  find_library(ZSTD_LIB zstd HINTS /opt/homebrew/lib /usr/local/lib)
  find_path(ZSTD_INC zstd.h HINTS /opt/homebrew/include /usr/local/include)
  if(CURL_FOUND AND ZSTD_LIB AND ZSTD_INC)
    add_executable(mc_fetch tools/mc_fetch.c tools/vendor/libs3/libs3.c)
    target_include_directories(mc_fetch PRIVATE tools/vendor/libs3 ${ZSTD_INC})
    set_property(TARGET mc_fetch PROPERTY C_STANDARD 23)
    target_link_libraries(mc_fetch CURL::libcurl ${ZSTD_LIB})
  else()
    message(STATUS "mc_fetch skipped (need libcurl + libzstd)")
  endif()
endif()
