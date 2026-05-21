include(FetchContent)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "" FORCE)

find_package(doctest CONFIG QUIET)
if(NOT TARGET doctest::doctest)
    FetchContent_Declare(
        doctest
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        URL https://github.com/doctest/doctest/archive/v2.4.12.tar.gz
    )
    FetchContent_MakeAvailable(doctest)
endif()

find_package(Qt6 REQUIRED COMPONENTS Core Gui Test)

function(vc_add_test)
    cmake_parse_arguments(T "" "NAME" "SOURCES;LIBS;INCLUDES;DEFS" ${ARGN})
    if(NOT T_SOURCES)
        set(T_SOURCES ${T_NAME}.cpp)
    endif()
    if(NOT DEFINED T_LIBS)
        set(T_LIBS doctest::doctest vc_core)
    endif()
    if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN
       OR VC_ENABLE_LSAN OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN)
        list(APPEND T_SOURCES "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.cpp")
    endif()
    add_executable(${T_NAME} ${T_SOURCES})
    target_link_libraries(${T_NAME} PRIVATE ${T_LIBS})
    target_include_directories(${T_NAME} PRIVATE ${T_INCLUDES})
    target_compile_definitions(${T_NAME} PRIVATE
        VC_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data" ${T_DEFS})
    if(VC_USE_PCH AND NOT T_NAME MATCHES "^test_(smoke|segmentation_autosave_state)$")
        target_precompile_headers(${T_NAME} REUSE_FROM vc_core)
    endif()
    add_test(NAME ${T_NAME} COMMAND ${T_NAME})
endfunction()

vc_add_test(NAME test_smoke LIBS doctest::doctest)
vc_add_test(NAME test_appendmask_render)
vc_add_test(NAME test_chunked_plane_sampler_fallback)
vc_add_test(NAME test_volume_chunk_errors)

vc_add_test(NAME test_segmentation_autosave_state
    LIBS doctest::doctest
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_segmentation_intersection_invalidation
    LIBS Qt6::Core Qt6::Gui Qt6::Test vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_quadsurface_save_roundtrip)
vc_add_test(NAME test_atomic_imwrite_multi)
vc_add_test(NAME test_post_process)

# Pin Blosc1-readable on-disk format; guards against silent Blosc2 migration.
vc_add_test(NAME test_blosc1_compat LIBS doctest::doctest vc_core Blosc::blosc)

vc_add_test(NAME test_merge_grid_resolve LIBS doctest::doctest vc_merge_grid)

# Opt-in via VC_RUN_E2E=1; spawns the built vc_merge_tifxyz binary.
add_executable(test_merge_e2e_small test_merge_e2e_small.cpp)
target_link_libraries(test_merge_e2e_small PRIVATE doctest::doctest vc_core nlohmann_json::nlohmann_json)
if(VC_RUN_E2E)
    add_test(NAME test_merge_e2e_small COMMAND test_merge_e2e_small)
    set_tests_properties(test_merge_e2e_small PROPERTIES
        ENVIRONMENT "VC_RUN_E2E=1;VC_MERGE_TIFXYZ_BIN=$<TARGET_FILE:vc_merge_tifxyz>")
endif()

# Benches: built always, opt-in to ctest via VC_RUN_BENCHES (loose wall-clock).
add_executable(bench_volume_sample bench_volume_sample.cpp)
target_link_libraries(bench_volume_sample PRIVATE doctest::doctest vc_core)
add_executable(bench_chunked_plane_sampler bench_chunked_plane_sampler.cpp)
target_link_libraries(bench_chunked_plane_sampler PRIVATE doctest::doctest vc_core)
if(VC_RUN_BENCHES)
    add_test(NAME bench_volume_sample COMMAND bench_volume_sample)
    add_test(NAME bench_chunked_plane_sampler COMMAND bench_chunked_plane_sampler)
endif()

# Per-sanitizer canary: triggers a violation the sanitizer must report.
# Miss (canary exits 0) fails ctest — proves the sanitizer is wired.
set(_canary_srcs sanitizer_canary.cpp)
if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN OR VC_ENABLE_LSAN
   OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN)
    list(APPEND _canary_srcs "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.cpp")
endif()
add_executable(sanitizer_canary ${_canary_srcs})
set_target_properties(sanitizer_canary PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

function(_add_canary name arg regex)
    add_test(NAME ${name} COMMAND sanitizer_canary ${arg})
    set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
endfunction()

if(VC_ENABLE_ASAN)
    _add_canary(sanitizer_canary_asan  asan  "AddressSanitizer: heap-buffer-overflow")
endif()
if(VC_ENABLE_UBSAN)
    _add_canary(sanitizer_canary_ubsan ubsan "runtime error: signed integer overflow")
endif()
if(VC_ENABLE_TSAN)
    _add_canary(sanitizer_canary_tsan  tsan  "ThreadSanitizer: data race")
endif()
if(VC_ENABLE_TYSAN)
    _add_canary(sanitizer_canary_tysan tysan "TypeSanitizer: type-aliasing-violation")
endif()
if(VC_ENABLE_NSAN)
    _add_canary(sanitizer_canary_nsan  nsan  "NumericalStabilitySanitizer: inconsistent")
endif()

vc_add_test(NAME test_gridstore)
vc_add_test(NAME test_gridstore_branches)

# Live network test: anonymous read of the PHerc 0172 zarr on AWS S3.
# Soft-skips on network failure so offline runs still pass; set
# VC_TEST_REQUIRE_NETWORK=1 to flip those skips into hard failures.
vc_add_test(NAME test_pherc0172_live)

vc_add_test(NAME test_tiff)
vc_add_test(NAME test_normalgridtools)

# Live S3: opens the PHerc 0172 volume and reads small chunks.
vc_add_test(NAME test_volume_live_s3)

vc_add_test(NAME test_normal_grid_volume)
vc_add_test(NAME test_surface_patch_index)
vc_add_test(NAME test_tiff_merge)
vc_add_test(NAME test_http_fetch_errors)

vc_add_test(NAME test_normal_grid_live)

vc_add_test(NAME test_normalgridtools_real)
vc_add_test(NAME test_tiff_more)
vc_add_test(NAME test_geometry)
vc_add_test(NAME test_affine_transform)
vc_add_test(NAME test_point_index)
vc_add_test(NAME test_plane_surface)
vc_add_test(NAME test_compositing)
vc_add_test(NAME test_quadsurface_basics)
vc_add_test(NAME test_load_json)
vc_add_test(NAME test_remote_url)
vc_add_test(NAME test_logging)
vc_add_test(NAME test_plane_surface_branches)
vc_add_test(NAME test_affine_transform_surface)
vc_add_test(NAME test_quadsurface_more)
vc_add_test(NAME test_thinning)
vc_add_test(NAME test_umbilicus)
vc_add_test(NAME test_colormaps)
vc_add_test(NAME test_quadsurface_io)
vc_add_test(NAME test_quadsurface_pointto)
vc_add_test(NAME test_quadsurface_snapshot)
vc_add_test(NAME test_quadsurface_save_paths)
vc_add_test(NAME test_volume_local)
vc_add_test(NAME test_zarr_helpers)
vc_add_test(NAME test_chunked_plane_sampler)
vc_add_test(NAME test_slicing)
vc_add_test(NAME test_volume_pkg)
vc_add_test(NAME test_volpkg_convert)
vc_add_test(NAME test_vcdataset)
vc_add_test(NAME test_zarr)
vc_add_test(NAME test_chunk_cache)
vc_add_test(NAME test_volume_extras)
vc_add_test(NAME test_zarr_more)
vc_add_test(NAME test_vcdataset_compressors)
vc_add_test(NAME test_volume_static_read)
vc_add_test(NAME test_volume_pyramid)
vc_add_test(NAME test_chunk_cache_persist)
vc_add_test(NAME test_zarr_chunk_fetcher)
vc_add_test(NAME test_chunked_plane_sampler_more)
vc_add_test(NAME test_slicing_more)
vc_add_test(NAME test_vcdataset_more)

vc_add_test(NAME test_segmentation)
vc_add_test(NAME test_chunked_tensor_meta LIBS doctest::doctest vc_core)
vc_add_test(NAME test_post_process_util)
vc_add_test(NAME test_render)
vc_add_test(NAME test_growth_config
    LIBS doctest::doctest vc_core
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/..)
vc_add_test(NAME test_growth_helpers
    LIBS doctest::doctest vc_core
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/..)
vc_add_test(NAME test_surf_tracker_data
    LIBS doctest::doctest vc_core
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/..)
vc_add_test(NAME test_sparse_chunk_spool)
vc_add_test(NAME test_neural_tracer_connection LIBS doctest::doctest vc_core)
vc_add_test(NAME test_quadsurface_final)
vc_add_test(NAME test_quadsurface_components)

# Real PHerc 0172 fixtures under data/segments/, reached via VC_TEST_FIXTURES_DIR.
vc_add_test(NAME test_quadsurface_fixtures)
vc_add_test(NAME test_volume_pkg_full)
vc_add_test(NAME test_load_quad_from_tifxyz)
vc_add_test(NAME test_surface_patch_index_more)
vc_add_test(NAME test_quadsurface_extras)
vc_add_test(NAME test_surface_patch_index_cache)
vc_add_test(NAME test_volume_pkg_more)
