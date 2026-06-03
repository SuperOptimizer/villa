# vc — volume-compressor: lossy 3D u8 codec with a sparse, appendable archive
# (single TU, C23, libc + libm only). Upstream:
# github.com/SuperOptimizer/volume-compressor (vendored copy of src/vc/{vc.c,vc.h}).
add_library(vc vc.c)

target_include_directories(vc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_features(vc PRIVATE c_std_23)

# Lossy decode-only codec: -march/-ffast-math/LTO intended (5x decode vs -O2).
target_compile_options(vc PRIVATE
    -O3
    -march=native
    -ffp-contract=fast
    -ffast-math
    -funroll-loops
    -w  # upstream vendored code — silence its warnings
)
set_target_properties(vc PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

target_link_libraries(vc PRIVATE m)

# Coarse multithreading: libvc's decode has `#pragma omp parallel for` over the
# independent atom grid (no-ops without OpenMP). The top-level project already
# runs find_package(OpenMP); just link the C target if it exists (re-running the
# find here trips set_property on the alias target).
if(TARGET OpenMP::OpenMP_C)
    target_link_libraries(vc PRIVATE OpenMP::OpenMP_C)
endif()
