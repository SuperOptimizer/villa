# c3d — 3D volumetric u8 compression codec (single TU, C23, libc only).
# SINGLE VENDOR: the c3d source lives in matter-compressor (libs/matter-compressor/
# src/c3d.{c,h}); this target builds from there so there is exactly one copy in the
# tree. utils_c3d_codec links this for the zarr codec (encode/decode) path.
set(MC_C3D_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../matter-compressor/src)
add_library(c3d ${MC_C3D_DIR}/c3d.c)

target_include_directories(c3d PUBLIC
    $<BUILD_INTERFACE:${MC_C3D_DIR}>
)

target_compile_features(c3d PRIVATE c_std_23)

# The repo-wide policy (AGENTS.md §3.1) is strict IEEE math — no
# -ffast-math / -Ofast. Keep it here too: c3d is a compression codec,
# and relaxing FP semantics across compilers/architectures would make
# encoded bytes non-reproducible between platforms. The upstream
# throughput targets were measured with -ffast-math but the reference
# decoder is fine without it.
target_compile_options(c3d PRIVATE
    -fno-math-errno
    -fno-trapping-math
    # c3d runs SINGLE-THREADED. Callers (tracer batch encode, mc_volume's
    # 24-thread prefetch) drive parallelism themselves; letting each c3d call
    # spawn an OpenMP team of nproc multiplies threads (24*16=384 in prefetch).
    # -fno-openmp overrides the global -fopenmp; c3d's omp_ calls are all
    # #ifdef _OPENMP-guarded with serial fallbacks, so the pragmas become no-ops.
    -fno-openmp
    # Upstream vendored code — we don't own the style, silence the noise.
    -w
)

target_link_libraries(c3d PRIVATE m)
