# libs3 — minimal C23 S3 client (single TU, libcurl-only).
# Upstream: github.com/SuperOptimizer/libs3 (vendored copy).
add_library(libs3 STATIC libs3.c)

target_include_directories(libs3 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_features(libs3 PRIVATE c_std_23)

find_package(Threads REQUIRED)
target_link_libraries(libs3 PUBLIC CURL::libcurl Threads::Threads)

# Upstream vendored code — we don't own the style, silence the noise
# (matches the libs/c3d policy).
target_compile_options(libs3 PRIVATE -w)

# -fPIC so vc_core (potentially built as a shared lib downstream) can link it.
set_target_properties(libs3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
