
if(VC_ENABLE_TSAN OR VC_ENABLE_LSAN OR VC_ENABLE_UBSAN OR VC_ENABLE_ASAN)
    set(SANITIZER_SRCS sanitizer_config.c)
else()
    set(SANITIZER_SRCS)
endif()

if(NOT VC_USE_OPENMP)
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
endif()

