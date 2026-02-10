# VCCompilerFlags.cmake - Compiler flag orchestration
#
# User-supplied CMAKE_CXX_FLAGS / CMAKE_EXE_LINKER_FLAGS are preserved.
# Our project flags are appended, so `-DCMAKE_CXX_FLAGS="-fno-inline"`
# on the command line will survive.

# Base project flags (appended, never overwrite)
set(VC_BASE_CXX_FLAGS "-std=c++23 -DWITH_BLOSC=1 -DWITH_ZLIB=1 -march=native -pipe")

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# ---- Compiler-specific flags -------------------------------------------------
set(VC_LIBCXX_FLAGS "")
set(VC_LTO_FLAGS "")
set(VC_LINKER_FLAGS "")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    include(LLVMToolchain)
else()
    include(GCCToolchain)
endif()

# ---- Build type flags --------------------------------------------------------
set(VC_BUILD_CXX_FLAGS "")
set(VC_BUILD_LINKER_FLAGS "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(VC_BUILD_CXX_FLAGS "-g3 -O0")
    if(NOT APPLE)
        set(VC_BUILD_LINKER_FLAGS "-g -Wl,--compress-debug-sections=zlib")
    else()
        set(VC_BUILD_LINKER_FLAGS "-g")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "QuickBuild")
    set(VC_BUILD_CXX_FLAGS "-O1")
    set(VC_BUILD_LINKER_FLAGS "${VC_LINKER_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(VC_BUILD_CXX_FLAGS "-O3 -g1 ${VC_LTO_FLAGS}")
    set(VC_BUILD_LINKER_FLAGS "${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "ReleaseUnsafe")
    set(VC_BUILD_CXX_FLAGS "-O3 -g1 ${VC_LTO_FLAGS} ${VC_UNSAFE_CXX_FLAGS}")
    set(VC_BUILD_LINKER_FLAGS "${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_UNSAFE_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(APPLE)
        set(VC_BUILD_CXX_FLAGS "-O3 ${VC_LTO_FLAGS} -g")
        set(VC_BUILD_LINKER_FLAGS "${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS}")
    else()
        set(VC_BUILD_CXX_FLAGS "-O3 ${VC_LTO_FLAGS} -g -gsplit-dwarf")
        set(VC_BUILD_LINKER_FLAGS "${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS} -Wl,--compress-debug-sections=zlib")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(VC_BUILD_CXX_FLAGS "-Oz ${VC_LTO_FLAGS} ${VC_SIZE_FLAGS}")
        set(VC_BUILD_LINKER_FLAGS "${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS} ${VC_SIZE_LINKER_FLAGS} ${VC_STRIP_FLAGS}")
    else()
        set(VC_BUILD_CXX_FLAGS "-Os ${VC_LTO_FLAGS}")
        set(VC_BUILD_LINKER_FLAGS "${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_STRIP_FLAGS}")
    endif()
endif()

# Compose final flags: project base + build-type + user-supplied (last wins)
set(CMAKE_CXX_FLAGS "${VC_BASE_CXX_FLAGS} ${VC_BUILD_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${VC_BUILD_LINKER_FLAGS} ${VC_LIBCXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
