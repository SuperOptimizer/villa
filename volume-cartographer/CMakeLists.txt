cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(VC_VERSION 3.0.2)

list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/vc-dependencies")

option(VC_VERSION_DATESTAMP "Append date stamp to version number" off)
if(VC_VERSION_DATESTAMP)
string(TIMESTAMP VC_VERSION_TWEAK "%s")
set(VC_VERSION ${VC_VERSION}.${VC_VERSION_TWEAK})
endif()

project(VC3D VERSION ${VC_VERSION})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(VC_USE_OPENMP "Enable OpenMP support" ON)
option(VC_USE_VALGRIND "Enable Valgrind support" off)
option(VC_USE_PCH "Enable precompiled headers (faster incremental builds, but disables ccache)" OFF)
option(VC_USE_UNITY_BUILD "Enable unity builds (experimental, may fail due to ODR conflicts)" OFF)
option(VC_FAST_DEBUG "Use -g1 instead of -g for faster debug builds (less debug info)" OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND NOT VC_USE_PCH)
    # Only use ccache if PCH is disabled (they don't work well together)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache (PCH disabled)")
elseif(VC_USE_PCH)
    message(STATUS "Using precompiled headers (ccache disabled)")
endif()

# Use modern C++ for everything and generate compile_commands database.
set(CMAKE_EXPORT_COMPILE_COMMANDS on)


# Choose what to install
option(VC_INSTALL_APPS      "Install VC core programs"    on)
option(VC_INSTALL_LIBS      "Install VC libraries"        on)
option(VC_INSTALL_UTILS     "Install VC utility programs" on)

# Some helpful constants to be used in subprojects.
include(VCConstants)

# Enable/disable PaStiX from the build (OFF by default)
option(VC_WITH_PASTIX "Build with PaStiX support" OFF)

# Look for external dependencies.
include(VCFindDependencies)


set(CMAKE_CXX_FLAGS "  -std=c++23 -DWITH_BLOSC=1 -DWITH_ZLIB=1 -march=native " )
set(CMAKE_EXE_LINKER_FLAGS " ")
include_directories(core/include)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Set LTO type and linker based on compiler and platform
# VC_LIBCXX_FLAGS: libc++ path (needed for ALL build types on macOS with Homebrew LLVM)
# VC_LTO_FLAGS: LTO flags (only for optimized builds)
# VC_LINKER_FLAGS: other linker flags (lld on Linux)
set(VC_LIBCXX_FLAGS "")
set(VC_LTO_FLAGS "")
set(VC_LINKER_FLAGS "")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang-specific devirtualization flags (work with or without LTO, but best with LTO)
    # -fstrict-vtable-pointers: Assume vtable pointers don't change after construction
    # -fwhole-program-vtables: LTO-time vtable optimization (requires LTO)
    # Note: -fvirtual-function-elimination requires -flto=full (not thin)
    set(VC_DEVIRT_FLAGS "-fstrict-vtable-pointers")
    set(VC_DEVIRT_LTO_FLAGS "-fwhole-program-vtables")

    # Aggressive optimization flags
    # -ffast-math: Break IEEE FP compliance for speed (reciprocal, reassociation, no errno)
    # -fno-finite-math-only: But allow NaN/Inf (used as sentinels in the codebase)
    # -funroll-loops: More aggressive loop unrolling
    # -ffp-contract=fast: Enable FMA fusion across statements
    # -fno-semantic-interposition: Better symbol resolution for shared libs
    # -fvisibility=hidden: Smaller symbol table, better code gen
    # -fvisibility-inlines-hidden: Hide inline function symbols
    set(VC_AGGRESSIVE_MATH "-ffast-math -fno-finite-math-only -funroll-loops -ffp-contract=fast")
    set(VC_VISIBILITY_FLAGS "-fno-semantic-interposition -fvisibility=hidden -fvisibility-inlines-hidden")

    # LLVM backend passes for additional optimizations
    # -mllvm -enable-gvn-hoist: Hoist identical computations
    # -mllvm -enable-loopinterchange: Enable loop interchange for better cache usage
    set(VC_LLVM_PASSES "-mllvm -enable-gvn-hoist -mllvm -enable-loopinterchange")

    if(APPLE)
        # macOS: enable LTO with lld linker (Homebrew LLVM includes lld)
        find_program(LLD_LINKER ld.lld HINTS "/opt/homebrew/opt/llvm/bin")
        if(LLD_LINKER)
            set(VC_LTO_FLAGS "-flto=thin -march=native ${VC_DEVIRT_FLAGS} ${VC_DEVIRT_LTO_FLAGS} ${VC_AGGRESSIVE_MATH} ${VC_VISIBILITY_FLAGS} ${VC_LLVM_PASSES}")
            set(VC_LINKER_FLAGS "-fuse-ld=lld")
            message(STATUS "Using Clang on macOS: thin LTO + devirt + aggressive optimizations + visibility")
        else()
            # Fallback: no LTO if lld not available (still use non-LTO devirt flags + math opts)
            set(VC_LTO_FLAGS "-march=native ${VC_DEVIRT_FLAGS} ${VC_AGGRESSIVE_MATH} ${VC_VISIBILITY_FLAGS}")
            message(STATUS "Using Clang on macOS: LTO disabled, partial devirt + aggressive math")
        endif()
        # Link against Homebrew LLVM's libc++ to match compiler ABI (all build types)
        if(EXISTS "/opt/homebrew/opt/llvm/lib/c++")
            set(VC_LIBCXX_FLAGS "-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++")
            message(STATUS "Using Clang on macOS: Homebrew libc++")
        else()
            message(STATUS "Using Clang on macOS: system libc++")
        endif()
    else()
        # Linux Clang: LTO + full devirtualization + aggressive opts
        set(VC_LTO_FLAGS "-flto=thin -march=native ${VC_DEVIRT_FLAGS} ${VC_DEVIRT_LTO_FLAGS} ${VC_AGGRESSIVE_MATH} ${VC_VISIBILITY_FLAGS} ${VC_LLVM_PASSES}")
        set(VC_LINKER_FLAGS "-fuse-ld=lld")
        message(STATUS "Using Clang: thin LTO + devirt + aggressive optimizations")
    endif()
else()
    set(VC_LTO_FLAGS "-flto=auto")
    message(STATUS "Using ${CMAKE_CXX_COMPILER_ID}: auto LTO with default linker")
endif()

# Apply libc++ flags to all build types (required for Homebrew LLVM on macOS)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LIBCXX_FLAGS}")

# Use pipes instead of temp files for faster compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

# Unity build support (combines TUs for faster full builds)
if(VC_USE_UNITY_BUILD)
    set(CMAKE_UNITY_BUILD ON)
    set(CMAKE_UNITY_BUILD_BATCH_SIZE 8)
    message(STATUS "Unity build enabled (batch size: 8)")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(VC_FAST_DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g1 -O0")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g1")
        message(STATUS "Fast debug mode: using -g1 (minimal debug info)")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "QuickBuild")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${VC_LTO_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LTO_FLAGS} ${VC_LINKER_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "ReleaseUnsafe")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${VC_LTO_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LTO_FLAGS} ${VC_LINKER_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${VC_LTO_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LTO_FLAGS} ${VC_LINKER_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os ${VC_LTO_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LTO_FLAGS} ${VC_LINKER_FLAGS}")
endif()


# Profile-Guided Optimization (PGO) support
# Usage:
#   1. Build with -DVC_PGO_MODE=GENERATE and run representative workloads
#   2. Rebuild with -DVC_PGO_MODE=USE to apply optimizations
set(VC_PGO_MODE "OFF" CACHE STRING "PGO mode: OFF, GENERATE, or USE")
set_property(CACHE VC_PGO_MODE PROPERTY STRINGS "OFF" "GENERATE" "USE")
set(VC_PGO_DIR "${CMAKE_BINARY_DIR}/pgo-profiles" CACHE PATH "Directory for PGO profile data")

if(NOT VC_PGO_MODE STREQUAL "OFF")
    # PGO is incompatible with sanitizers
    if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN OR VC_ENABLE_LSAN)
        message(FATAL_ERROR "PGO cannot be used together with sanitizers")
    endif()

    if(VC_PGO_MODE STREQUAL "GENERATE")
        message(STATUS "PGO: Instrumented build (profile generation mode)")
        message(STATUS "PGO: Profile data will be written to: ${VC_PGO_DIR}")

        # Create profile directory
        file(MAKE_DIRECTORY "${VC_PGO_DIR}")

        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            # Clang uses LLVM's instrumentation-based PGO
            set(VC_PGO_GEN_FLAGS "-fprofile-generate=${VC_PGO_DIR}")
            # For Clang, we can also use IR-based PGO which is more accurate:
            # set(VC_PGO_GEN_FLAGS "-fprofile-instr-generate=${VC_PGO_DIR}/default.profraw")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # GCC uses its own profile format
            set(VC_PGO_GEN_FLAGS "-fprofile-generate=${VC_PGO_DIR}")
        else()
            message(WARNING "PGO: Unsupported compiler ${CMAKE_CXX_COMPILER_ID}, disabling PGO")
            set(VC_PGO_MODE "OFF")
        endif()

        if(NOT VC_PGO_MODE STREQUAL "OFF")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VC_PGO_GEN_FLAGS}")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VC_PGO_GEN_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_PGO_GEN_FLAGS}")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${VC_PGO_GEN_FLAGS}")
        endif()

    elseif(VC_PGO_MODE STREQUAL "USE")
        message(STATUS "PGO: Optimized build (using profile data)")
        message(STATUS "PGO: Reading profiles from: ${VC_PGO_DIR}")

        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            # Check if we need to merge profiles first (for instrumentation-based PGO)
            file(GLOB PROFRAW_FILES "${VC_PGO_DIR}/*.profraw")
            if(PROFRAW_FILES)
                # Find llvm-profdata tool
                find_program(LLVM_PROFDATA llvm-profdata
                    HINTS "/opt/homebrew/opt/llvm/bin" "/usr/lib/llvm-*/bin")
                if(LLVM_PROFDATA)
                    set(MERGED_PROFILE "${VC_PGO_DIR}/merged.profdata")
                    message(STATUS "PGO: Merging .profraw files with ${LLVM_PROFDATA}")
                    execute_process(
                        COMMAND ${LLVM_PROFDATA} merge -output=${MERGED_PROFILE} ${PROFRAW_FILES}
                        RESULT_VARIABLE MERGE_RESULT
                    )
                    if(NOT MERGE_RESULT EQUAL 0)
                        message(WARNING "PGO: Failed to merge profile data, build may fail")
                    endif()
                    set(VC_PGO_USE_FLAGS "-fprofile-use=${MERGED_PROFILE}")
                else()
                    message(WARNING "PGO: llvm-profdata not found, cannot merge .profraw files")
                    set(VC_PGO_USE_FLAGS "-fprofile-use=${VC_PGO_DIR}")
                endif()
            else()
                # Assume frontend-based PGO with .gcda files or already merged data
                set(VC_PGO_USE_FLAGS "-fprofile-use=${VC_PGO_DIR}")
            endif()
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # GCC: -fprofile-correction helps with multi-threaded profile collection
            set(VC_PGO_USE_FLAGS "-fprofile-use=${VC_PGO_DIR} -fprofile-correction")
        else()
            message(WARNING "PGO: Unsupported compiler ${CMAKE_CXX_COMPILER_ID}, disabling PGO")
            set(VC_PGO_MODE "OFF")
        endif()

        if(NOT VC_PGO_MODE STREQUAL "OFF")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VC_PGO_USE_FLAGS}")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VC_PGO_USE_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_PGO_USE_FLAGS}")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${VC_PGO_USE_FLAGS}")
        endif()
    else()
        message(FATAL_ERROR "PGO: Invalid mode '${VC_PGO_MODE}'. Must be OFF, GENERATE, or USE")
    endif()
else()
    message(STATUS "PGO: Disabled")
endif()

# PGO helper targets (always available)
add_custom_target(pgo-clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${VC_PGO_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VC_PGO_DIR}"
    COMMENT "Cleaning PGO profile data in ${VC_PGO_DIR}"
)

add_custom_target(pgo-info
    COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "PGO (Profile-Guided Optimization) Workflow"
    COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Current PGO mode: ${VC_PGO_MODE}"
    COMMAND ${CMAKE_COMMAND} -E echo "Profile directory: ${VC_PGO_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Step 1: Build instrumented version"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake -DVC_PGO_MODE=GENERATE -DCMAKE_BUILD_TYPE=Release .."
    COMMAND ${CMAKE_COMMAND} -E echo "  ninja"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Step 2: Run representative workloads"
    COMMAND ${CMAKE_COMMAND} -E echo "  ./bin/vc_render_tifxyz -i ... -o ..."
    COMMAND ${CMAKE_COMMAND} -E echo "  ./bin/VC3D  # use typical workflows"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Step 3: Build optimized version"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake -DVC_PGO_MODE=USE -DCMAKE_BUILD_TYPE=Release .."
    COMMAND ${CMAKE_COMMAND} -E echo "  ninja"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Helper targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  ninja pgo-clean   - Remove all profile data"
    COMMAND ${CMAKE_COMMAND} -E echo "  ninja pgo-info    - Show this help"
    COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
    VERBATIM
)

option(VC_ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
option(VC_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(VC_ENABLE_TSAN "Enable ThreadSanitizer for data race detection" OFF)
option(VC_ENABLE_LSAN "Enable LeakSanitizer for memory leak detection (standalone)" OFF)

if(VC_ENABLE_ASAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be used together")
endif()

if(VC_ENABLE_LSAN AND VC_ENABLE_ASAN)
    message(WARNING "LeakSanitizer is already included with AddressSanitizer. Disabling standalone LeakSanitizer.")
    set(VC_ENABLE_LSAN OFF)
endif()

if(VC_ENABLE_LSAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "LeakSanitizer and ThreadSanitizer cannot be used together")
endif()

if(VC_ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer  -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DADDRESS_SANITIZER")
else()
    message(STATUS "Addressanitizer disabled")
endif()

if(VC_ENABLE_UBSAN)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
    set(UBSAN_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all")
    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=float-divide-by-zero")
    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=float-cast-overflow")
    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=nullability")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UBSAN_FLAGS} -fno-omit-frame-pointer  -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${UBSAN_FLAGS} -fno-omit-frame-pointer  -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${UBSAN_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${UBSAN_FLAGS}")
else()
    message(STATUS "UndefinedBehaviorSanitizer disabled")
endif()

if(VC_ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer  -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread -fno-omit-frame-pointer  -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message(STATUS "ThreadSanitizer disabled")
endif()

if(VC_ENABLE_LSAN)
    message(STATUS "LeakSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer  -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak -fno-omit-frame-pointer  -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=leak")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=leak")
else()
    message(STATUS "LeakSanitizer disabled ")
endif()

if(VC_USE_VALGRIND)
    message(STATUS "Valgrind support enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-avx512f -mno-fma -mno-bmi -mno-bmi2 -Og -g -fno-omit-frame-pointer " )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-avx512f -mno-avx512cd -mno-avx512bw -mno-avx512dq -mno-avx512vl -mno-fma")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-avx512f -mno-avx512cd -mno-avx512bw -mno-avx512dq -mno-avx512vl -mno-fma")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-lto ")
else()
    message(STATUS "Valgrind support disabled")
endif()

option(VC_DEVELOPER_WARNINGS "For developers: Enable extensive compiler warnings" OFF)
mark_as_advanced(VC_DEVELOPER_WARNINGS)
if(VC_DEVELOPER_WARNINGS)
    include(VCWarnings)
endif()

# Vectorization diagnostics
option(VC_VECTORIZATION_REPORT "Enable compiler vectorization diagnostics" OFF)
if(VC_VECTORIZATION_REPORT)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-optimized -fopt-info-vec-missed")
    endif()
    message(STATUS "Vectorization report enabled")
endif()

add_subdirectory(misc)
add_subdirectory(libs/OpenABF)
add_subdirectory(core)
add_subdirectory(apps)

include(VCPackageConfig)
include(VCInstall)
include(Packing)

