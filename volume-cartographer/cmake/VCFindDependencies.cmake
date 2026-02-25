# --- VC dependencies ----------------------------------------------------------
include(FetchContent)


# ---- Blosc (required for zarr chunk compression) ----------------------------
find_path(BLOSC_INCLUDE_DIR blosc.h)
find_library(BLOSC_LIBRARY NAMES blosc)
if(NOT BLOSC_INCLUDE_DIR OR NOT BLOSC_LIBRARY)
    message(FATAL_ERROR "blosc not found (need blosc.h and libblosc)")
endif()
add_library(Blosc::blosc UNKNOWN IMPORTED)
set_target_properties(Blosc::blosc PROPERTIES
    IMPORTED_LOCATION "${BLOSC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIR}")
message(STATUS "Blosc: ${BLOSC_LIBRARY}")

# ---- zlib (gzip/zlib compression) -------------------------------------------
find_package(ZLIB REQUIRED)

# ---- zstd -------------------------------------------------------------------
find_package(zstd REQUIRED)

# ---- Qt (apps / utils) -------------------------------------------------------
find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Guard old qt cmake helper on distros with Qt < 6.3
if(NOT DEFINED qt_generate_deploy_app_script)
    message(WARNING "WARNING qt_generate_deploy_app_script MISSING!")
    function(qt_generate_deploy_app_script)
    endfunction()
endif()

# ---- CUDA sparse toggle ------------------------------------------------------
option(VC_WITH_CUDA_SPARSE "use cudss" ON)
if (VC_WITH_CUDA_SPARSE)
    add_definitions(-DVC_USE_CUDA_SPARSE=1)
endif()

# ---- Ceres -------------------------------------------------------------------
find_package(Ceres REQUIRED)

# ---- Eigen -------------------------------------------------------------------
# Accept Eigen 3.3+ or 5.x (Eigen jumped from 3.4 to 5.0)
find_package(Eigen3 REQUIRED)
if (CMAKE_GENERATOR MATCHES "Ninja|.*Makefiles.*" AND "${CMAKE_BUILD_TYPE}" MATCHES "^$|Debug")
    message(AUTHOR_WARNING
        "Configuring a Debug build. Eigen performance will be degraded. "
        "Consider RelWithDebInfo for symbols, or Release for max performance.")
endif()

# ---- OpenCV ------------------------------------------------------------------
find_package(OpenCV 3 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 4 QUIET REQUIRED)
endif()

# ---- OpenMP ------------------------------------------------------------------
if (VC_USE_OPENMP)
    message(STATUS "OpenMP support enabled")
    if(APPLE)
        # On macOS, use standalone libomp package to match OpenBLAS/Ceres
        # (avoids duplicate libomp runtime error with LLVM's libomp)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
            message(STATUS "Using standalone libomp from: ${LIBOMP_PREFIX}")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            find_package(OpenMP REQUIRED)
        else()
            message(STATUS "Standalone libomp not found, using default OpenMP")
            find_package(OpenMP REQUIRED)
        endif()
    else()
        find_package(OpenMP REQUIRED)
    endif()

else()
    message(STATUS "OpenMP support disabled")

    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C  ALIAS openmp_stub)
    # Add openmp_stub to the export set so install(EXPORT) works
    install(TARGETS openmp_stub EXPORT "${targets_export_name}")
endif()

# ---- LZ4 (zarr lz4 codec) ---------------------------------------------------
find_path(LZ4_INCLUDE_DIR lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)
if(NOT LZ4_INCLUDE_DIR OR NOT LZ4_LIBRARY)
    message(FATAL_ERROR "lz4 not found (need lz4.h and liblz4)")
endif()
add_library(LZ4::lz4 UNKNOWN IMPORTED)
set_target_properties(LZ4::lz4 PROPERTIES
    IMPORTED_LOCATION "${LZ4_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}")
message(STATUS "LZ4: ${LZ4_LIBRARY}")

# ---- BZip2 (zarr bz2 codec) -------------------------------------------------
find_package(BZip2 REQUIRED)

# ---- LZMA (zarr lzma codec) -------------------------------------------------
find_path(LZMA_INCLUDE_DIR lzma.h)
find_library(LZMA_LIBRARY NAMES lzma)
if(NOT LZMA_INCLUDE_DIR OR NOT LZMA_LIBRARY)
    message(FATAL_ERROR "lzma not found (need lzma.h and liblzma)")
endif()
add_library(LZMA::lzma UNKNOWN IMPORTED)
set_target_properties(LZMA::lzma PROPERTIES
    IMPORTED_LOCATION "${LZMA_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LZMA_INCLUDE_DIR}")
message(STATUS "LZMA: ${LZMA_LIBRARY}")

# ---- FFmpeg (optional, for zarr video codecs) --------------------------------
option(VC_WITH_VIDEO_CODECS "Enable H264/H265/AV1 zarr video codecs (requires FFmpeg)" OFF)
if(VC_WITH_VIDEO_CODECS)
    find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
    find_library(AVCODEC_LIBRARY NAMES avcodec)
    find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
    find_library(AVUTIL_LIBRARY NAMES avutil)
    find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
    find_library(SWSCALE_LIBRARY NAMES swscale)
    if(NOT AVCODEC_LIBRARY OR NOT AVUTIL_LIBRARY OR NOT SWSCALE_LIBRARY)
        message(FATAL_ERROR "FFmpeg not found but VC_WITH_VIDEO_CODECS=ON. Need libavcodec, libavutil, libswscale.")
    endif()
    add_library(FFmpeg::avcodec UNKNOWN IMPORTED)
    set_target_properties(FFmpeg::avcodec PROPERTIES
        IMPORTED_LOCATION "${AVCODEC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${AVCODEC_INCLUDE_DIR}")
    add_library(FFmpeg::avutil UNKNOWN IMPORTED)
    set_target_properties(FFmpeg::avutil PROPERTIES
        IMPORTED_LOCATION "${AVUTIL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${AVUTIL_INCLUDE_DIR}")
    add_library(FFmpeg::swscale UNKNOWN IMPORTED)
    set_target_properties(FFmpeg::swscale PROPERTIES
        IMPORTED_LOCATION "${SWSCALE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SWSCALE_INCLUDE_DIR}")
    message(STATUS "FFmpeg: avcodec=${AVCODEC_LIBRARY} avutil=${AVUTIL_LIBRARY} swscale=${SWSCALE_LIBRARY}")
endif()

# ---- nlohmann/json -----------------------------------------------------------
FetchContent_Declare(
    json
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://github.com/nlohmann/json/archive/v3.12.0.tar.gz
)
FetchContent_GetProperties(json)
if (NOT json_POPULATED)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install   ON  CACHE INTERNAL "")
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ---- TIFF --------------------------------------------------------------------
find_package(TIFF REQUIRED)

# ---- Boost (apps/utils only) -------------------------------------------------
find_package(Boost REQUIRED COMPONENTS program_options)

# ---- PaStiX ------------------------------------------------------------------
if (VC_WITH_PASTIX)
  find_package(PaStiX REQUIRED)
  message(STATUS "PaStiX found: ${PASTIX_LIBRARY}")
  if (NOT TARGET vc3d_pastix)
    add_library(vc3d_pastix INTERFACE)
    target_link_libraries(vc3d_pastix INTERFACE PaStiX::PaStiX)
    target_compile_definitions(vc3d_pastix INTERFACE VC_HAVE_PASTIX=1)
  endif()
endif()
