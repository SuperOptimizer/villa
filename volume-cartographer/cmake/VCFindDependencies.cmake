# --- VC dependencies ----------------------------------------------------------
include(FetchContent)

option(VC_BUILD_JSON "Build in-source JSON library" OFF)

#find_package(CURL REQUIRED)
#find_package(OpenSSL REQUIRED)
#find_package(ZLIB REQUIRED)
#find_package(glog REQUIRED)

# ---- Blosc2 (direct dependency for custom zarr implementation) ---------------
find_package(Blosc2 REQUIRED CONFIG)
message(STATUS "Blosc2 found: Blosc2::blosc2")

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
find_package(Eigen3 3.3 REQUIRED)
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
    find_package(OpenMP REQUIRED)
else()
    message(STATUS "OpenMP support disabled")
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C  ALIAS openmp_stub)
    # Add openmp_stub to the export set so install(EXPORT) works
    install(TARGETS openmp_stub EXPORT "${targets_export_name}")
endif()

# ---- nlohmann/json -----------------------------------------------------------
if (VC_BUILD_JSON)
    FetchContent_Declare(
        json
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        URL https://github.com/nlohmann/json/archive/v3.11.3.tar.gz
    )
    FetchContent_GetProperties(json)
    if (NOT json_POPULATED)
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        set(JSON_Install   ON  CACHE INTERNAL "")
        FetchContent_Populate(json)
        add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
else()
    find_package(nlohmann_json 3.9.1 REQUIRED)
endif()

# ---- Boost (apps/utils only) -------------------------------------------------
find_package(Boost 1.58 REQUIRED COMPONENTS program_options)

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
