# --- VC dependencies ----------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# --- Blosc2 -------------------------------------------------------------------
find_package(Blosc2 REQUIRED)

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

# ---- GKlib (required by METIS, which is a transitive dependency of Ceres) ----
# METIS's CMake config doesn't properly export its GKlib dependency, so we need
# to find and link it explicitly
find_package(GKlib QUIET CONFIG HINTS ${CMAKE_PREFIX_PATH})
if (TARGET GKlib::GKlib)
    # Add GKlib to Ceres's interface libraries so any target linking Ceres gets GKlib
    if (TARGET Ceres::ceres)
        target_link_libraries(Ceres::ceres INTERFACE GKlib::GKlib)
    endif()
endif()

# ---- Eigen -------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED)
if (CMAKE_GENERATOR MATCHES "Ninja|.*Makefiles.*" AND "${CMAKE_BUILD_TYPE}" MATCHES "^$|Debug")
    message(AUTHOR_WARNING
        "Configuring a Debug build. Eigen performance will be degraded. "
        "Consider RelWithDebInfo for symbols, or Release for max performance.")
endif()

# ---- OpenCV ------------------------------------------------------------------
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 3 QUIET REQUIRED)
endif()

# ---- OpenMP ------------------------------------------------------------------
if (VC_USE_OPENMP)
    message(STATUS "OpenMP support enabled")
    find_package(OpenMP REQUIRED)
    set(XTENSOR_USE_OPENMP 1)
else()
    message(STATUS "OpenMP support disabled")
    set(XTENSOR_USE_OPENMP 0)
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C  ALIAS openmp_stub)
    # Add openmp_stub to the export set so install(EXPORT) works
    install(TARGETS openmp_stub EXPORT "${targets_export_name}")
endif()

# ---- xtensor/xsimd -----------------------------------------------------------
set(XTENSOR_USE_XSIMD 1)
find_package(xtl REQUIRED)
find_package(xsimd REQUIRED)
find_package(xtensor REQUIRED)
if (xtensor_INCLUDE_DIRS)
    include_directories(SYSTEM ${xtensor_INCLUDE_DIRS})
endif()

# ---- nlohmann/json -----------------------------------------------------------
find_package(nlohmann_json 3.9.1 REQUIRED)

# --- z5 -----------------------------------------------------------------------
find_package(z5 CONFIG REQUIRED)

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
