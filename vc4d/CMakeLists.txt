cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

project(VC4D VERSION 4.0.0 LANGUAGES CXX)

# ---- C++26 everywhere --------------------------------------------------------
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options -----------------------------------------------------------------
option(VC4D_BUILD_TESTS "Build unit tests" ON)
option(VC4D_BUILD_APP   "Build the Qt GUI application" ON)

# ---- Dependencies (minimal) -------------------------------------------------
# Qt 6 — the only required external library for the GUI.
# Core/Gui/Widgets/Network/Concurrent — same set as vc3d but we lean on Qt
# for JSON (QJsonDocument), HTTP (QNetworkAccessManager), images (QImage),
# threading (QtConcurrent), and math (QVector3D) wherever possible.
if(VC4D_BUILD_APP)
    find_package(Qt6 6.7 REQUIRED COMPONENTS Core Gui Widgets Network Concurrent OpenGLWidgets)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
endif()

# nlohmann/json — header-only, used for zarr metadata and volpkg config.
# Qt's QJsonDocument works for simple cases but zarr .zarray files use features
# (integer keys, nested arrays) that are easier with nlohmann.
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/archive/v3.12.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

# ---- Core library (header-heavy, no Qt dependency) ---------------------------
add_library(vc4d_core STATIC
    src/core/volume.cpp
    src/core/quad_surface.cpp
    src/core/zarr.cpp
    src/core/io.cpp
)
target_include_directories(vc4d_core PUBLIC include)
target_link_libraries(vc4d_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(vc4d_core PUBLIC cxx_std_26)

# ---- Cache library (uses Qt for networking) ----------------------------------
if(VC4D_BUILD_APP)
    add_library(vc4d_cache STATIC
        src/cache/tiered_cache.cpp
        src/cache/disk_store.cpp
    )
    target_include_directories(vc4d_cache PUBLIC include)
    target_link_libraries(vc4d_cache PUBLIC vc4d_core Qt6::Core Qt6::Network)
endif()

# ---- GUI application ---------------------------------------------------------
if(VC4D_BUILD_APP)
    add_executable(vc4d
        src/ui/main.cpp
        src/ui/main_window.cpp
        src/ui/app_state.cpp
        src/ui/volume_viewer.cpp
    )
    target_link_libraries(vc4d PRIVATE
        vc4d_core
        vc4d_cache
        Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Concurrent Qt6::OpenGLWidgets
    )
    target_include_directories(vc4d PRIVATE include)
endif()

# ---- Tests -------------------------------------------------------------------
if(VC4D_BUILD_TESTS)
    enable_testing()

    # Use Qt Test as the test framework — no googletest dependency needed
    find_package(Qt6 REQUIRED COMPONENTS Test)

    add_executable(vc4d_test_math   tests/test_math.cpp)
    target_link_libraries(vc4d_test_math PRIVATE vc4d_core Qt6::Test)
    add_test(NAME math COMMAND vc4d_test_math)

    add_executable(vc4d_test_grid   tests/test_grid.cpp)
    target_link_libraries(vc4d_test_grid PRIVATE vc4d_core Qt6::Test)
    add_test(NAME grid COMMAND vc4d_test_grid)
endif()
