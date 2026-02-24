configure_file(src/Version.cpp.in Version.cpp)

add_library(vc_core
        src/GridStore.cpp
        src/NormalGridVolume.cpp
        src/normalgridtools.cpp
        src/Segmentation.cpp
        src/Volume.cpp
        src/VolumePkg.cpp
        src/SurfacePatchIndex.cpp
        src/PointIndex.cpp
        src/Logging.cpp
        src/Slicing.cpp
        src/SlicingTiered.cpp
        src/Compositing.cpp
        src/PlaneSurface.cpp
        src/QuadSurface.cpp
        src/Render.cpp
        src/Tiff.cpp
        src/Zarr.cpp
        src/ChunkCache.cpp
        src/NetworkFilesystem.cpp
        src/cache/ChunkSource.cpp
        src/cache/DiskStore.cpp
        src/cache/IOPool.cpp
        src/cache/TieredChunkCache.cpp
        src/cache/VcDecompressor.cpp
        src/cache/SimpleCacheFactory.cpp
        src/VcDataset.cpp
        src/cache/HttpMetadataFetcher.cpp
        src/RemoteUrl.cpp
        src/RemoteScroll.cpp
        src/CoordGrid.cpp
        src/PostProcess.cpp
        src/Geometry.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp
)

target_link_libraries(vc_core
        PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_calib3d
        opencv_video
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        TIFF::TIFF
        CURL::libcurl
        vc_utils_zarr
        vc_utils_tensor
        vc_utils_json
)

target_compile_definitions(vc_core PUBLIC VC_USE_CURL)

# Precompiled headers for faster builds
if(VC_USE_PCH)
target_precompile_headers(vc_core PRIVATE
    # STL - containers
    <array>
    <deque>
    <list>
    <map>
    <queue>
    <set>
    <stack>
    <unordered_map>
    <unordered_set>
    <vector>
    # STL - algorithms and utilities
    <algorithm>
    <chrono>
    <cmath>
    <cstdint>
    <cstdlib>
    <cstring>
    <functional>
    <iterator>
    <limits>
    <numeric>
    <tuple>
    <type_traits>
    <utility>
    # STL - I/O and strings
    <filesystem>
    <fstream>
    <iomanip>
    <iostream>
    <sstream>
    <string>
    <string_view>
    # STL - memory and smart pointers
    <memory>
    <new>
    # STL - concurrency
    <atomic>
    <condition_variable>
    <mutex>
    <shared_mutex>
    <thread>
    # STL - other
    <cassert>
    <optional>
    <stdexcept>
    <variant>
    # OpenCV
    <opencv2/core.hpp>
    <opencv2/core/mat.hpp>
    <opencv2/imgproc.hpp>
    <opencv2/imgcodecs.hpp>
    # Boost (used by CLI apps)
    <boost/program_options.hpp>
)
endif()

qt6_wrap_cpp(metrics_moc include/vc/ui/VCCollection.hpp)
add_library(vc_ui src/VCCollection.cpp src/UDataManipulateUtils.cpp src/surface_metrics.cpp ${metrics_moc})
target_link_libraries(vc_ui PUBLIC vc_core Qt6::Core Qt6::Gui Qt6::Widgets)
if(VC_USE_PCH)
    target_precompile_headers(vc_ui REUSE_FROM vc_core)
endif()

add_library(vc_flattening src/ABFFlattening.cpp)
target_link_libraries(vc_flattening PUBLIC vc_core OpenABF)

add_library(vc_tracer src/GrowPatch.cpp src/GrowSurface.cpp src/NeuralTracerConnection.cpp src/ChunkedTensor.cpp)
target_link_libraries(vc_tracer PUBLIC vc_core vc_ui Ceres::ceres Qt6::Core)
target_include_directories(vc_tracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../libs/edt)
if(VC_USE_PCH)
    target_precompile_headers(vc_tracer REUSE_FROM vc_core)
endif()

if(VC_INSTALL_LIBS)
    install(
            TARGETS vc_core
            COMPONENT "Libraries"
            EXPORT "${targets_export_name}"
            ARCHIVE DESTINATION "lib"
            LIBRARY DESTINATION "lib"
            INCLUDES DESTINATION "${include_install_dir}"
            RUNTIME DESTINATION "bin"
    )

    install(
            TARGETS vc_flattening
            COMPONENT "Libraries"
            EXPORT "${targets_export_name}"
            ARCHIVE DESTINATION "lib"
            LIBRARY DESTINATION "lib"
            INCLUDES DESTINATION "${include_install_dir}"
            RUNTIME DESTINATION "bin"
    )

    install(
            TARGETS vc_ui
            COMPONENT "Libraries"
            EXPORT "${targets_export_name}"
            ARCHIVE DESTINATION "lib"
            LIBRARY DESTINATION "lib"
            INCLUDES DESTINATION "${include_install_dir}"
            RUNTIME DESTINATION "bin"
    )

    install(
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
            DESTINATION "${include_install_dir}"
            COMPONENT "Libraries"
            FILES_MATCHING REGEX ".*\\.(h|hpp)$"
    )
endif()
