

configure_file(Version.cpp.in Version.cpp)

add_library(vc_core
        Segmentation.cpp
        Volume.cpp
        VolumePkg.cpp
        SurfaceArea.cpp
        ChunkedTensor.cpp
        SurfacePatchIndex.cpp
        Logging.cpp
        ChunkCache.cpp
        GeometryUtils.cpp
        Slicing.cpp
        Surface.cpp
        Tiff.cpp
        PlaneSurface.cpp
        QuadSurface.cpp
        SurfaceMeta.cpp
        Render.cpp
        Thinning.cpp
        Umbilicus.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp
)

target_link_libraries(vc_core
        PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
        opencv_calib3d
        opencv_video
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        z5
        OpenMP::OpenMP_CXX
        tiff
        ${CURL_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        #${ZLIB_LIBRARIES}
)