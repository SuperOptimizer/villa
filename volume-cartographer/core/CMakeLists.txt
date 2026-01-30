configure_file(src/Version.cpp.in Version.cpp)

add_library(vc_core
        src/GridStore.cpp
        src/NormalGridVolume.cpp
        src/normalgridtools.cpp
        src/Segmentation.cpp
        src/Volume.cpp
        src/VolumePkg.cpp
        src/ChunkedTensor.cpp
        src/SurfacePatchIndex.cpp
        src/PointIndex.cpp
        src/Logging.cpp
        src/ChunkCache.cpp
        src/Slicing.cpp
        src/Compositing.cpp
        src/PlaneSurface.cpp
        src/QuadSurface.cpp
        src/Render.cpp
        src/Tiff.cpp
        src/Geometry.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
        src/ABFFlattening.cpp
        src/zarr/Tensor3D.cpp
        src/zarr/BloscCodec.cpp
        src/zarr/ZarrDataset.cpp
        src/zarr/ShardedStore.cpp
        src/csvs/csvs_impl.c
        src/csvs/CsvsDataset.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp
)

# ── Optional csvs codecs ─────────────────────────────────────────────────
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(OPENH264 IMPORTED_TARGET openh264)
    pkg_check_modules(X265 IMPORTED_TARGET x265)
    pkg_check_modules(LIBDE265 IMPORTED_TARGET libde265)
    pkg_check_modules(DAV1D IMPORTED_TARGET dav1d)
    pkg_check_modules(AOM IMPORTED_TARGET aom)
    if(NOT AOM_FOUND)
        find_library(AOM_LIBRARY aom)
        find_path(AOM_INCLUDE_DIR aom/aom_encoder.h)
        if(AOM_LIBRARY AND AOM_INCLUDE_DIR)
            set(AOM_FOUND TRUE)
            add_library(PkgConfig::AOM INTERFACE IMPORTED)
            set_target_properties(PkgConfig::AOM PROPERTIES
                INTERFACE_LINK_LIBRARIES "${AOM_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${AOM_INCLUDE_DIR}")
        endif()
    endif()
endif()

if(OPENH264_FOUND)
    target_compile_definitions(vc_core PRIVATE CSVS_HAS_H264)
    message(STATUS "csvs h264: YES")
else()
    message(STATUS "csvs h264: NO")
endif()

if(X265_FOUND AND LIBDE265_FOUND)
    target_compile_definitions(vc_core PRIVATE CSVS_HAS_H265)
    message(STATUS "csvs h265: YES")
else()
    message(STATUS "csvs h265: NO (x265=${X265_FOUND}, libde265=${LIBDE265_FOUND})")
endif()

if(AOM_FOUND AND DAV1D_FOUND)
    target_compile_definitions(vc_core PRIVATE CSVS_HAS_AV1)
    message(STATUS "csvs av1: YES")
else()
    message(STATUS "csvs av1: NO (aom=${AOM_FOUND}, dav1d=${DAV1D_FOUND})")
endif()

target_link_libraries(vc_core
        PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_calib3d
        opencv_video
        nlohmann_json::nlohmann_json
        Blosc2::blosc2
        zstd
        lz4
        OpenMP::OpenMP_CXX
        tiff
        OpenABF
        #${CURL_LIBRARIES}
        #${OPENSSL_LIBRARIES}
        #${ZLIB_LIBRARIES}
)

if(OPENH264_FOUND)
    target_link_libraries(vc_core PRIVATE PkgConfig::OPENH264)
endif()
if(X265_FOUND AND LIBDE265_FOUND)
    target_link_libraries(vc_core PRIVATE PkgConfig::X265 PkgConfig::LIBDE265)
endif()
if(AOM_FOUND AND DAV1D_FOUND)
    target_link_libraries(vc_core PRIVATE PkgConfig::AOM PkgConfig::DAV1D)
endif()

# csvs_impl.c needs C23 and GNU extensions for pread/pwrite/thread_local
set_source_files_properties(src/csvs/csvs_impl.c PROPERTIES
    LANGUAGE C
    COMPILE_OPTIONS "-std=gnu23"
)

qt6_wrap_cpp(metrics_moc include/vc/ui/VCCollection.hpp)
add_library(vc_ui src/VCCollection.cpp src/UDataManipulateUtils.cpp src/surface_metrics.cpp ${metrics_moc})
target_link_libraries(vc_ui PUBLIC vc_core Qt6::Core Qt6::Gui Qt6::Widgets)

add_library(vc_tracer src/GrowPatch.cpp src/GrowSurface.cpp src/NeuralTracerConnection.cpp)
target_link_libraries(vc_tracer PUBLIC vc_core vc_ui Ceres::ceres Qt6::Core)


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
