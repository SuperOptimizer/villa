add_library(utils STATIC src/zarr.cpp src/Json.cpp)
target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(utils PUBLIC cxx_std_23)
# http_fetch.hpp is now a header-only shim over the vendored libs3 C
# library (S3 transport + AWS credential resolution). Consumers that
# include the header transitively need libs3, so link it PUBLIC. libcurl
# stays for the rest of utils.
target_link_libraries(utils PUBLIC
    nlohmann_json::nlohmann_json
    CURL::libcurl
    libs3
    zstd::libzstd
    LZ4::lz4
    ZLIB::ZLIB
)
target_compile_definitions(utils PUBLIC UTILS_HAS_CURL=1)
set_target_properties(utils PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)

add_library(utils_c3d_codec STATIC src/c3d_codec.cpp)
target_include_directories(utils_c3d_codec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(utils_c3d_codec PUBLIC cxx_std_23)
target_link_libraries(utils_c3d_codec PUBLIC c3d)
target_compile_definitions(utils_c3d_codec PUBLIC UTILS_HAS_C3D_CODEC=1)
set_target_properties(utils_c3d_codec PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)
