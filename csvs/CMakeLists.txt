cmake_minimum_required(VERSION 3.16)
project(csvs C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_EXTENSIONS ON)

# ── Required: LZ4 ────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET liblz4)

# ── Optional codecs ──────────────────────────────────────────────────────────
pkg_check_modules(ZSTD IMPORTED_TARGET libzstd)
pkg_check_modules(OPENH264 IMPORTED_TARGET openh264)
pkg_check_modules(X265 IMPORTED_TARGET x265)
pkg_check_modules(LIBDE265 IMPORTED_TARGET libde265)
pkg_check_modules(DAV1D IMPORTED_TARGET dav1d)

# aom doesn't always have a .pc — try pkg-config then fall back to find_library
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

# H265 needs both x265 (encode) and libde265 (decode)
if(X265_FOUND AND LIBDE265_FOUND)
    set(H265_FOUND TRUE)
else()
    set(H265_FOUND FALSE)
endif()

# AV1 needs both aom (encode) and dav1d (decode)
if(AOM_FOUND AND DAV1D_FOUND)
    set(AV1_FOUND TRUE)
else()
    set(AV1_FOUND FALSE)
endif()

message(STATUS "csvs codecs:")
message(STATUS "  lz4      : YES (required)")
message(STATUS "  zstd     : ${ZSTD_FOUND}")
message(STATUS "  h264     : ${OPENH264_FOUND}")
message(STATUS "  h265     : ${H265_FOUND} (x265=${X265_FOUND}, libde265=${LIBDE265_FOUND})")
message(STATUS "  av1      : ${AV1_FOUND} (aom=${AOM_FOUND}, dav1d=${DAV1D_FOUND})")

# ── Shared library ───────────────────────────────────────────────────────────
add_library(csvs SHARED csvs_shlib.c)
target_include_directories(csvs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(csvs PRIVATE PkgConfig::LZ4)

if(ZSTD_FOUND)
    target_compile_definitions(csvs PRIVATE CSVS_HAS_ZSTD)
    target_link_libraries(csvs PRIVATE PkgConfig::ZSTD)
endif()

if(OPENH264_FOUND)
    target_compile_definitions(csvs PRIVATE CSVS_HAS_H264)
    target_link_libraries(csvs PRIVATE PkgConfig::OPENH264)
endif()

if(H265_FOUND)
    target_compile_definitions(csvs PRIVATE CSVS_HAS_H265)
    target_link_libraries(csvs PRIVATE PkgConfig::X265 PkgConfig::LIBDE265)
endif()

if(AV1_FOUND)
    target_compile_definitions(csvs PRIVATE CSVS_HAS_AV1)
    target_link_libraries(csvs PRIVATE PkgConfig::AOM PkgConfig::DAV1D)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_csvs test_csvs.c)
target_link_libraries(test_csvs PRIVATE PkgConfig::LZ4 m)

# Mirror the codec defines so tests compile the matching paths
if(ZSTD_FOUND)
    target_compile_definitions(test_csvs PRIVATE CSVS_HAS_ZSTD)
    target_link_libraries(test_csvs PRIVATE PkgConfig::ZSTD)
endif()
if(OPENH264_FOUND)
    target_compile_definitions(test_csvs PRIVATE CSVS_HAS_H264)
    target_link_libraries(test_csvs PRIVATE PkgConfig::OPENH264)
endif()
if(H265_FOUND)
    target_compile_definitions(test_csvs PRIVATE CSVS_HAS_H265)
    target_link_libraries(test_csvs PRIVATE PkgConfig::X265 PkgConfig::LIBDE265)
endif()
if(AV1_FOUND)
    target_compile_definitions(test_csvs PRIVATE CSVS_HAS_AV1)
    target_link_libraries(test_csvs PRIVATE PkgConfig::AOM PkgConfig::DAV1D)
endif()

add_test(NAME test_csvs COMMAND test_csvs)
