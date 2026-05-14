#!/bin/bash
# fromscratch.sh — build VC3D's third-party deps from source.
#
# What this builds, and why (in dep order; later libs FIND earlier ones in $PREFIX):
#
#   compression:        zlib, zstd, lz4, libdeflate, c-blosc (v1)
#   image:              libjpeg-turbo, libpng, libtiff
#   geometry/math:      gmp, mpfr, cgal (header-only modulo gmp/mpfr), eigen3
#   numerical:          openblas (with lapack), metis, suitesparse (cholmod/amd/...),
#                       gflags, glog, ceres
#   json:               nlohmann_json
#   pkging glue:        boost (program_options only)
#   imaging:            opencv  (core/imgproc/imgcodecs/calib3d/flann/ximgproc;
#                                WITH_TBB=OFF — fixes shutdown crash; minimal modules)
#   ui:                 qt6     (qtbase + qtdeclarative-less subset)
#   allocator:          mimalloc (override malloc/free for VC3D)
#
# What this DOES NOT build (use system packages):
#   - compiler toolchain (gcc/g++, gfortran), ninja, cmake, ccache, pkg-config
#   - X11/XCB/wayland/EGL/GL/OpenGL stack
#   - fontconfig / freetype / harfbuzz (Qt finds them from system)
#   - dbus, glib, gtk3 (qt platformthemes/gtk3 finds them)
#   - hwloc, scotch  (PaStiX is built by libs/flatboi/CMakeLists.txt via FetchContent;
#                     it uses these from system. If you want them rebuilt too,
#                     extend the script — they're moderate-effort additions.)
#
# Prerequisites (Debian/Ubuntu):
#   sudo apt install -y build-essential gcc g++ gfortran cmake ninja-build \
#       ccache pkg-config git wget xz-utils curl ca-certificates \
#       libx11-dev libxext-dev libxfixes-dev libxi-dev libxrender-dev \
#       libxcb1-dev libx11-xcb-dev libxcb-glx0-dev libxcb-keysyms1-dev \
#       libxcb-image0-dev libxcb-shm0-dev libxcb-icccm4-dev libxcb-sync-dev \
#       libxcb-xfixes0-dev libxcb-shape0-dev libxcb-randr0-dev \
#       libxcb-render-util0-dev libxcb-util-dev libxcb-xinerama0-dev \
#       libxcb-xkb-dev libxcb-cursor-dev libxkbcommon-dev libxkbcommon-x11-dev \
#       libgl-dev libegl-dev libfontconfig1-dev libfreetype-dev \
#       libharfbuzz-dev libdbus-1-dev libicu-dev libgtk-3-dev \
#       libavahi-client-dev libcurl4-openssl-dev libssl-dev \
#       libhwloc-dev libscotch-dev libmpfr-dev libgmp-dev \
#       libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
#       libswresample-dev
#
# Usage:
#   ./scripts/fromscratch.sh [PREFIX]
#
# Default PREFIX is ~/vc-deps. To build VC3D against it:
#   cmake -S . -B build/from-scratch -G Ninja \
#         -DCMAKE_PREFIX_PATH="$HOME/vc-deps" \
#         -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   cmake --build build/from-scratch
#
# The script is idempotent: each library has a .done marker in $PREFIX/.done/.
# Re-running skips finished libs. To force a rebuild, delete the marker
# (rm $PREFIX/.done/opencv && rerun).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX="${1:-$HOME/vc-deps}"
JOBS="${JOBS:-$(nproc)}"
BUILDDIR="${BUILDDIR:-$PREFIX/build}"
SRCDIR="${SRCDIR:-$PREFIX/src-cache}"
DONEDIR="$PREFIX/.done"

mkdir -p "$PREFIX" "$BUILDDIR" "$SRCDIR" "$DONEDIR"

# --- versions (pinned for reproducibility) -----------------------------------
ZLIB_VERSION="1.3.1"
ZSTD_VERSION="1.5.6"
LZ4_VERSION="1.10.0"
LIBDEFLATE_VERSION="1.22"
BLOSC1_VERSION="1.21.6"
JPEG_VERSION="3.0.4"
PNG_VERSION="1.6.44"
TIFF_VERSION="4.7.0"
GMP_VERSION="6.3.0"
MPFR_VERSION="4.2.1"
CGAL_VERSION="6.0.1"
EIGEN_VERSION="3.4.0"
OPENBLAS_VERSION="0.3.29"
METIS_VERSION="5.2.1"
SUITESPARSE_VERSION="7.8.3"
GFLAGS_VERSION="2.2.2"
GLOG_VERSION="0.7.1"
CERES_VERSION="2.2.0"
NLOHMANN_JSON_VERSION="3.11.3"
BOOST_VERSION="1.86.0"
OPENCV_VERSION="4.10.0"
OPENCV_CONTRIB_VERSION="4.10.0"
QT_VERSION="6.10.0"
MIMALLOC_VERSION="2.1.7"

# --- common flags ------------------------------------------------------------
# Use system gcc/g++/gfortran. ccache via launcher when available.
CC="$(command -v gcc)"
CXX="$(command -v g++)"
FC="$(command -v gfortran || true)"
export CC CXX FC

CCACHE_LAUNCHER=""
if command -v ccache >/dev/null 2>&1; then
    CCACHE_LAUNCHER="ccache"
fi

# Build flags: -O3 -march=native, position-independent (so static libs can be
# linked into the eventual shared bits), LTO off for now (LTO on opencv/qt
# explodes build time without much win for the things VC3D touches).
COMMON_C_FLAGS="-O3 -fPIC -DNDEBUG"
COMMON_CXX_FLAGS="-O3 -fPIC -DNDEBUG"

# RPATH: every binary built here should find its libs in $PREFIX/lib without
# LD_LIBRARY_PATH. Use $ORIGIN-relative so the tree is relocatable.
RPATH_FLAGS="-Wl,-rpath,\$ORIGIN/../lib -Wl,-rpath-link,${PREFIX}/lib"

CMAKE_COMMON=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DCMAKE_PREFIX_PATH="$PREFIX"
    -DCMAKE_C_COMPILER="$CC"
    -DCMAKE_CXX_COMPILER="$CXX"
    -DCMAKE_C_FLAGS="$COMMON_C_FLAGS"
    -DCMAKE_CXX_FLAGS="$COMMON_CXX_FLAGS"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib"
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
    # CMake 4 dropped support for projects requiring < 3.5. Many older
    # libraries (c-blosc 1.x, METIS, gflags, ...) still declare
    # cmake_minimum_required(VERSION 2.8/3.0). Bumping the policy version
    # lets them configure unchanged. Drop this once the upstreams catch up.
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)
if [ -n "$CCACHE_LAUNCHER" ]; then
    CMAKE_COMMON+=(
        -DCMAKE_C_COMPILER_LAUNCHER="$CCACHE_LAUNCHER"
        -DCMAKE_CXX_COMPILER_LAUNCHER="$CCACHE_LAUNCHER"
    )
fi

# --- helpers -----------------------------------------------------------------
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; NC='\033[0m'
log()   { echo -e "${GRN}[build]${NC} $*" >&2; }
warn()  { echo -e "${YEL}[warn ]${NC} $*" >&2; }
fatal() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

is_done() { [ -f "$DONEDIR/$1" ]; }
mark()    { touch "$DONEDIR/$1"; }

fetch() {
    # fetch <url> <tarball>
    local url="$1" tar="$2"
    if [ ! -f "$SRCDIR/$tar" ]; then
        log "fetch $tar"
        wget --no-verbose -O "$SRCDIR/$tar.part" "$url"
        mv "$SRCDIR/$tar.part" "$SRCDIR/$tar"
    fi
}

extract() {
    # extract <tarball> <expected-dirname>  →  echoes the absolute path
    local tar="$1" dirname="$2"
    local target="$BUILDDIR/$dirname"
    if [ ! -d "$target" ]; then
        log "extract $tar"
        mkdir -p "$target.tmp"
        tar -xf "$SRCDIR/$tar" -C "$target.tmp" --strip-components=1
        mv "$target.tmp" "$target"
    fi
    echo "$target"
}

# Wrap cmake configure+build+install around a single source dir.
cmake_build() {
    # cmake_build <name> <src> [extra cmake args ...]
    local name="$1" src="$2"; shift 2
    local bld="$BUILDDIR/$name-bld"
    rm -rf "$bld"
    cmake -S "$src" -B "$bld" "${CMAKE_COMMON[@]}" "$@"
    cmake --build "$bld" --parallel "$JOBS"
    cmake --install "$bld"
}

# --- builders ---------------------------------------------------------------

build_zlib() {
    is_done zlib && return 0
    fetch "https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.xz" "zlib-${ZLIB_VERSION}.tar.xz"
    local src; src=$(extract "zlib-${ZLIB_VERSION}.tar.xz" "zlib-${ZLIB_VERSION}")
    cmake_build zlib "$src"
    mark zlib
}

build_zstd() {
    is_done zstd && return 0
    fetch "https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz" "zstd-${ZSTD_VERSION}.tar.gz"
    local src; src=$(extract "zstd-${ZSTD_VERSION}.tar.gz" "zstd-${ZSTD_VERSION}")
    # zstd ships its cmake in build/cmake/
    cmake_build zstd "$src/build/cmake" \
        -DZSTD_BUILD_PROGRAMS=OFF \
        -DZSTD_BUILD_TESTS=OFF
    mark zstd
}

build_lz4() {
    is_done lz4 && return 0
    fetch "https://github.com/lz4/lz4/archive/v${LZ4_VERSION}.tar.gz" "lz4-${LZ4_VERSION}.tar.gz"
    local src; src=$(extract "lz4-${LZ4_VERSION}.tar.gz" "lz4-${LZ4_VERSION}")
    cmake_build lz4 "$src/build/cmake" \
        -DBUILD_STATIC_LIBS=OFF \
        -DLZ4_BUILD_CLI=OFF \
        -DLZ4_BUILD_LEGACY_LZ4C=OFF
    mark lz4
}

build_libdeflate() {
    is_done libdeflate && return 0
    fetch "https://github.com/ebiggers/libdeflate/archive/v${LIBDEFLATE_VERSION}.tar.gz" "libdeflate-${LIBDEFLATE_VERSION}.tar.gz"
    local src; src=$(extract "libdeflate-${LIBDEFLATE_VERSION}.tar.gz" "libdeflate-${LIBDEFLATE_VERSION}")
    cmake_build libdeflate "$src" \
        -DLIBDEFLATE_BUILD_GZIP=OFF
    mark libdeflate
}

build_blosc1() {
    is_done blosc1 && return 0
    fetch "https://github.com/Blosc/c-blosc/archive/v${BLOSC1_VERSION}.tar.gz" "c-blosc-${BLOSC1_VERSION}.tar.gz"
    local src; src=$(extract "c-blosc-${BLOSC1_VERSION}.tar.gz" "c-blosc-${BLOSC1_VERSION}")
    cmake_build blosc1 "$src" \
        -DBUILD_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_FUZZERS=OFF \
        -DPREFER_EXTERNAL_ZLIB=ON \
        -DPREFER_EXTERNAL_ZSTD=ON \
        -DPREFER_EXTERNAL_LZ4=ON
    mark blosc1
}

build_jpeg() {
    is_done jpeg && return 0
    fetch "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${JPEG_VERSION}/libjpeg-turbo-${JPEG_VERSION}.tar.gz" "libjpeg-turbo-${JPEG_VERSION}.tar.gz"
    local src; src=$(extract "libjpeg-turbo-${JPEG_VERSION}.tar.gz" "libjpeg-turbo-${JPEG_VERSION}")
    cmake_build jpeg "$src" \
        -DENABLE_STATIC=OFF \
        -DWITH_TURBOJPEG=OFF
    mark jpeg
}

build_png() {
    is_done png && return 0
    fetch "https://download.sourceforge.net/libpng/libpng-${PNG_VERSION}.tar.xz" "libpng-${PNG_VERSION}.tar.xz"
    local src; src=$(extract "libpng-${PNG_VERSION}.tar.xz" "libpng-${PNG_VERSION}")
    cmake_build png "$src" \
        -DPNG_STATIC=OFF \
        -DPNG_TESTS=OFF \
        -DPNG_TOOLS=OFF
    mark png
}

build_tiff() {
    is_done tiff && return 0
    fetch "https://download.osgeo.org/libtiff/tiff-${TIFF_VERSION}.tar.xz" "tiff-${TIFF_VERSION}.tar.xz"
    local src; src=$(extract "tiff-${TIFF_VERSION}.tar.xz" "tiff-${TIFF_VERSION}")
    cmake_build tiff "$src" \
        -Dtiff-tools=OFF \
        -Dtiff-tests=OFF \
        -Dtiff-contrib=OFF \
        -Dtiff-docs=OFF \
        -Dwebp=OFF \
        -Djbig=OFF \
        -Dlerc=OFF
    mark tiff
}

# CGAL is mostly header-only; configures fine against external gmp+mpfr.
# Use system libgmp-dev / libmpfr-dev for simplicity (they're tiny and stable).
build_cgal() {
    is_done cgal && return 0
    fetch "https://github.com/CGAL/cgal/releases/download/v${CGAL_VERSION}/CGAL-${CGAL_VERSION}.tar.xz" "CGAL-${CGAL_VERSION}.tar.xz"
    local src; src=$(extract "CGAL-${CGAL_VERSION}.tar.xz" "CGAL-${CGAL_VERSION}")
    cmake_build cgal "$src" \
        -DWITH_examples=OFF \
        -DWITH_demos=OFF \
        -DWITH_tests=OFF
    mark cgal
}

build_eigen() {
    is_done eigen && return 0
    fetch "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" "eigen-${EIGEN_VERSION}.tar.gz"
    local src; src=$(extract "eigen-${EIGEN_VERSION}.tar.gz" "eigen-${EIGEN_VERSION}")
    cmake_build eigen "$src" \
        -DBUILD_TESTING=OFF \
        -DEIGEN_BUILD_DOC=OFF
    mark eigen
}

# OpenBLAS with embedded LAPACK + LAPACKE (Fortran). Built with gfortran.
# DYNAMIC_ARCH=0 + TARGET=auto: tune to this machine (matches march=native).
build_openblas() {
    is_done openblas && return 0
    fetch "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" "OpenBLAS-${OPENBLAS_VERSION}.tar.gz"
    local src; src=$(extract "OpenBLAS-${OPENBLAS_VERSION}.tar.gz" "OpenBLAS-${OPENBLAS_VERSION}")
    [ -n "$FC" ] || fatal "gfortran not found; install gfortran for OpenBLAS LAPACK"
    # Use OpenBLAS's native Makefile — its cmake driver has assorted bugs in
    # the prebuild getarch_2nd step depending on toolchain. Native make is
    # what every distro uses and is reliably green.
    pushd "$src" >/dev/null
        make -j"$JOBS" \
            CC="$CC" FC="$FC" \
            COMMON_OPT="$COMMON_C_FLAGS" \
            USE_THREAD=1 USE_OPENMP=1 NUM_THREADS=128 \
            DYNAMIC_ARCH=0 NO_STATIC=1 NO_SHARED=0 \
            LIBPREFIX=libopenblas \
            BUILD_LAPACK_DEPRECATED=0 NO_AFFINITY=1
        make install PREFIX="$PREFIX"
    popd >/dev/null
    # OpenBLAS's Makefile doesn't drop a BLAS/LAPACK FindConfig that
    # Ceres/SuiteSparse will pick up via find_package. They both have
    # legacy paths that work given the .so layout + pkgconfig OpenBLAS
    # installs in $PREFIX/lib/pkgconfig/openblas.pc.
    mark openblas
}

# METIS via GKlib (modern packaging from KarypisLab). Need GKlib first.
build_gklib() {
    is_done gklib && return 0
    fetch "https://github.com/KarypisLab/GKlib/archive/refs/heads/master.tar.gz" "gklib-master.tar.gz"
    local src; src=$(extract "gklib-master.tar.gz" "GKlib-master")
    # GKlib's wrapper Makefile calls cmake. Just call cmake directly so we
    # control flags / install dir.
    cmake_build gklib "$src" \
        -DNO_X86=0
    mark gklib
}

build_metis() {
    is_done metis && return 0
    fetch "https://github.com/KarypisLab/METIS/archive/v${METIS_VERSION}.tar.gz" "metis-${METIS_VERSION}.tar.gz"
    local src; src=$(extract "metis-${METIS_VERSION}.tar.gz" "METIS-${METIS_VERSION}")
    # METIS declares cmake_minimum_required(VERSION 2.8) which CMake 4
    # rejects. Its wrapper Makefile calls cmake without forwarding our
    # CMAKE_POLICY_VERSION_MINIMUM, so patch the source instead.
    sed -i 's/cmake_minimum_required(VERSION 2\.8)/cmake_minimum_required(VERSION 3.10)/' "$src/CMakeLists.txt"
    pushd "$src" >/dev/null
        # IDXTYPEWIDTH=32 is required by Eigen's MetisSupport, which Ceres
        # pulls in. METIS's wrapper Makefile defaults to 32-bit when neither
        # `i64` nor `r64` are passed; the env-var check is `ifneq($(i64),not-set)`
        # so passing i64=0 *would* still flip it to 64. Omit those args.
        make config shared=1 cc="$CC" prefix="$PREFIX" gklib_path="$PREFIX"
        make -j"$JOBS"
        make install
    popd >/dev/null
    mark metis
}

# SuiteSparse: CHOLMOD/AMD/CAMD/COLAMD/CCOLAMD/SPQR/UMFPACK/etc.
# We need at least CHOLMOD + AMD + COLAMD for Ceres SuiteSparse support.
build_suitesparse() {
    is_done suitesparse && return 0
    fetch "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v${SUITESPARSE_VERSION}.tar.gz" "suitesparse-${SUITESPARSE_VERSION}.tar.gz"
    local src; src=$(extract "suitesparse-${SUITESPARSE_VERSION}.tar.gz" "SuiteSparse-${SUITESPARSE_VERSION}")
    cmake_build suitesparse "$src" \
        -DBUILD_TESTING=OFF \
        -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;camd;ccolamd;colamd;cholmod;spqr" \
        -DBLA_VENDOR=OpenBLAS \
        -DCHOLMOD_GPL=ON \
        -DCHOLMOD_PARTITION=ON
    mark suitesparse
}

build_gflags() {
    is_done gflags && return 0
    fetch "https://github.com/gflags/gflags/archive/v${GFLAGS_VERSION}.tar.gz" "gflags-${GFLAGS_VERSION}.tar.gz"
    local src; src=$(extract "gflags-${GFLAGS_VERSION}.tar.gz" "gflags-${GFLAGS_VERSION}")
    cmake_build gflags "$src" \
        -DBUILD_TESTING=OFF \
        -DGFLAGS_BUILD_STATIC_LIBS=OFF \
        -DGFLAGS_BUILD_SHARED_LIBS=ON
    mark gflags
}

build_glog() {
    is_done glog && return 0
    fetch "https://github.com/google/glog/archive/v${GLOG_VERSION}.tar.gz" "glog-${GLOG_VERSION}.tar.gz"
    local src; src=$(extract "glog-${GLOG_VERSION}.tar.gz" "glog-${GLOG_VERSION}")
    cmake_build glog "$src" \
        -DBUILD_TESTING=OFF \
        -DWITH_GFLAGS=ON \
        -DWITH_UNWIND=OFF
    mark glog
}

build_ceres() {
    is_done ceres && return 0
    fetch "https://github.com/ceres-solver/ceres-solver/archive/${CERES_VERSION}.tar.gz" "ceres-${CERES_VERSION}.tar.gz"
    local src; src=$(extract "ceres-${CERES_VERSION}.tar.gz" "ceres-solver-${CERES_VERSION}")
    cmake_build ceres "$src" \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DUSE_CUDA=OFF \
        -DEIGENSPARSE=ON \
        -DSUITESPARSE=ON \
        -DSCHUR_SPECIALIZATIONS=ON
    mark ceres
}

build_json() {
    is_done json && return 0
    fetch "https://github.com/nlohmann/json/archive/v${NLOHMANN_JSON_VERSION}.tar.gz" "nlohmann-json-${NLOHMANN_JSON_VERSION}.tar.gz"
    local src; src=$(extract "nlohmann-json-${NLOHMANN_JSON_VERSION}.tar.gz" "json-${NLOHMANN_JSON_VERSION}")
    cmake_build json "$src" \
        -DJSON_BuildTests=OFF \
        -DJSON_Install=ON
    mark json
}

# Boost — only program_options actually used by VC3D, but the cmake config needs
# more headers in place. Use the bootstrap+b2 path (boost doesn't ship cmake
# natively in a way cmake's find_package likes).
build_boost() {
    is_done boost && return 0
    local v_under="${BOOST_VERSION//./_}"
    fetch "https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${v_under}.tar.gz" "boost-${BOOST_VERSION}.tar.gz"
    local src; src=$(extract "boost-${BOOST_VERSION}.tar.gz" "boost_${v_under}")
    pushd "$src" >/dev/null
        ./bootstrap.sh --prefix="$PREFIX" --with-libraries=program_options
        # b2's `cmake-install` target generates the CMake config files that
        # find_package(Boost CONFIG REQUIRED) expects; without them VC3D will
        # only find Boost via legacy FindBoost.cmake.
        ./b2 install -j"$JOBS" link=shared variant=release \
            cflags="$COMMON_C_FLAGS" cxxflags="$COMMON_CXX_FLAGS" \
            --layout=system
    popd >/dev/null
    mark boost
}

# OpenCV: minimum-viable for VC3D. WITH_TBB=OFF is the critical flag — fixes
# the libtbbmalloc shutdown crash. Modules limited to what VC3D actually
# #includes: core, imgproc, imgcodecs, calib3d, flann, features2d (for ximgproc).
# All third-party image bloat off (WEBP/JPEG2000/OpenEXR/JasPer).
build_opencv() {
    is_done opencv && return 0
    fetch "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz" "opencv-${OPENCV_VERSION}.tar.gz"
    fetch "https://github.com/opencv/opencv_contrib/archive/${OPENCV_CONTRIB_VERSION}.tar.gz" "opencv-contrib-${OPENCV_CONTRIB_VERSION}.tar.gz"
    local src; src=$(extract "opencv-${OPENCV_VERSION}.tar.gz" "opencv-${OPENCV_VERSION}")
    local contrib; contrib=$(extract "opencv-contrib-${OPENCV_CONTRIB_VERSION}.tar.gz" "opencv_contrib-${OPENCV_CONTRIB_VERSION}")
    cmake_build opencv "$src" \
        -DOPENCV_EXTRA_MODULES_PATH="$contrib/modules" \
        -DBUILD_LIST="core,imgproc,imgcodecs,calib3d,flann,features2d,ximgproc,videoio" \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_java=OFF \
        -DWITH_TBB=OFF \
        -DWITH_IPP=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_GDAL=OFF \
        -DWITH_GDCM=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_V4L=OFF \
        -DWITH_LIBV4L=OFF \
        -DWITH_VTK=OFF \
        -DWITH_PROTOBUF=OFF \
        -DWITH_QUIRC=OFF \
        -DWITH_ITT=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENCLAMDFFT=OFF \
        -DWITH_OPENCLAMDBLAS=OFF \
        -DBUILD_opencv_world=OFF \
        -DOPENCV_GENERATE_PKGCONFIG=OFF
    mark opencv
}

# Qt6: only qtbase. qtbase ships Core/Gui/Widgets/Concurrent/Network/OpenGLWidgets
# — which is the exact set VC3D links. To skip all the other Qt modules we build
# qtbase as a standalone CMake subtree (the qt-everywhere bundle has each module
# as a subdir; qtbase is the root).
build_qt6() {
    is_done qt6 && return 0
    fetch "https://download.qt.io/official_releases/qt/${QT_VERSION%.*}/${QT_VERSION}/single/qt-everywhere-src-${QT_VERSION}.tar.xz" "qt-everywhere-src-${QT_VERSION}.tar.xz"
    local src; src=$(extract "qt-everywhere-src-${QT_VERSION}.tar.xz" "qt-everywhere-src-${QT_VERSION}")
    cmake_build qt6 "$src/qtbase" \
        -DQT_BUILD_EXAMPLES=OFF \
        -DQT_BUILD_TESTS=OFF \
        -DQT_BUILD_TOOLS_WHEN_CROSSCOMPILING=OFF \
        -DFEATURE_vulkan=OFF \
        -DFEATURE_sql=OFF \
        -DFEATURE_printsupport=OFF \
        -DFEATURE_pdf=OFF
    mark qt6
}

build_mimalloc() {
    is_done mimalloc && return 0
    fetch "https://github.com/microsoft/mimalloc/archive/v${MIMALLOC_VERSION}.tar.gz" "mimalloc-${MIMALLOC_VERSION}.tar.gz"
    local src; src=$(extract "mimalloc-${MIMALLOC_VERSION}.tar.gz" "mimalloc-${MIMALLOC_VERSION}")
    cmake_build mimalloc "$src" \
        -DMI_BUILD_TESTS=OFF \
        -DMI_BUILD_OBJECT=OFF \
        -DMI_BUILD_STATIC=OFF
    mark mimalloc
}

# --- main --------------------------------------------------------------------
main() {
    log "PREFIX  = $PREFIX"
    log "JOBS    = $JOBS"
    log "CC/CXX  = $CC / $CXX"
    log "FC      = ${FC:-<not found>}"
    log "ccache  = ${CCACHE_LAUNCHER:-<not found>}"
    log ""

    # 1. compression (build first; everything else may link these)
    build_zlib
    build_zstd
    build_lz4
    build_libdeflate
    build_blosc1

    # 2. image codecs
    build_jpeg
    build_png
    build_tiff

    # 3. math
    build_eigen
    build_openblas
    build_gklib
    build_metis
    build_suitesparse
    build_gflags
    build_glog
    build_ceres

    # 4. geometry
    build_cgal

    # 5. data
    build_json
    build_boost

    # 6. imaging
    build_opencv

    # 7. ui
    build_qt6

    # 8. allocator
    build_mimalloc

    log ""
    log "=== Done. Configure VC3D with:"
    log "  cmake -S . -B build/from-scratch -G Ninja \\"
    log "    -DCMAKE_PREFIX_PATH='$PREFIX' \\"
    log "    -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    log "  cmake --build build/from-scratch"
}

main "$@"
