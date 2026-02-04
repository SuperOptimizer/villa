#!/usr/bin/env bash
# build_dependencies_macos.sh - macOS version using Homebrew LLVM toolchain
# - Builds VC3D WITHOUT PaStiX (PaStiX/Scotch are Linux-only in this flow)
# - Uses Homebrew LLVM for C++23 support
# - Uses standalone libomp (not LLVM's) to match OpenBLAS/Ceres
# - Fetches libigl from GitHub at a pinned commit and overlays libs/libigl_changes
#
# Note on libc++: Homebrew packages (Qt, OpenCV, Ceres) link to system libc++
# while code we compile uses Homebrew LLVM's libc++. This works because:
# - Qt uses its own types (QString, QObject*) at API boundaries
# - OpenCV uses cv::Mat, not std::vector at API boundaries
# - Most std:: types are ABI-compatible between recent libc++ versions
#
# Note on OpenMP: We use standalone libomp package, NOT LLVM's libomp, because
# OpenBLAS (used by Ceres) links to standalone libomp. Using LLVM's libomp
# would cause "multiple OpenMP runtimes" error at runtime.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
LLVM_PREFIX="$(brew --prefix llvm)"
LIBOMP_PREFIX="$(brew --prefix libomp)"

export CC="${LLVM_PREFIX}/bin/clang"
export CXX="${LLVM_PREFIX}/bin/clang++"
export AR="${LLVM_PREFIX}/bin/llvm-ar"
export RANLIB="${LLVM_PREFIX}/bin/llvm-ranlib"
export NM="${LLVM_PREFIX}/bin/llvm-nm"

# Use Homebrew LLVM's libc++ for consistent ABI in our code
LLVM_LIBCXX_FLAGS="-stdlib=libc++ -L${LLVM_PREFIX}/lib/c++ -Wl,-rpath,${LLVM_PREFIX}/lib/c++"

export INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/vc-dependencies}"
export BUILD_DIR="${BUILD_DIR:-$HOME/vc-dependencies-build}"
export COMMON_FLAGS="-march=native -w"
export COMMON_LDFLAGS="${LLVM_LIBCXX_FLAGS}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBS_DIR="$REPO_ROOT/libs"

# libigl pin (latest on main as of 2025-10-02)
LIBIGL_COMMIT="ae8f959ea26d7059abad4c698aba8d6b7c3205e8"
LIBIGL_DIR="$LIBS_DIR/libigl"
LIBIGL_CHANGES_DIR="$LIBS_DIR/libigl_changes"

# Determine parallelism
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
export JOBS

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

# ---------------------------------------------------------------------------
# Check we're on macOS
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS. Use build_dependencies.sh for Linux." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Check Homebrew dependencies
# ---------------------------------------------------------------------------
log "Checking Homebrew dependencies"

REQUIRED_PACKAGES=(
  llvm
  ccache
  ninja
  cmake
  pkg-config
  qt@6
  boost
  ceres-solver
  opencv
  xsimd
  c-blosc
  spdlog
  gsl
  sdl2
  curl
  eigen
  libtiff
  nlohmann-json
  libomp
)

MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
  if ! brew list "$pkg" &>/dev/null; then
    MISSING_PACKAGES+=("$pkg")
  fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
  log "Installing missing packages: ${MISSING_PACKAGES[*]}"
  brew install "${MISSING_PACKAGES[@]}"
fi

# Verify LLVM is properly installed
if [[ ! -x "${CC}" ]]; then
  echo "ERROR: Homebrew LLVM not found at ${LLVM_PREFIX}" >&2
  echo "Install with: brew install llvm" >&2
  exit 1
fi

log "Using Homebrew LLVM toolchain:"
echo "  CC:  ${CC}"
echo "  CXX: ${CXX}"
"${CXX}" --version | head -1

# ---------------------------------------------------------------------------
# Create build directories
# ---------------------------------------------------------------------------
log "Creating build directories"
mkdir -p "$BUILD_DIR" "$INSTALL_PREFIX"

# Helper to run cmake with consistent toolchain
run_cmake() {
  cmake "$@" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_AR="${AR}" \
    -DCMAKE_RANLIB="${RANLIB}" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS} ${LLVM_LIBCXX_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}"
}

# ---------------------------------------------------------------------------
# z5 (pinned) → $INSTALL_PREFIX
# ---------------------------------------------------------------------------
if [[ ! -f "$INSTALL_PREFIX/include/z5/z5.hxx" ]]; then
  log "Building z5 (pinned) into $INSTALL_PREFIX"
  pushd "$BUILD_DIR" >/dev/null
  rm -rf z5
  git clone https://github.com/constantinpape/z5.git z5
  pushd z5 >/dev/null
  Z5_COMMIT=ee2081bb974fe0d0d702538400c31c38b09f1629
  git fetch origin "$Z5_COMMIT" --depth 1
  git checkout --detach "$Z5_COMMIT"
  # Align z5 with xtensor 0.25's header layout
  sed -i '' 's|xtensor/containers/xadapt.hpp|xtensor/xadapt.hpp|' \
    include/z5/multiarray/xtensor_util.hxx || true
  popd >/dev/null

  run_cmake -S z5 -B z5/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DWITH_BLOSC=ON -DWITH_ZLIB=ON -DBUILD_Z5PY=OFF -DBUILD_TESTS=OFF
  cmake --build z5/build -j"$JOBS"
  cmake --install z5/build
  popd >/dev/null
else
  log "z5 already installed, skipping"
fi

# ---------------------------------------------------------------------------
# libigl (Git clone + overlay)
# ---------------------------------------------------------------------------
if [[ ! -d "$LIBIGL_DIR/.git" ]]; then
  log "Cloning libigl at pinned commit into $LIBIGL_DIR"
  rm -rf "$LIBIGL_DIR"
  git clone https://github.com/libigl/libigl.git "$LIBIGL_DIR"
  pushd "$LIBIGL_DIR" >/dev/null
  git fetch origin "$LIBIGL_COMMIT" --depth 1
  git checkout --detach "$LIBIGL_COMMIT"
  git submodule update --init --recursive
  popd >/dev/null

  if [[ -d "$LIBIGL_CHANGES_DIR" ]]; then
    log "Overlaying custom changes from $LIBIGL_CHANGES_DIR into $LIBIGL_DIR"
    cp -a "$LIBIGL_CHANGES_DIR/." "$LIBIGL_DIR/"
  fi
else
  log "libigl already cloned, skipping"
fi

# ---------------------------------------------------------------------------
# Skip Flatboi on macOS (requires PaStiX/Scotch which are Linux-only here)
# ---------------------------------------------------------------------------
log "Skipping Flatboi build (PaStiX/Scotch not available on macOS in this flow)"

# ---------------------------------------------------------------------------
# Build main project (VC3D) WITHOUT PaStiX
# ---------------------------------------------------------------------------
log "Configuring & building VC3D (no PaStiX)"
mkdir -p "$REPO_ROOT/build"
pushd "$REPO_ROOT/build" >/dev/null

run_cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX;$(brew --prefix)" \
  -DVC_BUILD_Z5=OFF \
  -DVC_BUILD_JSON=OFF \
  -DVC_WITH_PASTIX=OFF \
  -DVC_WITH_CUDA_SPARSE=OFF \
  -DVC_USE_OPENMP=ON

cmake --build . -j"$JOBS"
popd >/dev/null

log "VC3D built successfully (without PaStiX)."

# ---------------------------------------------------------------------------
# Verify build
# ---------------------------------------------------------------------------
log "Verifying build"
"$REPO_ROOT/build/bin/VC3D" --version

log "Checking library linkage:"
echo "libc++:"
otool -L "$REPO_ROOT/build/bin/VC3D" | grep "c++" || echo "  (none found)"
echo "OpenMP:"
otool -L "$REPO_ROOT/build/bin/VC3D" | grep "omp" | grep -v opencv || echo "  (none found)"

log "All done."
echo ""
echo "Built artifacts:"
echo "  VC3D:         $REPO_ROOT/build/bin/VC3D"
echo "  CLI tools:    $REPO_ROOT/build/bin/"
echo "  Dependencies: $INSTALL_PREFIX"
echo ""
echo "To run VC3D:"
echo "  $REPO_ROOT/build/bin/VC3D"
