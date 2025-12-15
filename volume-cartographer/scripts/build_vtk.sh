#!/usr/bin/env bash
# build_vtk.sh - Build VTK from source with Qt6 support
# This is needed because the system VTK is built with Qt5, but VC3D uses Qt6
#
# Prerequisites: Run setup_vtk_sudo.sh first to install system dependencies

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/vc-dependencies}"
export BUILD_DIR="${BUILD_DIR:-$HOME/vc-dependencies-build}"
export COMMON_FLAGS="-march=native -w"

# VTK git branch - use latest master for bug fixes
VTK_GIT_REF="${VTK_GIT_REF:-master}"
VTK_REPO="https://gitlab.kitware.com/vtk/vtk.git"

# Determine parallelism
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
else
  JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi
export JOBS

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33mWARNING: %s\033[0m\n" "$*"; }
error() { printf "\033[1;31mERROR: %s\033[0m\n" "$*"; }

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Linux. For macOS, adapt packages/paths." >&2
  exit 1
fi

log "Checking prerequisites"

# Check for required tools
for cmd in cmake ninja wget; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    error "$cmd is required but not found. Run setup_vtk_sudo.sh first."
    exit 1
  fi
done

# Check for Qt6
if ! pkg-config --exists Qt6Core 2>/dev/null; then
  error "Qt6 not found. Run setup_vtk_sudo.sh first to install dependencies."
  exit 1
fi

# Check for OpenGL dev headers
if [[ ! -f /usr/include/GL/gl.h ]]; then
  error "OpenGL headers not found. Run setup_vtk_sudo.sh first."
  exit 1
fi

log "All prerequisites satisfied"

# ---------------------------------------------------------------------------
# Create build directories
# ---------------------------------------------------------------------------
mkdir -p "$BUILD_DIR" "$INSTALL_PREFIX"

# ---------------------------------------------------------------------------
# Clone VTK from git
# ---------------------------------------------------------------------------
VTK_SRC="$BUILD_DIR/vtk"

if [[ -d "$VTK_SRC" ]]; then
  log "VTK source already exists at $VTK_SRC, updating..."
  pushd "$VTK_SRC" >/dev/null
  git fetch origin
  git checkout "$VTK_GIT_REF"
  git pull origin "$VTK_GIT_REF" 2>/dev/null || true
  popd >/dev/null
else
  log "Cloning VTK ($VTK_GIT_REF)"
  git clone --depth 1 --branch "$VTK_GIT_REF" "$VTK_REPO" "$VTK_SRC"
fi

# ---------------------------------------------------------------------------
# Configure VTK with Qt6 and required modules
# ---------------------------------------------------------------------------
log "Configuring VTK ($VTK_GIT_REF) with Qt6 support"

VTK_BUILD="$BUILD_DIR/vtk-build"
rm -rf "$VTK_BUILD"
mkdir -p "$VTK_BUILD"

cmake -S "$VTK_SRC" -B "$VTK_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="$COMMON_FLAGS" \
  -DCMAKE_C_FLAGS="$COMMON_FLAGS" \
  \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCUMENTATION=OFF \
  \
  -DVTK_GROUP_ENABLE_Qt=YES \
  -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES \
  -DVTK_MODULE_ENABLE_VTK_RenderingQt=YES \
  -DVTK_QT_VERSION=6 \
  \
  -DVTK_GROUP_ENABLE_Rendering=YES \
  -DVTK_MODULE_ENABLE_VTK_RenderingCore=YES \
  -DVTK_MODULE_ENABLE_VTK_RenderingVolume=YES \
  -DVTK_MODULE_ENABLE_VTK_RenderingVolumeOpenGL2=YES \
  -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2=YES \
  -DVTK_MODULE_ENABLE_VTK_InteractionStyle=YES \
  \
  -DVTK_MODULE_ENABLE_VTK_CommonCore=YES \
  -DVTK_MODULE_ENABLE_VTK_CommonDataModel=YES \
  -DVTK_MODULE_ENABLE_VTK_CommonExecutionModel=YES \
  -DVTK_MODULE_ENABLE_VTK_FiltersSources=YES \
  \
  -DVTK_GROUP_ENABLE_Views=NO \
  -DVTK_GROUP_ENABLE_Web=NO \
  -DVTK_GROUP_ENABLE_MPI=NO \
  -DVTK_MODULE_ENABLE_VTK_RenderingRayTracing=NO \
  -DVTK_WRAP_PYTHON=OFF \
  -DVTK_WRAP_JAVA=OFF

# ---------------------------------------------------------------------------
# Build VTK
# ---------------------------------------------------------------------------
log "Building VTK (this may take a while...)"
cmake --build "$VTK_BUILD" -j"$JOBS"

# ---------------------------------------------------------------------------
# Install VTK
# ---------------------------------------------------------------------------
log "Installing VTK to $INSTALL_PREFIX"
cmake --install "$VTK_BUILD"

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
log "Verifying VTK installation"

if ls "$INSTALL_PREFIX"/lib/cmake/vtk-*/vtk-config.cmake >/dev/null 2>&1; then
  log "VTK cmake config found"
else
  warn "VTK cmake config not found at expected location"
fi

# Check for Qt6 integration
if ls "$INSTALL_PREFIX"/lib/libvtkGUISupportQt*.so* >/dev/null 2>&1; then
  log "VTK Qt support libraries found"
else
  warn "VTK Qt support libraries not found"
fi

# ---------------------------------------------------------------------------
# Print usage instructions
# ---------------------------------------------------------------------------
cat << EOF

================================================================================
VTK ($VTK_GIT_REF) with Qt6 support has been installed to:
  $INSTALL_PREFIX

To use this VTK with your CMake project, add to your cmake command:
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX"

Or set the environment variable:
  export CMAKE_PREFIX_PATH="$INSTALL_PREFIX:\$CMAKE_PREFIX_PATH"

Example cmake configuration for VC3D:
  cmake .. -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \\
    -DVC_WITH_VTK=ON

================================================================================
EOF

log "VTK build complete!"
