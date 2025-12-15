#!/usr/bin/env bash
# setup_vtk_sudo.sh - Install system dependencies for building VTK with Qt6
# Run this once with sudo before running build_vtk.sh

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Linux (Ubuntu/Debian)." >&2
  exit 1
fi

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run with sudo or as root." >&2
  exit 1
fi

log "Updating package lists"
apt-get update

log "Installing VTK build dependencies"
apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  ninja-build \
  clang \
  wget \
  ca-certificates \
  pkg-config \
  \
  qt6-base-dev \
  libqt6opengl6-dev \
  \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libegl1-mesa-dev \
  libgles2-mesa-dev \
  \
  libx11-dev \
  libxext-dev \
  libxt-dev \
  libxrender-dev \
  libxcursor-dev \
  libxrandr-dev \
  libxinerama-dev \
  libxi-dev

log "VTK build dependencies installed successfully"
log "You can now run build_vtk.sh (without sudo)"
