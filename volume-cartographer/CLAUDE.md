# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Volume Cartographer (VC3D) is a C++23/Qt6 toolkit for virtually unwrapping volumetric datasets, specialized for analyzing Herculaneum papyri. It consists of the VC3D GUI application and 25+ command-line tools.

**Recommended deployment:** Docker image `ghcr.io/scrollprize/villa/volume-cartographer:edge`

## Build Commands

```bash
# Standard build (requires dependencies - see ubuntu-24.04-noble.Dockerfile)
mkdir build && cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja

# Run GUI
./bin/VC3D

# Build types: Debug, Release, QuickBuild, RelWithDebInfo
```

**Key CMake options:**
- `VC_USE_OPENMP=ON` - Enable OpenMP parallelization
- `VC_WITH_PASTIX=OFF` - PaStiX sparse solver
- `VC_ENABLE_ASAN/UBSAN/TSAN/LSAN=OFF` - Sanitizers (mutually exclusive)

## Code Style

Uses `clang-format` (Google-based, 140 column limit, 4-space indent). All code must be formatted before commit:

```bash
git add .
git clang-format
git add . && git commit
```

## Architecture

### Libraries (core/)
- **vc_core** - Core data types (Volume, Segmentation, VolumePkg), surface representation (QuadSurface), volume I/O (Zarr), rendering, geometry
- **vc_ui** - Qt6 utilities, VCCollection, surface metrics
- **vc_tracer** - Segmentation algorithms (GrowPatch, GrowSurface, NeuralTracerConnection), Ceres optimization

### GUI (apps/VC3D/)
- `CWindow.cpp` - Main window (very large file)
- `CVolumeViewer.*` - 3D OpenGL visualization
- `segmentation/` - Editing tools and controllers
- `overlays/` - 20+ overlay types

### CLI Tools (apps/src/)
Key tools: `vc_render_tifxyz`, `vc_grow_seg_from_seed`, `vc_flatten`, `vc_calc_surface_metrics`, `vc_gen_normalgrids`, `vc_tifxyz2obj`

### Key Headers (core/include/vc/)
- `core/types/Volume.hpp` - Zarr volume dataset
- `core/types/VolumePkg.hpp` - Package container (volumes + segmentations)
- `core/types/Segmentation.hpp` - Surface + metadata
- `core/util/QuadSurface.hpp` - Main surface type (2D grid → 3D coords)
- `core/util/GridStore.hpp` - 3D normal vector fields

## Data Format

### Coordinate Systems
- **Internal (Zarr):** ZYX ordering - `shape[0]=Z, shape[1]=Y, shape[2]=X`
- **API:** XYZ ordering - methods return `{width, height, slices}`

### Volume Package Structure
```
scroll.volpkg/
├── config.json
├── volumes/
│   └── volume.zarr/     # Multi-resolution Zarr (0/, 1/, 2/...)
│       └── meta.json    # Required: type, uuid, width, height, slices, voxelsize, format:"zarr"
└── paths/               # Segmentations
    └── segment_id/
        ├── x.tif, y.tif, z.tif  # Surface coordinates (uint16)
        ├── meta.json
        └── mask.tif (optional)
```

## Key Patterns

- Factory methods: `static std::shared_ptr<T> T::New(...)`
- File paths: Always use `std::filesystem::path`
- Points: `cv::Vec3f` for 3D points, `cv::Mat_<cv::Vec3f>` for grids
- Parallelization: OpenMP pragmas `#pragma omp parallel for`
- JSON validation: `vc::json::require_fields()`, `require_type()`

## Dependencies

Major: Qt6, OpenCV, Ceres, Eigen3, Boost (program_options), nlohmann/json, z5 (Zarr)

Vendored via FetchContent: xtensor stack, z5, libigl (with local patches)

## Notes

- Ubuntu users: Run `ulimit -n 750000` before VC3D (default limit too low)
- Building from source requires nix-like environment for atomic rename support
- On Windows, use Docker or WSL
