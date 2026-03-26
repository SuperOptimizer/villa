# vc3d → vc4d Migration Guide

## Philosophy

vc4d is a clean-room rewrite. It is **not** backwards-compatible with vc3d source code, but it reads the same file formats (volpkg, zarr, tifxyz, OBJ). Porting code from vc3d to vc4d means adopting new types and patterns.

## Type Mapping

| vc3d | vc4d | Notes |
|------|------|-------|
| `cv::Vec3f` | `vc4d::Vec3f` | Aggregate, constexpr, no OpenCV |
| `cv::Vec2f` | `vc4d::Vec2f` | Same |
| `cv::Mat_<cv::Vec3f>` | `vc4d::Grid<Vec3f>` | `optional<T>` cells instead of -1 sentinel |
| `cv::Mat_<uint8_t>` | `vc4d::Grid<uint8_t>` or `vc4d::ByteGrid` | Same pattern |
| `Rect3D` | `vc4d::Box3f` | Full AABB with intersect/contains/merge |
| `Surface` (base class) | *(removed)* | Only QuadSurface exists |
| `QuadSurface` | `vc4d::QuadSurface` | No I/O methods, no lazy loading, no virtuals |
| `Volume` | `vc4d::Volume` | Cache is injected, not owned |
| `VolumePkg` | `vc4d::VolumePkg` | No static state, clear ownership |
| `Segmentation` | `vc4d::Segmentation` | Surface loaded on demand, not in ctor |
| `CState` | `vc4d::AppState` | Not a god object, no raw pointers |
| `CWindow` | `vc4d::MainWindow` | Delegates to focused components |
| `CTiledVolumeViewer` | `vc4d::VolumeViewer` | QOpenGLWidget instead of QGraphicsView |
| `TieredChunkCache` | `vc4d::TieredCache` | Simpler API, shared_mutex |
| `VCCollection` | *(use std::vector directly)* | No wrapper needed |
| `POI` | `struct { Vec3f pos; Vec3f normal; std::string surface_id; }` | Value type |

## Dependency Elimination

| vc3d dependency | vc4d replacement | Why |
|----------------|-----------------|-----|
| OpenCV | `Grid<T>`, `Vec3f`, Qt QImage | cv::Mat used as generic 2D array; cv::Vec3f as 3-vector |
| Eigen | `Vec3f`, `Vec2f` | Used for basic linear algebra only |
| xtensor | `std::vector<uint8_t>` | Chunk buffers don't need a tensor library |
| z5 | `vc4d::ZarrDataset` | ~200 lines for our subset of zarr |
| Boost | Qt equivalents | program_options → QCommandLineParser |
| CGAL | *(defer)* | Only needed for advanced geometry; add when needed |
| Ceres | *(defer)* | Only needed for surface optimization; add when needed |
| OpenMP | `QtConcurrent`, `std::jthread` | Explicit parallelism, no hidden thread pools |

**Remaining dependencies:** Qt 6, nlohmann/json, blosc2 (for zarr decompression).

## Pattern Changes

### 1. No more sentinel -1

```cpp
// vc3d — checking every access
if (points(r, c)[0] != -1.f) {
    auto p = points(r, c);
    // use p
}

// vc4d — optional-based
if (auto p = grid(r, c)) {
    // use *p
}

// vc4d — iterate only valid cells
for (auto [r, c, p] : grid.valid_points()) {
    // p is always valid
}
```

### 2. No more lazy loading with const_cast

```cpp
// vc3d — const_cast hack for lazy loading
const cv::Mat_<cv::Vec3f>* QuadSurface::rawPointsPtr() const {
    const_cast<QuadSurface*>(this)->ensureLoaded();
    return _points.get();
}

// vc4d — loading is the caller's responsibility
auto surf = vc4d::io::load_surface(path);  // loaded when you ask for it
surf.points();  // always available, no lazy loading
```

### 3. I/O is separate from data types

```cpp
// vc3d — I/O mixed into data type
surface->save(path, uuid);
surface->writeValidMask();
QuadSurface(path);  // constructor does I/O

// vc4d — free functions
auto surf = vc4d::io::load_surface(path);
vc4d::io::save_surface(surf, path);
```

### 4. No more virtual base class

```cpp
// vc3d — virtual Surface with 8 pure virtual methods
class Surface {
    virtual void move(...) = 0;
    virtual bool valid(...) = 0;
    virtual cv::Vec3f loc(...) = 0;
    // ... 5 more
};

// vc4d — concrete QuadSurface, the only surface type that ever existed
class QuadSurface {
    // Direct methods, no vtable overhead
};
```

### 5. Composable operations instead of methods

```cpp
// vc3d — operations as methods
std::unique_ptr<QuadSurface> surface_diff(QuadSurface* a, QuadSurface* b, float tol);

// vc4d — free functions returning values
QuadSurface result = vc4d::surface_diff(a, b, 2.0f);
```

### 6. Cache injection instead of ownership

```cpp
// vc3d — Volume owns its cache
volume->setCacheBudget(8ULL << 30);
volume->tieredCache();  // lazy creation inside Volume

// vc4d — cache is created externally and injected
auto cache = std::make_shared<TieredCache>(config);
volume->set_cache(cache);  // multiple volumes can share a cache
```

## File Format Compatibility

vc4d reads the same file formats as vc3d:

- **volpkg directories** — `config.json`, `volumes/`, `paths/`
- **zarr arrays** — `.zarray` metadata, chunked data files
- **tifxyz** — 3-channel float TIFF point grids
- **OBJ meshes** — Wavefront format
- **meta.json** — surface/volume metadata

No conversion tools are needed. Open your existing volpkg in vc4d directly.

## Build System

```bash
cd vc4d
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DVC4D_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

**Required:** CMake 3.30+, C++26 compiler (GCC 15+, Clang 19+), Qt 6.7+
**Optional:** nlohmann/json (fetched automatically)
