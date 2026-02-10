# VC3D Streaming Tile-Based Renderer

Ordered list of APIs, data structures, and components to implement.

---

## 1. Type Conventions

Use xtensor + OpenCV types directly throughout the new API:

- **Array data** — `xt::xtensor<T, N>` for all N-dimensional data
- **Small vectors** — `cv::Vec` for individual points, normals, offsets
- **Coordinate maps** — `cv::Mat_<cv::Vec3f>` for tifxyz

---

## 2. Samplers

Lightweight structs passed to Volume read methods. Default is `Trilinear`.

```cpp
namespace vc {
struct Nearest {};
struct Trilinear {};
struct Tricubic {};
struct Composite {
    cv::Mat_<cv::Vec3f> normals;      // per-pixel normals (or empty → use constantNormal)
    cv::Vec3f constantNormal;
    float zStep;
    int zStart, zEnd;
    CompositeParams params;
};
}
```

---

## 3. ReadMode

Controls blocking behavior and resolution selection.

- **`Exact{scale}`** — blocks until the read is done at the requested resolution.
- **`BestEffort{scale}`** — returns immediately with the best data currently cached. If the requested resolution isn't available, returns upsampled data from the best cached pyramid level. The coarsest level is always hot, so this never fails.

**Scale semantics:** output resolution relative to native voxels. `1.0` = one output pixel per native voxel, `0.5` = half resolution, `2.0` = superresolution. Volume picks the coarsest pyramid level that meets the requested scale if the caller doesn't specify a level.

**Scale selection logic:**

- `Exact` — Volume blocks until the needed chunks are loaded and decompressed, then returns data at the requested resolution.
- `BestEffort` — Volume walks the pyramid from coarsest to finest, looking for the best level whose chunks are already hot in cache. Returns data from that level, upsampled to the output dimensions if necessary.

---

## 4. Volume Read API

Callers specify what region and dimensions they want and how to handle resolution. Chunks and shards are invisible.

### `read0d` — single scalar at one 3D point

```cpp
template<typename Sampler = Trilinear>
uint8_t Volume::read0d(cv::Vec3f point, Sampler s = {}, ReadMode mode = Exact{}, float scale = 1.0f);
```

### `read1d` — sample along an ordered list of 3D points

```cpp
template<typename Sampler = Trilinear>
xt::xtensor<uint8_t, 1> Volume::read1d(
    const std::vector<cv::Vec3f>& points, Sampler s = {}, ReadMode mode = Exact{}, float scale = 1.0f);
```

### `read2d` (planar) — sample a planar region

Replaces `readInterpolated3D`, `readCompositeFast`, `readMultiSlice`.

```cpp
template<typename Mode = Exact, typename Sampler = Trilinear>
xt::xtensor<uint8_t, 2> Volume::read2d(
    cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
    Mode mode = {}, Sampler s = {});
```

### `read2d` (coordinate map) — sample at explicit coordinates (curved surfaces / QuadSurface)

```cpp
template<typename Mode = Exact, typename Sampler = Trilinear>
auto Volume::read2d(
    const cv::Mat_<cv::Vec3f>& coords, Mode mode = {}, Sampler s = {});
```

### `read3d` (oriented box) — sample an oriented 3D box

Replaces `readArea3D`. Any orientation.

```cpp
template<typename Mode = Exact>
auto Volume::read3d(
    cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV, cv::Vec3f axisW,
    Mode mode = {});
```

### `read3d` (multi-depth from coordinate map) — sample a surface at multiple depths along normals

Replaces `readMultiSlice`. Returns a 3D tensor `{numLayers, rows, cols}`.

```cpp
template<typename Mode = Exact, typename Sampler = Trilinear>
auto Volume::read3d(
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets,
    Mode mode = {}, Sampler s = {});
```

- `coords` is the base surface.
- `stepDirs` is per-pixel normal/depth direction.
- `offsets` are the depth values to sample at (e.g. `{-5, -4, ..., 0, ..., 4, 5}`).
- Each layer `i` samples at `coords + offsets[i] * stepDirs`.

### Usage examples

```cpp
// --- Exact mode (default) — blocking, guaranteed resolution ---

// Planar slice at native resolution (default: Exact{1.0})
auto tile = volume->read2d(origin, u, v);

// Explicit half-res for a thumbnail or export
auto half = volume->read2d(origin, u, v, vc::Exact{0.5f});

// Curved surface coordinates (QuadSurface)
auto tile = volume->read2d(coords);

// Nearest neighbor sampling
auto tile = volume->read2d(origin, u, v, vc::Exact{}, vc::Nearest{});

// Composite rendering
auto tile = volume->read2d(origin, u, v, vc::Exact{},
    vc::Composite{normals, {}, 1.0f, -5, 5, params});

// Single point probe
uint8_t val = volume->read0d({x, y, z});

// 3D oriented box at half resolution
auto block = volume->read3d(origin, u, v, w, vc::Exact{0.5f});

// Multi-depth from a surface — 11 layers at offsets -5..+5
std::vector<float> offsets;
for (int i = -5; i <= 5; i++) offsets.push_back(i);
auto stack = volume->read3d(coords, normals, offsets);
// stack shape: {11, rows, cols}

// --- BestEffort mode — non-blocking, progressive rendering ---

// Get the best currently-cached data for this region (targeting native res)
auto tile = volume->read2d(origin, u, v, vc::BestEffort{1.0f});

// Kick off background fetch, then immediately get what's available
volume->prefetch(origin, u, v, 1.0f);
auto tile = volume->read2d(origin, u, v, vc::BestEffort{1.0f});
// First call: might be upsampled from coarsest mip
// After chunks arrive and we re-render: progressively sharper
```

---

## 5. ChunkSource

Single class handling both local and remote volumes. Configured with a source path/URI, a local cache path, and a storage type:

- **`StorageType::Local`** — source path is on fast local disk. Chunks are read directly, no caching layer needed.
- **`StorageType::Network`** — source path is on slow storage (NFS, s3fs) or a URL. On cache miss, the chunk is copied/fetched from the source and written to the local cache. Subsequent reads come from the cache.

The source path can be either a filesystem path or a URL. Volume can auto-detect local vs. remote storage.

For `StorageType::Network` with an HTTP(S) URI, `ensureLocal()` does HTTP GET for `{sourcePath}/{scaleLevel}/{iz}.{iy}.{ix}`. For `StorageType::Network` with a filesystem path (NFS/s3fs), `ensureLocal()` copies the chunk file from source to cache.

---

## 6. ChunkCache

Template `ChunkCache<T>` with three internal tiers:

- **Hot:** Decompressed chunks in RAM as `shared_ptr<xt::xtensor<T, 3>>`.
- **Warm:** Compressed `vector<char>` in RAM.
- **Disk:** On-disk chunk storage.

Chunks are written to a local zarr mirror directory. User can enable access-time-based LRU deletion for the mirror.

```cpp
shared_ptr<xt::xtensor<T, 3>> ChunkCache<T>::get(int scaleLevel, int iz, int iy, int ix);
```

`get()` checks hot → warm → disk → remote source, promoting up the tiers as it goes.

---

## 7. Volume::prefetch

Schedule chunk fetches for a region — returns immediately. `ChunkSource::ensureLocal()` runs on background IO threads (2-4 threads). Priority queue with dedup.

```cpp
void Volume::prefetch(cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                      float targetScale = 1.0f);
```

`prefetch` figures out which chunks are needed for the region at `targetScale`, and schedules background IO for any that aren't already cached. `BestEffort` reads don't require a preceding `prefetch` — they always return data (at least from the coarsest mip). But `prefetch` drives progressive refinement: it tells the IO system what to fetch next so that future `BestEffort` reads return higher-resolution data.

---

## 8. Surface::readParamsForTile

Surface classes gain a method to produce `Volume::read2d` parameters for a given screen-space tile rect. This keeps surface geometry knowledge inside `Surface`. The tile renderer just calls this and forwards to `Volume`.

```cpp
struct TileReadParams {
    // Planar: origin, axisU, axisV populated
    // Curved: coords sub-mat populated
    // Surface type determines which variant is used
    std::variant<PlanarParams, CoordMapParams> params;
    float scale;
};
TileReadParams Surface::readParamsForTile(QRectF tileSceneRect, float scale);
```

---

## 9. TileGrid

Screen-space tiling. Pure geometry — no volume or surface knowledge.

```cpp
TileGrid(int tilePixelSize = 256);
```

Given a viewport rect, produces `TileKey{col, row}` + scene rect for each visible tile.

---

## 10. TileScene

Map of `TileKey → QGraphicsPixmapItem*`. Surface-agnostic — it only knows about pixel tiles and where to place them.

```cpp
void TileScene::setTile(TileKey key, QImage image, QRectF sceneRect, uint64_t generation);
void TileScene::gc(QRectF viewport, uint64_t generation, float margin);
```

`gc()` removes off-screen tiles from older generations.

---

## 11. Immutable Surface References

Surfaces are held as `shared_ptr<const Surface>`. When the surface changes, a new object is created and the `shared_ptr` is swapped (atomic). Render threads grab a `shared_ptr` at the start of their work — they hold a valid reference for the duration, no deep copy needed.

---

## 12. RenderPool

Thread pool with up to `ncpu` worker threads. Concurrent task queue. One thread per task.

**Task:** `{TileKey, readParams (from Surface), scale, epoch, renderParams (window/level, colormap, etc.)}`.

**Worker loop:**

1. Pull task from queue
2. Check epoch — if stale, discard and pull next
3. Call `volume->read(readParams, scale, sampler)`
4. Apply post-processing: window/level, colormap, overlay blend → `QImage`
5. Push to completed queue (main thread polls or gets signaled)

**Main thread receives completed tiles:**

1. Check epoch — if stale (viewport changed since this was submitted), discard
2. Place tile in TileScene

---

## 13. TileRenderController

Connects the pieces. For each visible tile:

1. `TileGrid` gives the screen-space tile rect
2. `Surface` gives the read params for that rect
3. `Volume::read2d` gives pixel data
4. Post-process: window/level, colormap, overlay blend → `QImage`
5. `TileScene` places the tile

The controller knows about Volume and Surface, but TileScene and the Qt display layer do not.

### Progressive rendering

On viewport change, render tiles from low scale to high scale: 32x → 16x → 8x → 4x → 2x → 1x downscaling. Each pass replaces previous tile images. Volume picks the best pyramid level automatically, so low-scale renders are fast.

### Progressive rendering flow (off-main-thread)

1. Viewport changes → main thread increments epoch
2. Main thread grabs `shared_ptr<const Surface>`, computes visible tiles via `TileGrid`
3. Submits all tiles to `RenderPool` at low scale with current epoch
4. Returns immediately — GUI stays responsive
5. As low-scale tiles complete, they are placed in the scene (user sees low-res immediately)
6. If viewport changes mid-render: epoch increments, in-flight tasks are discarded when they complete (epoch check), new tasks submitted for the new viewport

### Interactive rendering flow (non-blocking IO)

1. Viewport changes → `volume->prefetch(region, 1.0f)` kicks off background fetches for native-res chunks
2. `volume->read2d(region, BestEffort{1.0f})` returns immediately with the best cached data
3. Tiles are displayed with current data
4. When chunks arrive (notified via callback/signal from ChunkSource), affected tiles re-render — `volume->read2d(region, BestEffort{1.0f})` now returns better data
5. Eventually native-res data is fully cached, no more refinement needed

No specific chunk pinning is needed — visible chunks are most-recently-accessed, so LRU naturally protects them.

---

## 14. Modify CVolumeViewer

Remove `fBaseImageItem`, `curr_img_area`, cached normals, cache member. Add `TileScene` + `TileRenderController`. `renderVisible()` delegates to controller. Per-tile overlays rendered independently.

Two usage patterns — same Volume, same read methods, different modes:

- **Batch (CLI tools):** Use `volume->read2d(..., Exact{...})` (the default) which blocks until all needed chunks are available.
- **Interactive (VC3D):** Use `volume->prefetch(...)` or `BestEffort` to schedule chunk fetches in the background. Re-render when more data arrives.
