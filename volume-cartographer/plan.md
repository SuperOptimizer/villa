# Scale-Space Streaming Tile Renderer

## Context

The current `CVolumeViewer` renders the entire visible viewport as a single monolithic `QGraphicsPixmapItem`. This means:
- Any pan/zoom/scroll triggers a full re-render of the entire visible area
- The render blocks the main thread (synchronous `render_area()` → `readInterpolated3D()`)
- No progressive loading — the user sees nothing until the entire render completes
- Scale-space transitions are all-or-nothing (no multi-resolution display)
- High-latency IO (web/network zarr) freezes the entire UI

The goal is a new generic `ScaleSpaceTileViewer` widget that decomposes the viewport into independent tiles rendered asynchronously at multiple pyramid levels, keeping the UI responsive under all conditions.

## Architecture Overview

```
ScaleSpaceTileViewer (QWidget, owns scene + view)
  ├── TileScheduler (QObject on main thread, orchestrates everything)
  │     ├── TileMap: unordered_map<TileKey, TileData>
  │     ├── IO thread pool (2-4 std::threads, blocking cache->get())
  │     └── Render thread (1 QThread with OMP parallelism)
  ├── QGraphicsScene (tiles as QGraphicsPixmapItems, z-ordered by level)
  └── CVolumeViewerView (reused, handles input events)
```

## Key Design Decisions

### Tile Coordinate System
- `TileKey{level, row, col}` — level is pyramid index, row/col are tile grid indices
- Tiles rendered at a fixed size (configurable, default 256×256 pixels)
- Tiles placed at fixed scene coordinates — QGraphicsView scrolls by moving its viewport, not by moving items
- At pyramid level L, a tile covers `tileSize * 2^L` scene-space pixels of level 0

### Zoom Model (Critical)
Decompose `_scale` into two components:
- `tileLevel`: which pyramid level tiles are rendered at (integer)
- `viewZoom`: QGraphicsView transform applied on top (continuous)
- `viewZoom = _scale / pow(2, -tileLevel)`

This means: tiles are only re-rendered when crossing a pyramid level boundary. Between boundaries, QGraphicsView's transform provides instant smooth zoom (tiles look slightly blurry or crisp depending on direction). This is the standard map-tile approach.

### Threading
- **Main thread**: QGraphicsScene item management, user input, tile visibility computation
- **Render thread** (1 QThread): `Surface::gen()` + `readInterpolated3D()` per tile, with OMP parallelism internally. One render thread avoids OMP thread contention.
- **IO pool** (2-4 threads): `cache->get()` calls that may block on disk/network. Separate from render so IO latency doesn't block rendering of cached tiles.
- Communication: Qt signals/slots with `Qt::QueuedConnection` (automatic thread-safe dispatch to main thread)

### Progressive Multi-Level Display
- When viewport changes, request tiles at target level AND coarser levels
- Coarse tiles (cheaper, faster) appear first as background
- Fine tiles overlay on top as they render (higher QGraphicsScene z-value)
- Z-order: `level N` (coarsest) at z=0, `level 0` (finest) at z=N, overlays at z=N+10

### Partial Rendering (Missing Chunks)
- Render thread uses `getIfCached()` (non-blocking) instead of `get()`
- Pixels where chunks are missing → alpha=0 (transparent)
- Coarser-level tiles provide visual fallback underneath
- Missing chunk keys tracked → dispatched to IO pool → tile re-rendered when chunks arrive
- New `readInterpolated3DPartial()` function in Slicing.hpp

## New Files

| File | Purpose |
|------|---------|
| `apps/VC3D/ScaleSpaceTileViewer.hpp/.cpp` | Main widget: scene setup, zoom/scroll handling, overlay support |
| `apps/VC3D/TileScheduler.hpp/.cpp` | Tile state machine, priority scheduling, thread dispatch |
| `apps/VC3D/TileTypes.hpp` | `TileKey`, `TileData`, `TileState` enum, `TileRenderParams` |
| `apps/VC3D/TileRenderWorker.hpp/.cpp` | QObject on render thread: gen() + sample per tile |
| `apps/VC3D/TileIOWorker.hpp/.cpp` | QObject on IO threads: chunk prefetch |
| `core/include/vc/core/io/ChunkSource.hpp` | Abstract chunk IO interface + LocalSource + HTTPSource |
| `core/src/ChunkSourceHTTP.cpp` | libcurl-based HTTP chunk fetcher |

## Modified Files

| File | Change |
|------|--------|
| `core/include/vc/core/util/Slicing.hpp` | Add `readInterpolated3DPartial()` declaration |
| `core/src/Slicing.cpp` | Implement partial read (uses `getIfCached`, outputs alpha mask) |
| `core/include/vc/core/util/ChunkCache.hpp` | Accept `ChunkSource*` instead of direct z5 IO |
| `core/src/ChunkCache.cpp` | `loadChunk()` delegates to `ChunkSource` |
| `core/include/vc/core/types/Volume.hpp` | Support URL paths, create appropriate ChunkSource |
| `core/src/Volume.cpp` | Parse URL vs filesystem path, construct HTTPSource or LocalSource |
| `core/CMakeLists.txt` | Link libcurl |
| `cmake/VCFindDependencies.cmake` | Find libcurl |
| `apps/VC3D/CMakeLists.txt` | Add new source files |
| `apps/VC3D/ViewerManager.hpp/.cpp` | Add `createTileViewer()` factory method |

## Tile State Machine

```
Needed → Rendering → Ready → Displayed → Stale → Rendering → ...
                                              ↑
  (settings change: z_off, composite, etc.)───┘
```

Each tile tracks: key, state, QGraphicsPixmapItem*, rendered QImage, list of missing chunk keys, completeness ratio.

Tiles outside viewport + 1-tile margin are disposed (scene item removed, data freed).

## Data Flow: User Pans

1. `CVolumeViewerView::scrollContentsBy()` → `ScaleSpaceTileViewer::onScrolled()`
2. Compute new visible scene rect from `fGraphicsView->mapToScene(viewport->geometry())`
3. `TileScheduler::updateViewport(visibleRect, tileLevel)`
4. Compute required TileKeys for target level + coarse levels
5. Existing tiles in map: no action (preserved on 2D scroll!)
6. New tile keys: create TileData, dispatch to render thread
7. Old tiles outside margin: dispose
8. Main thread returns immediately
9. Render thread completes tile → signals main thread → pixmap placed in scene

## Phased Implementation

### Phase 1: Single-Level Tile Foundation
- `ScaleSpaceTileViewer` with QGraphicsScene, `CVolumeViewerView` (reused)
- `TileScheduler` with tile map, visibility computation
- **Synchronous** rendering (blocks main thread per tile, like current code)
- Both PlaneSurface and QuadSurface support
- Tile creation/disposal on scroll/pan — existing tiles preserved
- Coordinate transform: `Surface::gen()` called per tile with tile-specific offset
- No composite, no overlays, no async

**Validates**: tiling grid, coordinate mapping, scroll preservation, tile lifecycle

### Phase 2: Async Rendering
- `TileRenderWorker` on dedicated QThread
- Render tiles asynchronously; main thread stays responsive
- Placeholder tiles (gray or transparent) while rendering
- Priority ordering: center tiles first, then outward

**Validates**: thread safety of Surface::gen() + ChunkCache, non-blocking UX

### Phase 3: Multi-Level Progressive Loading
- Request tiles at multiple pyramid levels simultaneously
- Z-order management (coarse under fine)
- QGraphicsView transform for smooth intra-level zoom
- Level transitions: keep coarse tiles until fine tiles cover viewport

**Validates**: scale-space rendering, progressive loading UX

### Phase 4: IO Separation + Partial Rendering
- IO thread pool (2-4 threads) for `cache->get()` calls
- New `readInterpolated3DPartial()` uses `getIfCached()` only
- Alpha mask output for missing-chunk pixels
- Track missing chunks → dispatch IO → re-render tile on completion
- "Required chunks" set for cache eviction hints

**Validates**: non-blocking IO, partial rendering, chunk arrival re-render

### Phase 5: Composite + Overlays + Full Integration
- `readCompositeFast()` per tile (composite rendering mode)
- Colormap, window/level, ISO cutoff per tile
- Overlay system: `setOverlayGroup()`/`clearOverlayGroup()` matching CVolumeViewer
- Cursor, intersection lines, point collection overlays
- ViewerManager integration (`createTileViewer()`)
- Debug HUD: tile count, cache stats, pending renders

**Validates**: feature parity with CVolumeViewer

## Critical Implementation Details

### Surface::gen() Per Tile
For **PlaneSurface**: `gen(coords, normals, tileSize, {0,0,0}, scale, {scene_x, scene_y, z_off})`
- `scene_x`, `scene_y` are the tile's top-left corner in scene coordinates

For **QuadSurface**: `gen(coords, normals, tileSize, ptr, scale, {offset_x, offset_y, z_off})`
- `ptr` is the surface center pointer
- `offset_x/y` = tile's position relative to ptr, in surface-space units
- `offset = (tile_scene_pos - vis_center) / scale + {-tileSize/2, -tileSize/2, z_off}`

### Thread Safety
- `Surface::gen()`: QuadSurface uses `_loadMutex` for lazy load; after `ensureLoaded()`, raw points are immutable. Concurrent `gen()` calls with separate output buffers are safe.
- `ChunkCache::get()`: Thread-safe by design (shared_mutex + per-key locks). Multiple IO threads can call concurrently.
- `ChunkCache::getIfCached()`: Read-only, shared lock. Safe from any thread.
- QGraphicsScene: All item add/remove/update on main thread only (enforced by signal/slot pattern).

### QGraphicsScene Item Count
- 1920×1080 viewport / 256×256 tiles = 8×5 = 40 tiles per level
- 3 visible levels × 40 = 120 tiles + margin ≈ 200 items
- Well within QGraphicsScene's efficient range

### OMP Contention
- Single render thread with OMP avoids thread oversubscription
- IO threads do no CPU work (just blocking disk/net reads)
- Main thread only does lightweight scene manipulation

## Verification Plan

### Phase 1 Testing
1. Create a test harness that instantiates `ScaleSpaceTileViewer` with a plane surface and a test volume
2. Verify tiles appear correctly: zoom to known location, screenshot, compare against current CVolumeViewer output
3. Pan: verify existing tiles persist (don't flash/disappear), new edge tiles appear
4. QuadSurface: load a segmentation, verify tile stitching (no gaps, no overlap artifacts)

### Phase 2 Testing
1. Pan rapidly — UI should remain responsive (not freeze during tile renders)
2. Add timing: log time from tile request to display
3. Stress test: resize window rapidly, switch surfaces

### Phase 3-5 Testing
1. Zoom out to coarsest level, zoom in — verify progressive fill-in
2. Simulate slow IO (sleep in cache->get): verify coarse tiles appear, fine tiles fill in
3. Composite mode: compare output against CVolumeViewer composite at same viewport
4. Overlay placement: intersection lines, cursor, points should align identically

## Key Files to Reference

- `apps/VC3D/CVolumeViewerRender.cpp` — Current render pipeline to replicate per-tile
- `apps/VC3D/CVolumeViewer.cpp:324-348` — `recalcScales()` zoom-to-level mapping
- `core/src/Slicing.cpp:220-438` — `readVolumeImpl()` prefetch + sample pattern
- `core/include/vc/core/util/ChunkCache.hpp` — Cache API (get/getIfCached/prefetch)
- `core/include/vc/core/util/Surface.hpp` — `gen()` interface
- `apps/VC3D/ViewerManager.cpp:88` — Viewer creation pattern to replicate
- `apps/VC3D/CVolumeViewerOverlay.cpp:183-228` — Overlay group management to replicate
