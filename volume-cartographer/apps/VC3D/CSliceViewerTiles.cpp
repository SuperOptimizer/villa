// CSliceViewerTiles.cpp -- tile management, render worker thread, tile rendering
//
// This file implements the non-blocking tile rendering pipeline:
//   1. updateVisibleTiles / enqueueTilesForViewport  -- main thread
//   2. renderWorkerLoop / renderTile                 -- background thread
//   3. consumeReadyTiles                             -- main thread (timer callback)
//   4. removeOffscreenTiles / invalidateAllTiles     -- main thread

#include "CSliceViewer.hpp"
#include "VolumeViewerCmaps.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QPixmap>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

// =========================================================================
// Tile request construction  (main thread)
// =========================================================================

CSliceViewer::TileRenderRequest CSliceViewer::makeTileRequest(
    const TileKey& key, int dsIdx, float dsScaleVal)
{
    TileRenderRequest req{};
    req.key = key;
    req.sceneRect = {key.tx * TILE_SIZE, key.ty * TILE_SIZE, TILE_SIZE, TILE_SIZE};
    req.scale = _scale;
    req.dsIdx = dsIdx;
    req.dsScale = dsScaleVal;
    req.z_off = _z_off;
    req.generation = _viewGeneration.load(std::memory_order_relaxed);

    auto surf = _surf_weak.lock();
    req.surfRef = surf;  // prevent surface destruction while rendering

    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    auto* quad = dynamic_cast<QuadSurface*>(surf.get());

    req.isPlane = (plane != nullptr);

    if (plane) {
        req.planeOrigin = plane->origin();
        req.planeNormal = plane->normal(plane->pointer(), {});
        req.planeVx = plane->basisX();
        req.planeVy = plane->basisY();
    }

    if (quad) {
        // Pre-compute the surface pointer for this tile's center
        float cx = req.sceneRect.x + TILE_SIZE / 2.0f;
        float cy = req.sceneRect.y + TILE_SIZE / 2.0f;
        cv::Vec3f ptr = quad->pointer();
        quad->move(ptr, {cx / _scale, cy / _scale, 0});
        req.quadPtr = ptr;
    }

    // Determine rendering mode
    bool isSegView = (_surf_name == "segmentation");
    req.useComposite = (isSegView && _composite_enabled &&
                        (_composite_layers_front > 0 || _composite_layers_behind > 0));
    req.usePlaneComposite = (plane != nullptr && _plane_composite_enabled &&
                             (_plane_composite_layers_front > 0 || _plane_composite_layers_behind > 0));

    req.compositeParams = currentCompositeParams();
    req.compositeLayersFront = isSegView ? _composite_layers_front : _plane_composite_layers_front;
    req.compositeLayersBehind = isSegView ? _composite_layers_behind : _plane_composite_layers_behind;
    req.compositeReverse = _composite_reverse_direction;

    // Window/level
    req.windowLow = _baseWindowLow;
    req.windowHigh = _baseWindowHigh;
    req.stretchValues = _stretchValues;
    req.colormapId = _baseColormapId;
    req.isoCutoff = _iso_cutoff;
    req.useFastInterpolation = _useFastInterpolation;

    return req;
}

// =========================================================================
// Enqueue tiles for a viewport rectangle  (main thread)
// =========================================================================

void CSliceViewer::enqueueTilesForViewport(const QRectF& bbox, int dsIdx, float dsScaleVal)
{
    // Compute tile grid bounds that cover the viewport
    int tx0 = static_cast<int>(std::floor(bbox.left()   / TILE_SIZE));
    int ty0 = static_cast<int>(std::floor(bbox.top()    / TILE_SIZE));
    int tx1 = static_cast<int>(std::floor(bbox.right()  / TILE_SIZE));
    int ty1 = static_cast<int>(std::floor(bbox.bottom() / TILE_SIZE));

    uint64_t gen = _viewGeneration.load(std::memory_order_relaxed);

    std::vector<TileRenderRequest> newRequests;
    newRequests.reserve(static_cast<size_t>((tx1 - tx0 + 1) * (ty1 - ty0 + 1)));

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            TileKey key{tx, ty};
            auto it = _tiles.find(key);

            // Skip if this tile is already rendered at this generation and
            // at an equal-or-better (lower or equal dsIdx) pyramid level
            if (it != _tiles.end() && it->second.generation == gen &&
                it->second.renderedDsIdx <= dsIdx) {
                continue;
            }

            newRequests.push_back(makeTileRequest(key, dsIdx, dsScaleVal));
        }
    }

    if (newRequests.empty()) return;

    // Push into the render queue
    {
        std::lock_guard<std::mutex> lk(_renderQueueMutex);
        for (auto& r : newRequests)
            _renderQueue.push_back(std::move(r));
    }
    _renderQueueCV.notify_one();
}

// =========================================================================
// Remove tiles that are far outside the visible area  (main thread)
// =========================================================================

void CSliceViewer::removeOffscreenTiles(const QRectF& visibleBBox)
{
    // Keep a margin of 2 tiles around the visible area
    const float margin = TILE_SIZE * 2.0f;
    QRectF keepRect = visibleBBox.adjusted(-margin, -margin, margin, margin);

    std::vector<TileKey> toRemove;
    for (auto& [key, tile] : _tiles) {
        QRectF tileRect(key.tx * TILE_SIZE, key.ty * TILE_SIZE, TILE_SIZE, TILE_SIZE);
        if (!keepRect.intersects(tileRect))
            toRemove.push_back(key);
    }

    for (auto& key : toRemove) {
        auto it = _tiles.find(key);
        if (it != _tiles.end()) {
            if (it->second.item) {
                fScene->removeItem(it->second.item);
                delete it->second.item;
            }
            _tiles.erase(it);
        }
    }
}

// =========================================================================
// Invalidate all tiles (zoom/surface change)  (main thread)
// =========================================================================

void CSliceViewer::invalidateAllTiles()
{
    // Bump generation so in-flight renders become stale
    _viewGeneration.fetch_add(1, std::memory_order_relaxed);

    // Clear the render queue of stale requests
    {
        std::lock_guard<std::mutex> lk(_renderQueueMutex);
        _renderQueue.clear();
    }

    // Remove all tile pixmaps from the scene
    for (auto& [key, tile] : _tiles) {
        if (tile.item) {
            fScene->removeItem(tile.item);
            delete tile.item;
        }
    }
    _tiles.clear();
}

// =========================================================================
// Consume ready tiles (main thread, called by timer)
// =========================================================================

void CSliceViewer::consumeReadyTiles()
{
    std::vector<TileResult> results;
    {
        std::lock_guard<std::mutex> lk(_resultMutex);
        if (_readyTiles.empty()) return;
        results.swap(_readyTiles);
    }

    uint64_t gen = _viewGeneration.load(std::memory_order_relaxed);

    for (auto& r : results) {
        // Discard results from stale generations
        if (r.generation != gen)
            continue;

        if (r.image.isNull())
            continue;

        auto& tile = _tiles[r.key];

        // Only update if this result is equal-or-better quality
        if (tile.generation == gen && tile.renderedDsIdx <= r.dsIdx)
            continue;

        QPixmap pixmap = QPixmap::fromImage(r.image);

        if (!tile.item) {
            tile.item = fScene->addPixmap(pixmap);
            tile.item->setZValue(-1);  // behind overlays
        } else {
            tile.item->setPixmap(pixmap);
        }

        tile.item->setOffset(r.key.tx * TILE_SIZE, r.key.ty * TILE_SIZE);

        // If this tile was rendered at a coarser level, scale the pixmap up
        // to cover the full tile area
        if (r.dsIdx > 0) {
            // The tile was rendered at a lower resolution; the image is
            // TILE_SIZE pixels but represents the correct scene area, so
            // no scaling needed -- the gen() call already produced the
            // correct number of pixels for the tile's scene rectangle.
        }

        tile.renderedDsIdx = r.dsIdx;
        tile.generation = gen;
    }
}

// =========================================================================
// Render worker loop  (background thread)
// =========================================================================

void CSliceViewer::renderWorkerLoop()
{
    while (true) {
        TileRenderRequest req;

        // Wait for work
        {
            std::unique_lock<std::mutex> lk(_renderQueueMutex);
            _renderQueueCV.wait(lk, [this] {
                return _shutdown.load(std::memory_order_acquire) || !_renderQueue.empty();
            });

            if (_shutdown.load(std::memory_order_acquire))
                return;

            req = std::move(_renderQueue.front());
            _renderQueue.pop_front();
        }

        // Check if request is stale
        if (req.generation != _viewGeneration.load(std::memory_order_relaxed))
            continue;

        // Render the tile
        QImage img = renderTile(req);

        // Post result
        {
            std::lock_guard<std::mutex> lk(_resultMutex);
            _readyTiles.push_back({req.key, std::move(img), req.dsIdx, req.generation});
        }
    }
}

// =========================================================================
// Render a single tile  (background thread)
// =========================================================================

QImage CSliceViewer::renderTile(const TileRenderRequest& req)
{
    if (!req.surfRef || !volume)
        return {};

    z5::Dataset* ds = volume->zarrDataset(req.dsIdx);
    if (!ds || !cache)
        return {};

    const int w = req.sceneRect.width;
    const int h = req.sceneRect.height;
    const cv::Size tileSize(w, h);

    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    cv::Mat_<uint8_t> gray;

    // ---- Generate coordinates ----
    if (req.isPlane) {
        // PlaneSurface: gen() is a pure function, safe to call from any thread
        req.surfRef->gen(&coords, nullptr, tileSize,
                         cv::Vec3f(0, 0, 0), req.scale,
                         {static_cast<float>(req.sceneRect.x),
                          static_cast<float>(req.sceneRect.y),
                          req.z_off});
    } else {
        // QuadSurface: use pre-computed ptr
        req.surfRef->gen(&coords, &normals, tileSize,
                         req.quadPtr, req.scale,
                         {-w / 2.0f, -h / 2.0f, req.z_off});
    }

    if (coords.empty())
        return {};

    // ---- Sample volume data ----
    if (req.useComposite) {
        // Composite rendering (segmentation view)
        int zStart = req.compositeReverse ? -req.compositeLayersBehind : -req.compositeLayersFront;
        int zEnd   = req.compositeReverse ?  req.compositeLayersFront  :  req.compositeLayersBehind;

        if (!normals.empty()) {
            readCompositeFast(gray, ds,
                              coords * req.dsScale,
                              normals,
                              req.dsScale, zStart, zEnd,
                              req.compositeParams, *cache);
        } else {
            // Plane composite with constant normal
            readCompositeFastConstantNormal(gray, ds,
                                            coords * req.dsScale,
                                            req.planeNormal,
                                            req.dsScale, zStart, zEnd,
                                            req.compositeParams, *cache);
        }
    } else if (req.usePlaneComposite) {
        // Plane composite rendering
        int zStart = req.compositeReverse ? -req.compositeLayersBehind : -req.compositeLayersFront;
        int zEnd   = req.compositeReverse ?  req.compositeLayersFront  :  req.compositeLayersBehind;

        readCompositeFastConstantNormal(gray, ds,
                                        coords * req.dsScale,
                                        req.planeNormal,
                                        req.dsScale, zStart, zEnd,
                                        req.compositeParams, *cache);
    } else {
        // Single-slice rendering
        readInterpolated3D(gray, ds, coords * req.dsScale, cache, req.useFastInterpolation);
    }

    if (gray.empty())
        return {};

    // ---- Post-processing ----

    // ISO cutoff
    if (req.isoCutoff > 0) {
        cv::threshold(gray, gray, req.isoCutoff - 1, 0, cv::THRESH_TOZERO);
    }

    // Window/level or stretch
    cv::Mat processed;
    if (req.stretchValues) {
        double minVal, maxVal;
        cv::minMaxLoc(gray, &minVal, &maxVal);
        double range = std::max(1.0, maxVal - minVal);
        cv::Mat tmp;
        gray.convertTo(tmp, CV_32F);
        tmp -= static_cast<float>(minVal);
        tmp /= static_cast<float>(range);
        tmp.convertTo(processed, CV_8U, 255.0f);
    } else {
        float wLow = std::clamp(req.windowLow, 0.0f, 255.0f);
        float wHigh = std::clamp(req.windowHigh, wLow + 1.0f, 255.0f);
        float span = std::max(1.0f, wHigh - wLow);
        cv::Mat tmp;
        gray.convertTo(tmp, CV_32F);
        tmp -= wLow;
        tmp /= span;
        cv::max(tmp, 0.0f, tmp);
        cv::min(tmp, 1.0f, tmp);
        tmp.convertTo(processed, CV_8U, 255.0f);
    }

    // Colormap
    cv::Mat color;
    if (!req.colormapId.empty()) {
        const auto& spec = volume_viewer_cmaps::resolve(req.colormapId);
        color = volume_viewer_cmaps::makeColors(processed, spec);
    } else {
        if (processed.channels() == 1)
            cv::cvtColor(processed, color, cv::COLOR_GRAY2BGR);
        else
            color = processed.clone();
    }

    if (color.empty())
        return {};

    // ---- Convert to QImage ----
    cv::Mat rgb;
    cv::cvtColor(color, rgb, cv::COLOR_BGR2RGB);
    QImage img(static_cast<const uchar*>(rgb.data),
               rgb.cols, rgb.rows, static_cast<int>(rgb.step),
               QImage::Format_RGB888);
    // Deep copy so the cv::Mat can be freed
    return img.copy();
}
