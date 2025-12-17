#include "CVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "VolumeViewerCmaps.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>

#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <omp.h>


#include <QPainter>
#include <optional>

#include <opencv2/imgproc.hpp>

#define COLOR_FOCUS QColor(50, 255, 215)


void CVolumeViewer::renderVisible(bool force)
{
    auto surf = _surf_weak.lock();
    if (surf && _surf_col) {
        auto currentSurface = _surf_col->surface(_surf_name);
        if (!currentSurface) {
            // Surface was cleared (e.g. during volume reload) without a change signal
            // reaching this viewer yet; drop the dangling pointer before rendering.
            _surf_weak.reset();
            surf.reset();
        }
    }

    if (!volume || !volume->zarrDataset() || !surf)
        return;

    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();

    if (!force && QRectF(curr_img_area).contains(bbox))
        return;


    curr_img_area = {static_cast<int>(bbox.left()),static_cast<int>(bbox.top()), static_cast<int>(bbox.width()), static_cast<int>(bbox.height())};

    cv::Mat img = render_area({curr_img_area.x(), curr_img_area.y(), curr_img_area.width(), curr_img_area.height()});

    QImage qimg = Mat2QImage(img);
    if (_overlayImageValid && !_overlayImage.isNull()) {
        qimg = qimg.convertToFormat(QImage::Format_RGBA8888);
        QPainter painter(&qimg);
        painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
        painter.drawImage(0, 0, _overlayImage);
    }

    QPixmap pixmap = QPixmap::fromImage(qimg, fSkipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);

    // Add the QPixmap to the scene as a QGraphicsPixmapItem
    if (!fBaseImageItem)
        fBaseImageItem = fScene->addPixmap(pixmap);
    else
        fBaseImageItem->setPixmap(pixmap);

    if (!_center_marker) {
        _center_marker = fScene->addEllipse({-10,-10,20,20}, QPen(COLOR_FOCUS, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _center_marker->setZValue(11);
    }

    _center_marker->setParentItem(fBaseImageItem);

    fBaseImageItem->setOffset(curr_img_area.topLeft());
}


cv::Mat_<uint8_t> CVolumeViewer::render_composite(const cv::Rect &roi) {
    cv::Mat_<uint8_t> img;

    auto surf = _surf_weak.lock();
    if (!surf)
        return img;

    cv::Vec2f roi_c = {static_cast<float>(roi.x + roi.width / 2), static_cast<float>(roi.y + roi.height / 2)};
    cv::Vec3f ptr = surf->pointer();
    cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
    surf->move(ptr, diff / _scale);
    _ptr = ptr;
    _vis_center = roi_c;

    // Check if we can reuse cached normals
    // Cache key: roi size, scale, ptr position, and surface instance
    // Note: z_off is NOT part of the cache key - normals don't depend on z_off,
    // and we apply the z offset ourselves after retrieving cached values
    bool cacheValid = (!_cachedNormals.empty() &&
                       _cachedNormalsSize == roi.size() &&
                       std::abs(_cachedNormalsScale - _scale) < 1e-6f &&
                       cv::norm(_cachedNormalsPtr - ptr) < 1e-6f &&
                       _cachedNormalsSurf.lock() == surf);

    cv::Mat_<cv::Vec3f> base_coords;
    cv::Mat_<cv::Vec3f> normals;

    if (cacheValid) {
        // Reuse cached coordinates and normals
        // Cached coords are at z_off=0, so apply current z_off if needed
        if (std::abs(_z_off - _cachedNormalsZOff) < 1e-6f) {
            // Same z_off, use cached coords directly
            base_coords = _cachedBaseCoords;
        } else {
            // Different z_off - apply offset along normals using work buffer
            const int h = _cachedBaseCoords.rows;
            const int w = _cachedBaseCoords.cols;

            // Reuse work buffer if size matches, otherwise reallocate
            if (_coordsWorkBuffer.rows != h || _coordsWorkBuffer.cols != w) {
                _coordsWorkBuffer.create(h, w);
            }

            const float z_delta = _z_off - _cachedNormalsZOff;
            #pragma omp parallel for collapse(2)
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    const cv::Vec3f& src = _cachedBaseCoords(j, i);
                    const cv::Vec3f& n = _cachedNormals(j, i);
                    if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                        _coordsWorkBuffer(j, i) = src + n * z_delta;
                    } else {
                        _coordsWorkBuffer(j, i) = src;
                    }
                }
            }
            base_coords = _coordsWorkBuffer;
        }
        normals = _cachedNormals;
    } else {
        // Generate coordinates and normals for base layer
        surf->gen(&base_coords, &normals, roi.size(), ptr, _scale,
                  {static_cast<float>(-roi.width / 2), static_cast<float>(-roi.height / 2), _z_off});

        // Cache for next render - gen() returns freshly allocated data, so we can
        // just copy the cv::Mat header (shallow copy) since we own the data
        _cachedBaseCoords = base_coords;
        _cachedNormals = normals;
        _cachedNormalsSize = roi.size();
        _cachedNormalsScale = _scale;
        _cachedNormalsPtr = ptr;
        _cachedNormalsZOff = _z_off;
        _cachedNormalsSurf = surf;
    }

    // Determine the z range based on front and behind layers
    int z_start = _composite_reverse_direction ? -_composite_layers_behind : -_composite_layers_front;
    int z_end = _composite_reverse_direction ? _composite_layers_front : _composite_layers_behind;

    // Setup compositing parameters
    CompositeParams params;
    params.method = _composite_method;
    params.alphaMin = _composite_alpha_min / 255.0f;
    params.alphaMax = _composite_alpha_max / 255.0f;
    params.alphaOpacity = _composite_material / 255.0f;
    params.alphaCutoff = _composite_alpha_threshold / 10000.0f;
    params.isoCutoff = static_cast<uint8_t>(_iso_cutoff);
    // Scale parameters
    params.gradientScale = _composite_gradient_scale;
    params.stddevScale = _composite_stddev_scale;
    params.laplacianScale = _composite_laplacian_scale;
    params.rangeScale = _composite_range_scale;
    params.gradientSumScale = _composite_gradient_sum_scale;
    params.sobelScale = _composite_sobel_scale;
    params.localContrastScale = _composite_local_contrast_scale;
    params.entropyScale = _composite_entropy_scale;
    params.peakThreshold = _composite_peak_threshold;
    params.peakCountScale = _composite_peak_count_scale;
    params.countThreshold = _composite_count_threshold;
    params.thresholdCountScale = _composite_threshold_count_scale;
    params.percentile = _composite_percentile;
    params.weightedMeanSigma = _composite_weighted_mean_sigma;

    // Always use fast path (nearest neighbor, no mutex, specialized cache)
    readCompositeFast(
        img,
        volume->zarrDataset(_ds_sd_idx),
        base_coords * _ds_scale,
        normals,
        _ds_scale,  // z step per layer (in dataset coordinates)
        z_start, z_end,
        params,
        _fastCompositeCache
    );

    // Apply postprocessing
    if (!img.empty()) {
        // Stretch values to full range
        if (_postStretchValues) {
            double minVal, maxVal;
            cv::minMaxLoc(img, &minVal, &maxVal);
            if (maxVal > minVal) {
                img.convertTo(img, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            }
        }

        // Remove small connected components
        if (_postRemoveSmallComponents && _postMinComponentSize > 1) {
            // Create binary mask of non-zero pixels
            cv::Mat_<uint8_t> binary;
            cv::threshold(img, binary, 0, 255, cv::THRESH_BINARY);

            // Find connected components
            cv::Mat labels, stats, centroids;
            int numComponents = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

            // Create mask of components to keep (those >= min size)
            cv::Mat_<uint8_t> keepMask = cv::Mat_<uint8_t>::zeros(img.size());
            for (int i = 1; i < numComponents; i++) {  // Start from 1 to skip background
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area >= _postMinComponentSize) {
                    keepMask.setTo(255, labels == i);
                }
            }

            // Apply mask to original image
            cv::Mat_<uint8_t> filtered;
            img.copyTo(filtered, keepMask);
            img = filtered;
        }
    }

    return img;
}

cv::Mat_<uint8_t> CVolumeViewer::renderCompositeForSurface(std::shared_ptr<QuadSurface> surface, cv::Size outputSize)
{
    if (!surface || !_composite_enabled || !volume) {
        return cv::Mat_<uint8_t>();
    }

    // Save current state
    float oldScale = _scale;
    cv::Vec2f oldVisCenter = _vis_center;
    auto oldSurf = _surf_weak.lock();
    float oldZOff = _z_off;
    cv::Vec3f oldPtr = _ptr;
    float oldDsScale = _ds_scale;
    int oldDsSdIdx = _ds_sd_idx;

    // Render at 1:1 with the surface's internal grid (raw points size)
    // Use surface's scale so that gen() computes sx = _scale/_scale = 1.0,
    // sampling 1:1 from the raw points grid
    cv::Size rawPointsSize = surface->rawPointsPtr()->size();
    float surfScale = surface->_scale[0];

    std::cout << "[renderCompositeForSurface] outputSize: " << outputSize.width << "x" << outputSize.height
              << ", rawPointsSize: " << rawPointsSize.width << "x" << rawPointsSize.height
              << ", surface->_scale: " << surface->_scale[0] << "x" << surface->_scale[1] << std::endl;

    _surf_weak = surface;
    _scale = surfScale;  // Use surface's scale so gen() samples 1:1 from raw points
    _z_off = 0.0f;

    recalcScales();

    std::cout << "[renderCompositeForSurface] after recalcScales: _scale=" << _scale
              << ", _ds_scale=" << _ds_scale << ", _ds_sd_idx=" << _ds_sd_idx << std::endl;

    _ptr = surface->pointer();
    // Use raw points size for the ROI so we cover the whole surface
    cv::Rect roi(-rawPointsSize.width/2, -rawPointsSize.height/2,
                 rawPointsSize.width, rawPointsSize.height);

    _vis_center = cv::Vec2f(0, 0);

    cv::Mat_<uint8_t> result = render_composite(roi);

    std::cout << "[renderCompositeForSurface] result size: " << result.cols << "x" << result.rows << std::endl;

    // Resize to requested output size if different
    if (result.size() != outputSize) {
        std::cout << "[renderCompositeForSurface] resizing from " << result.cols << "x" << result.rows
                  << " to " << outputSize.width << "x" << outputSize.height << std::endl;
        cv::resize(result, result, outputSize, 0, 0, cv::INTER_LINEAR);
    }

    _surf_weak = oldSurf;
    _scale = oldScale;
    _vis_center = oldVisCenter;
    _z_off = oldZOff;
    _ptr = oldPtr;
    _ds_scale = oldDsScale;
    _ds_sd_idx = oldDsSdIdx;

    return result;
}


cv::Mat CVolumeViewer::render_area(const cv::Rect &roi)
{
    auto surf = _surf_weak.lock();
    if (!surf)
        return cv::Mat();

    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> baseGray;
    const int baseWindowLowInt = static_cast<int>(std::clamp(_baseWindowLow, 0.0f, 255.0f));
    const int baseWindowHighInt = static_cast<int>(
        std::clamp(_baseWindowHigh, static_cast<float>(baseWindowLowInt + 1), 255.0f));
    const float baseWindowSpan = std::max(1.0f, static_cast<float>(baseWindowHighInt - baseWindowLowInt));

    _overlayImageValid = false;
    _overlayImage = QImage();

    const QRect roiRect(roi.x, roi.y, roi.width, roi.height);

    const bool useComposite = (_surf_name == "segmentation" && _composite_enabled &&
                               (_composite_layers_front > 0 || _composite_layers_behind > 0));

    cv::Mat baseColor;

    z5::Dataset* baseDataset = volume ? volume->zarrDataset(_ds_sd_idx) : nullptr;

    // Check if this is a plane surface that should use plane composite rendering
    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surf.get());
    const bool usePlaneComposite = (plane != nullptr && _plane_composite_enabled &&
                                    (_plane_composite_layers_front > 0 || _plane_composite_layers_behind > 0));

    if (useComposite) {
        baseGray = render_composite(roi);
    } else if (usePlaneComposite) {
        // Plane composite: generate coords first, then composite along plane normal
        surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale, {static_cast<float>(roi.x), static_cast<float>(roi.y), _z_off});
        baseGray = render_composite_plane(roi, coords, plane->normal(cv::Vec3f(0, 0, 0)));
    } else {
        if (plane) {
            surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale, {static_cast<float>(roi.x), static_cast<float>(roi.y), _z_off});

        } else {
            cv::Vec2f roi_c = {roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f};
            _ptr = surf->pointer();
            cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
            surf->move(_ptr, diff / _scale);
            _vis_center = roi_c;
            surf->gen(&coords, nullptr, roi.size(), _ptr, _scale, {-roi.width / 2.0f, -roi.height / 2.0f, _z_off});
        }

        if (!baseDataset) {
            return cv::Mat();
        }
        readInterpolated3D(baseGray, baseDataset, coords * _ds_scale, cache, _useFastInterpolation);
    }

    if (baseGray.empty()) {
        return cv::Mat();
    }

    // Apply ISO cutoff - zero out values below threshold
    if (_iso_cutoff > 0 && !baseGray.empty()) {
        cv::threshold(baseGray, baseGray, _iso_cutoff - 1, 0, cv::THRESH_TOZERO);
    }

    cv::Mat baseProcessed;

    // Apply stretching if enabled
    if (_stretchValues) {
        double minVal, maxVal;
        cv::minMaxLoc(baseGray, &minVal, &maxVal);
        const double range = std::max(1.0, maxVal - minVal);

        cv::Mat baseFloat;
        baseGray.convertTo(baseFloat, CV_32F);
        baseFloat -= static_cast<float>(minVal);
        baseFloat /= static_cast<float>(range);
        baseFloat.convertTo(baseProcessed, CV_8U, 255.0f);
    } else {
        // Apply window/level transformation
        cv::Mat baseFloat;
        baseGray.convertTo(baseFloat, CV_32F);
        baseFloat -= static_cast<float>(baseWindowLowInt);
        baseFloat /= baseWindowSpan;
        cv::max(baseFloat, 0.0f, baseFloat);
        cv::min(baseFloat, 1.0f, baseFloat);
        baseFloat.convertTo(baseProcessed, CV_8U, 255.0f);
    }

    // Apply colormap if specified
    if (!_baseColormapId.empty()) {
        const auto& spec = volume_viewer_cmaps::resolve(_baseColormapId);
        baseColor = volume_viewer_cmaps::makeColors(baseProcessed, spec);
    } else {
        // Convert to BGR
        if (baseProcessed.channels() == 1) {
            cv::cvtColor(baseProcessed, baseColor, cv::COLOR_GRAY2BGR);
        } else {
            baseColor = baseProcessed.clone();
        }
    }

    if (_overlayVolume && _overlayOpacity > 0.0f) {
        if (coords.empty()) {
            if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
                surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale, {static_cast<float>(roi.x), static_cast<float>(roi.y), _z_off});
            } else {
                cv::Vec2f roi_c = {roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f};
                _ptr = surf->pointer();
                cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
                surf->move(_ptr, diff / _scale);
                _vis_center = roi_c;
                surf->gen(&coords, nullptr, roi.size(), _ptr, _scale, {-roi.width / 2.0f, -roi.height / 2.0f, _z_off});
            }
        }

        if (!coords.empty()) {
            int overlayIdx = 0;
            float overlayScale = 1.0f;
            if (_overlayVolume->numScales() > 0) {
                overlayIdx = std::min<int>(_ds_sd_idx, static_cast<int>(_overlayVolume->numScales()) - 1);
                overlayScale = std::pow(2.0f, -overlayIdx);
            }

            cv::Mat_<uint8_t> overlayValues;
            z5::Dataset* overlayDataset = _overlayVolume->zarrDataset(overlayIdx);
            readInterpolated3D(overlayValues, overlayDataset, coords * overlayScale, cache, /*nearest_neighbor=*/true);

            if (!overlayValues.empty()) {
                const int windowLow = static_cast<int>(std::clamp(_overlayWindowLow, 0.0f, 255.0f));
                const int windowHigh = static_cast<int>(std::clamp(_overlayWindowHigh, static_cast<float>(windowLow + 1), 255.0f));

                cv::Mat activeMask;
                cv::compare(overlayValues, windowLow, activeMask, cv::CmpTypes::CMP_GE);

                if (cv::countNonZero(activeMask) > 0) {
                    cv::Mat overlayScaled;
                    overlayValues.convertTo(overlayScaled, CV_32F);
                    overlayScaled -= static_cast<float>(windowLow);
                    overlayScaled.setTo(0.0f, overlayScaled < 0.0f);
                    const float windowSpan = std::max(1.0f, static_cast<float>(windowHigh - windowLow));
                    overlayScaled /= windowSpan;
                    cv::threshold(overlayScaled, overlayScaled, 1.0f, 1.0f, cv::THRESH_TRUNC);

                    cv::Mat overlayColorInput;
                    overlayScaled.convertTo(overlayColorInput, CV_8U, 255.0f);

                    const auto& spec = volume_viewer_cmaps::resolve(_overlayColormapId);
                    cv::Mat overlayColor = volume_viewer_cmaps::makeColors(overlayColorInput, spec);

                    if (!overlayColor.empty()) {
                        cv::Mat inactiveMask;
                        cv::bitwise_not(activeMask, inactiveMask);
                        overlayColor.setTo(cv::Scalar(0, 0, 0), inactiveMask);

                        cv::Mat overlayBGRA;
                        cv::cvtColor(overlayColor, overlayBGRA, cv::COLOR_BGR2BGRA);

                        std::vector<cv::Mat> channels;
                        cv::split(overlayBGRA, channels);
                        const uchar alphaValue = static_cast<uchar>(std::round(std::clamp(_overlayOpacity, 0.0f, 1.0f) * 255.0f));
                        channels[3].setTo(alphaValue, activeMask);
                        channels[3].setTo(0, inactiveMask);
                        cv::merge(channels, overlayBGRA);

                        cv::cvtColor(overlayBGRA, overlayBGRA, cv::COLOR_BGRA2RGBA);
                        QImage overlayImage(overlayBGRA.data, overlayBGRA.cols, overlayBGRA.rows, overlayBGRA.step, QImage::Format_RGBA8888);
                        _overlayImage = overlayImage.copy();
                        _overlayImageValid = true;
                    }
                }
            }
        }
    }

    // Surface overlap detection
    if (_surfaceOverlayEnabled && !_surfaceOverlayName.empty() && _surf_col && !baseColor.empty()) {
        auto overlaySurf = _surf_col->surface(_surfaceOverlayName);
        if (overlaySurf && overlaySurf != surf) {
            cv::Mat_<cv::Vec3f> overlayCoords;

            // Generate coordinates for overlay surface using the same ROI parameters
            if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
                overlaySurf->gen(&overlayCoords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale,
                               {static_cast<float>(roi.x), static_cast<float>(roi.y), _z_off});
            } else {
                cv::Vec2f roi_c = {roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f};
                auto overlayPtr = overlaySurf->pointer();
                cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
                overlaySurf->move(overlayPtr, diff / _scale);
                overlaySurf->gen(&overlayCoords, nullptr, roi.size(), overlayPtr, _scale,
                               {-roi.width / 2.0f, -roi.height / 2.0f, _z_off});
            }

            // Compute distances and create overlap mask
            if (!overlayCoords.empty() && overlayCoords.size() == coords.size()) {
                cv::Mat_<uint8_t> overlapMask(baseColor.size(), uint8_t(0));

                #pragma omp parallel for collapse(2)
                for (int y = 0; y < coords.rows; ++y) {
                    for (int x = 0; x < coords.cols; ++x) {
                        const cv::Vec3f& basePos = coords(y, x);
                        const cv::Vec3f& overlayPos = overlayCoords(y, x);

                        // Check if both positions are valid (not -1)
                        if (basePos[0] >= 0 && overlayPos[0] >= 0) {
                            // Compute Euclidean distance
                            cv::Vec3f diff = basePos - overlayPos;
                            float distance = std::sqrt(diff.dot(diff));

                            if (distance < _surfaceOverlapThreshold) {
                                overlapMask(y, x) = 255;
                            }
                        }
                    }
                }

                // Blend yellow highlight where surfaces overlap
                if (cv::countNonZero(overlapMask) > 0) {
                    const cv::Vec3b highlightColor(0, 255, 255); // Yellow in BGR
                    const float blendFactor = 0.5f; // 50% blend

                    for (int y = 0; y < baseColor.rows; ++y) {
                        for (int x = 0; x < baseColor.cols; ++x) {
                            if (overlapMask(y, x) > 0) {
                                cv::Vec3b& pixel = baseColor.at<cv::Vec3b>(y, x);
                                pixel = pixel * (1.0f - blendFactor) + highlightColor * blendFactor;
                            }
                        }
                    }
                }
            }
        }
    }

    return baseColor;
}

void CVolumeViewer::setBaseColormap(const std::string& colormapId)
{
    if (_baseColormapId == colormapId) {
        return;
    }
    _baseColormapId = colormapId;
    if (volume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setStretchValues(bool enabled)
{
    if (_stretchValues == enabled) {
        return;
    }
    _stretchValues = enabled;
    if (volume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setSurfaceOverlayEnabled(bool enabled)
{
    if (_surfaceOverlayEnabled == enabled) {
        return;
    }
    _surfaceOverlayEnabled = enabled;
    if (volume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setSurfaceOverlay(const std::string& surfaceName)
{
    if (_surfaceOverlayName == surfaceName) {
        return;
    }
    _surfaceOverlayName = surfaceName;
    if (volume && _surfaceOverlayEnabled) {
        renderVisible(true);
    }
}

void CVolumeViewer::setSurfaceOverlapThreshold(float threshold)
{
    threshold = std::max(0.1f, threshold);
    if (std::abs(threshold - _surfaceOverlapThreshold) < 1e-6f) {
        return;
    }
    _surfaceOverlapThreshold = threshold;
    if (volume && _surfaceOverlayEnabled) {
        renderVisible(true);
    }
}

void CVolumeViewer::setPlaneCompositeEnabled(bool enabled)
{
    if (_plane_composite_enabled == enabled) {
        return;
    }
    _plane_composite_enabled = enabled;
    if (volume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setPlaneCompositeLayers(int front, int behind)
{
    front = std::max(0, front);
    behind = std::max(0, behind);
    if (_plane_composite_layers_front == front && _plane_composite_layers_behind == behind) {
        return;
    }
    _plane_composite_layers_front = front;
    _plane_composite_layers_behind = behind;
    if (volume && _plane_composite_enabled) {
        renderVisible(true);
    }
}

cv::Mat_<uint8_t> CVolumeViewer::render_composite_plane(const cv::Rect &roi, const cv::Mat_<cv::Vec3f> &coords, const cv::Vec3f &planeNormal)
{
    cv::Mat_<uint8_t> img;

    if (coords.empty() || !volume || !volume->zarrDataset(_ds_sd_idx)) {
        return img;
    }

    // Determine z range based on front and behind layers
    // For planes, "front" means along the positive normal direction
    int z_start = _composite_reverse_direction ? -_plane_composite_layers_behind : -_plane_composite_layers_front;
    int z_end = _composite_reverse_direction ? _plane_composite_layers_front : _plane_composite_layers_behind;

    // Setup compositing parameters (reuse the same parameters as segmentation composite)
    CompositeParams params;
    params.method = _composite_method;
    params.alphaMin = _composite_alpha_min / 255.0f;
    params.alphaMax = _composite_alpha_max / 255.0f;
    params.alphaOpacity = _composite_material / 255.0f;
    params.alphaCutoff = _composite_alpha_threshold / 10000.0f;
    params.isoCutoff = static_cast<uint8_t>(_iso_cutoff);
    params.gradientScale = _composite_gradient_scale;
    params.stddevScale = _composite_stddev_scale;
    params.laplacianScale = _composite_laplacian_scale;
    params.rangeScale = _composite_range_scale;
    params.gradientSumScale = _composite_gradient_sum_scale;
    params.sobelScale = _composite_sobel_scale;
    params.localContrastScale = _composite_local_contrast_scale;
    params.entropyScale = _composite_entropy_scale;
    params.peakThreshold = _composite_peak_threshold;
    params.peakCountScale = _composite_peak_count_scale;
    params.countThreshold = _composite_count_threshold;
    params.thresholdCountScale = _composite_threshold_count_scale;
    params.percentile = _composite_percentile;
    params.weightedMeanSigma = _composite_weighted_mean_sigma;

    // Always use fast path with constant normal (nearest neighbor, no mutex)
    readCompositeFastConstantNormal(
        img,
        volume->zarrDataset(_ds_sd_idx),
        coords * _ds_scale,
        planeNormal,  // Single constant normal for all pixels
        _ds_scale,    // z step per layer (in dataset coordinates)
        z_start, z_end,
        params,
        _fastCompositeCache
    );

    return img;
}