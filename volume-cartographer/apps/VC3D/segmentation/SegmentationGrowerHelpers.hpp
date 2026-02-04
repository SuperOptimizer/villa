/**
 * @file SegmentationGrowerHelpers.hpp
 * @brief Helper function declarations for SegmentationGrower
 *
 * This file declares utility functions used by SegmentationGrower.
 * Implementation is in SegmentationGrowerHelpers.cpp.
 */

#pragma once

#include "SegmentationCorrections.hpp"

#include <QString>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

class QuadSurface;
class SurfacePatchIndex;
class SurfacePanelController;
class ViewerManager;
class VolumePkg;

// Structure holding bounding box information for corrections
struct CorrectionsBounds {
    cv::Vec3f worldMin;
    cv::Vec3f worldMax;
    cv::Rect gridRegion;
};

// Returns the cache root path for a volume package
QString cacheRootForVolumePkg(const std::shared_ptr<VolumePkg>& pkg);

// Ensures the surface has a generations channel (creates one if missing)
void ensureGenerationsChannel(QuadSurface* surface);

// Ensures the surface has a meta JSON object (creates one if missing)
void ensureSurfaceMetaObject(QuadSurface* surface);

// Checks if a point is invalid (NaN or sentinel value)
bool isInvalidPoint(const cv::Vec3f& value);

// Converts world position to grid index using approximation
std::optional<std::pair<int, int>> worldToGridIndexApprox(
    QuadSurface* surface,
    const cv::Vec3f& worldPos,
    cv::Vec3f& pointerSeed,
    bool& pointerSeedValid,
    SurfacePatchIndex* patchIndex = nullptr);

// Locates grid index using patch index acceleration
std::optional<std::pair<int, int>> locateGridIndexWithPatchIndex(
    QuadSurface* surface,
    SurfacePatchIndex* patchIndex,
    const cv::Vec3f& worldPos,
    cv::Vec3f& pointerSeed,
    bool& pointerSeedValid);

// Computes the affected grid bounds for correction points
std::optional<cv::Rect> computeCorrectionsAffectedBounds(
    QuadSurface* surface,
    const SegmentationCorrectionsPayload& corrections,
    ViewerManager* viewerManager);

// Computes 3D world-space bounding box from correction points
std::optional<CorrectionsBounds> computeCorrectionsBounds(
    const SegmentationCorrectionsPayload& corrections,
    QuadSurface* surface,
    float minWorldSize = 512.0f);

// Crops a QuadSurface to a 2D grid region
std::unique_ptr<QuadSurface> cropSurfaceToGridRegion(
    const QuadSurface* surface,
    const cv::Rect& gridRegion);

// Generates ISO 8601 timestamp string for folder naming
std::string generateTimestampString();

// Saves correction annotation data to disk
void saveCorrectionsAnnotation(
    const std::filesystem::path& volpkgRoot,
    const std::string& segmentId,
    const QuadSurface* beforeCrop,
    const QuadSurface* afterCrop,
    const SegmentationCorrectionsPayload& corrections,
    const std::vector<std::string>& volumeIds,
    const std::string& growthVolumeId,
    const CorrectionsBounds& bounds);

// Queues index update for a surface within given bounds
void queueIndexUpdateForBounds(
    SurfacePatchIndex* index,
    const std::shared_ptr<QuadSurface>& surface,
    const cv::Rect& vertexRect);

// Synchronizes surface metadata with the volume package
void synchronizeSurfaceMeta(
    const std::shared_ptr<VolumePkg>& pkg,
    QuadSurface* surface,
    SurfacePanelController* panel);

// Refreshes all segmentation viewers
void refreshSegmentationViewers(ViewerManager* manager);
