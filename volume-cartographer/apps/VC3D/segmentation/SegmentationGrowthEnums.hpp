#pragma once

#include <QString>

enum class SegmentationGrowthMethod {
    Tracer = 0,
    Corrections = 1,
    Extrapolation = 2,
};

inline QString segmentationGrowthMethodToString(SegmentationGrowthMethod method)
{
    switch (method) {
    case SegmentationGrowthMethod::Tracer:
        return QStringLiteral("Tracer");
    case SegmentationGrowthMethod::Corrections:
        return QStringLiteral("Corrections");
    case SegmentationGrowthMethod::Extrapolation:
        return QStringLiteral("Extrapolation");
    }
    return QStringLiteral("Unknown");
}

inline SegmentationGrowthMethod segmentationGrowthMethodFromInt(int value)
{
    if (value == static_cast<int>(SegmentationGrowthMethod::Corrections)) {
        return SegmentationGrowthMethod::Corrections;
    }
    if (value == static_cast<int>(SegmentationGrowthMethod::Extrapolation)) {
        return SegmentationGrowthMethod::Extrapolation;
    }
    return SegmentationGrowthMethod::Tracer;
}

enum class SegmentationGrowthDirection {
    All = 0,
    Up,
    Down,
    Left,
    Right,
};

inline QString segmentationGrowthDirectionToString(SegmentationGrowthDirection direction)
{
    switch (direction) {
    case SegmentationGrowthDirection::All:
        return QStringLiteral("All");
    case SegmentationGrowthDirection::Up:
        return QStringLiteral("Up");
    case SegmentationGrowthDirection::Down:
        return QStringLiteral("Down");
    case SegmentationGrowthDirection::Left:
        return QStringLiteral("Left");
    case SegmentationGrowthDirection::Right:
        return QStringLiteral("Right");
    }
    return QStringLiteral("All");
}

inline SegmentationGrowthDirection segmentationGrowthDirectionFromInt(int value)
{
    switch (value) {
    case static_cast<int>(SegmentationGrowthDirection::Up):
        return SegmentationGrowthDirection::Up;
    case static_cast<int>(SegmentationGrowthDirection::Down):
        return SegmentationGrowthDirection::Down;
    case static_cast<int>(SegmentationGrowthDirection::Left):
        return SegmentationGrowthDirection::Left;
    case static_cast<int>(SegmentationGrowthDirection::Right):
        return SegmentationGrowthDirection::Right;
    default:
        return SegmentationGrowthDirection::All;
    }
}

enum class SegmentationDirectionFieldOrientation {
    Normal = 0,
    Horizontal = 1,
    Vertical = 2,
};

inline QString segmentationDirectionFieldOrientationKey(SegmentationDirectionFieldOrientation orientation)
{
    switch (orientation) {
    case SegmentationDirectionFieldOrientation::Horizontal:
        return QStringLiteral("horizontal");
    case SegmentationDirectionFieldOrientation::Vertical:
        return QStringLiteral("vertical");
    case SegmentationDirectionFieldOrientation::Normal:
    default:
        return QStringLiteral("normal");
    }
}

inline SegmentationDirectionFieldOrientation segmentationDirectionFieldOrientationFromInt(int value)
{
    switch (value) {
    case static_cast<int>(SegmentationDirectionFieldOrientation::Horizontal):
        return SegmentationDirectionFieldOrientation::Horizontal;
    case static_cast<int>(SegmentationDirectionFieldOrientation::Vertical):
        return SegmentationDirectionFieldOrientation::Vertical;
    default:
        return SegmentationDirectionFieldOrientation::Normal;
    }
}

enum class ExtrapolationType {
    Linear = 0,
    Quadratic = 1,
    LinearFit = 2,  // Linear extrapolation + SDT Newton refinement
    SkeletonPath = 3,  // 2D skeleton analysis + 3D Dijkstra path following
};

inline QString extrapolationTypeToString(ExtrapolationType type)
{
    switch (type) {
    case ExtrapolationType::Linear:
        return QStringLiteral("Linear");
    case ExtrapolationType::Quadratic:
        return QStringLiteral("Quadratic");
    case ExtrapolationType::LinearFit:
        return QStringLiteral("Linear+Fit");
    case ExtrapolationType::SkeletonPath:
        return QStringLiteral("Skeleton Path");
    }
    return QStringLiteral("Linear");
}

inline ExtrapolationType extrapolationTypeFromInt(int value)
{
    if (value == static_cast<int>(ExtrapolationType::Quadratic)) {
        return ExtrapolationType::Quadratic;
    }
    if (value == static_cast<int>(ExtrapolationType::LinearFit)) {
        return ExtrapolationType::LinearFit;
    }
    if (value == static_cast<int>(ExtrapolationType::SkeletonPath)) {
        return ExtrapolationType::SkeletonPath;
    }
    return ExtrapolationType::Linear;
}
