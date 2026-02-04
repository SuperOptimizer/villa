#pragma once

#include <cmath>
#include <cstdint>

#include <opencv2/core/matx.hpp>
#include <qtypes.h>

struct ColPoint
{
    uint64_t id;
    uint64_t collectionId;
    cv::Vec3f p = {0,0,0};
    float winding_annotation = NAN;
    qint64 creation_time = 0;
};

struct CollectionMetadata
{
    bool absolute_winding_number = false;
};
