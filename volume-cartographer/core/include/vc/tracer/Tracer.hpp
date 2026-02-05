#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/matx.hpp>
#include "vc/tracer/TracerParams.hpp"

// Forward declarations
class QuadSurface;
class VCCollection;
namespace z5 { class Dataset; }
template<typename T> class ChunkCache;
struct Chunked3dFloatFromUint8;
struct Chunked3dVec3fFromUint8;

struct DirectionField
{
    DirectionField(std::string dir,
                   std::unique_ptr<Chunked3dVec3fFromUint8> field,
                   std::unique_ptr<Chunked3dFloatFromUint8> weight_dataset,
                   float weight = 1.0f)
        : direction(std::move(dir))
        , field_ptr(std::move(field))
        , weight_ptr(std::move(weight_dataset))
        , weight(weight)
    {
    }

    std::string direction;
    std::unique_ptr<Chunked3dVec3fFromUint8> field_ptr;
    std::unique_ptr<Chunked3dFloatFromUint8> weight_ptr;
    float weight{1.0f};
};

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const TracerParams &params, float voxelsize = 1.0);
QuadSurface *tracer(z5::Dataset *ds, float scale, ChunkCache<uint8_t> *cache, cv::Vec3f origin, const TracerParams &params, const std::string &cache_root = "", float voxelsize = 1.0, const std::vector<DirectionField> &direction_fields = {}, QuadSurface* resume_surf = nullptr, const std::filesystem::path& tgt_path = "", const std::string& meta_params_json = "", const VCCollection* corrections = nullptr);
