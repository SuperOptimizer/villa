#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Render.hpp"
#include "vc/core/util/JsonSafe.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;


void print_usage(const char* prog_name) {
    std::cout << "usage: " << prog_name << " [OPTIONS]\n\n"
              << "Generates masks for segment surfaces.\n\n"
              << "Options:\n"
              << "  --segment <path>              Path to tiff/xyz segment\n"
              << "  --segments <path1,path2,...>  Comma-separated list of segment paths\n"
              << "  --volpkg <path>               Path to volpkg directory (processes all 'inspect' segments)\n"
              << "  --volume <path>               Path to zarr volume (optional, for image layers)\n"
              << "  --output <path>               Output mask path (only with --segment)\n"
              << "  --unique                      Generate unique_mask.tif for inspect segments (blacks out overlaps)\n"
              << "  --overwrite                   Overwrite existing mask files\n\n"
              << "Examples:\n"
              << "  " << prog_name << " --segment /path/to/segment\n"
              << "  " << prog_name << " --segments /path/seg1,/path/seg2 --volume /path/volume.zarr\n"
              << "  " << prog_name << " --volpkg /path/to/volpkg --unique\n"
              << std::endl;
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    // Parse arguments
    fs::path single_segment;
    std::vector<fs::path> segment_paths;
    fs::path volpkg_path;
    fs::path volume_path;
    fs::path output_mask_path;
    bool overwrite = false;
    bool generate_unique = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (arg == "--segment" && i + 1 < argc) {
            single_segment = argv[++i];
        }
        else if (arg == "--segments" && i + 1 < argc) {
            std::string segments_str = argv[++i];
            size_t start = 0;
            size_t end = segments_str.find(',');
            while (end != std::string::npos) {
                segment_paths.push_back(segments_str.substr(start, end - start));
                start = end + 1;
                end = segments_str.find(',', start);
            }
            segment_paths.push_back(segments_str.substr(start));
        }
        else if (arg == "--volpkg" && i + 1 < argc) {
            volpkg_path = argv[++i];
        }
        else if (arg == "--volume" && i + 1 < argc) {
            volume_path = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            output_mask_path = argv[++i];
        }
        else if (arg == "--overwrite") {
            overwrite = true;
        }
        else if (arg == "--unique") {
            generate_unique = true;
        }
    }

    // Validate arguments
    int mode_count = (!single_segment.empty() ? 1 : 0) +
                     (!segment_paths.empty() ? 1 : 0) +
                     (!volpkg_path.empty() ? 1 : 0);

    if (mode_count == 0) {
        std::cerr << "Error: Must specify --segment, --segments, or --volpkg" << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (mode_count > 1) {
        std::cerr << "Error: Can only use one of --segment, --segments, or --volpkg" << std::endl;
        return EXIT_FAILURE;
    }

    // Handle single segment mode
    if (!single_segment.empty()) {
        segment_paths.push_back(single_segment);
        if (output_mask_path.empty()) {
            output_mask_path = single_segment / "mask.tif";
        }
    }

    // Handle volpkg mode
    std::shared_ptr<VolumePkg> vpkg;
    if (!volpkg_path.empty()) {
        try {
            vpkg = VolumePkg::New(volpkg_path);

            // Get all segmentation IDs and filter for inspect tags
            auto seg_ids = vpkg->segmentationIDs();
            std::cout << "Found " << seg_ids.size() << " segments in volpkg" << std::endl;

            for (const auto& seg_id : seg_ids) {
                auto seg = vpkg->segmentation(seg_id);
                if (!seg) continue;

                auto surf = vpkg->loadSurface(seg_id);
                if (!surf || !surf->surface() || !surf->surface()->meta) {
                    continue;
                }

                // Check if segment has 'inspect' tag
                if (vc::json_safe::has_tag(surf->surface()->meta, "inspect")) {
                    segment_paths.push_back(surf->path);
                    std::cout << "  Found inspect segment: " << seg_id << std::endl;
                }
            }

            if (segment_paths.empty()) {
                std::cout << "No segments with 'inspect' tag found in volpkg" << std::endl;
                return EXIT_SUCCESS;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading volpkg: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Load volume if provided
    std::shared_ptr<Volume> volume;
    ChunkCache<uint8_t>* cache = nullptr;

    if (!volume_path.empty()) {
        try {
            volume = Volume::New(volume_path);
            cache = new ChunkCache<uint8_t>(1ULL * 1024ULL * 1024ULL * 1024ULL);
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading volume: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Process each segment
    int success_count = 0;
    int skip_count = 0;

    for (const auto& seg_path : segment_paths) {
        try {
            std::string seg_name = seg_path.filename().string();
            fs::path mask_path = output_mask_path.empty() ?
                                (seg_path / "mask.tif") : output_mask_path;

            // Check if mask already exists
            if (fs::exists(mask_path) && !overwrite) {
                std::cout << seg_name << ": Mask already exists, skipping (use --overwrite to regenerate)" << std::endl;
                skip_count++;
                continue;
            }

            std::cout << seg_name << ": Loading surface..." << std::endl;

            // Load the surface
            QuadSurface* surf = load_quad_from_tifxyz(seg_path);

            cv::Mat_<uint8_t> mask;
            cv::Mat_<uint8_t> img;

            // Check if this is an inspect segment and we need to generate unique mask
            if (generate_unique && vpkg) {
                std::cout << seg_name << ": Generating unique mask..." << std::endl;

                // Load all other surfaces for comparison
                std::vector<QuadSurface*> other_surfs;
                auto all_seg_ids = vpkg->segmentationIDs();

                for (const auto& other_id : all_seg_ids) {
                    auto other_surf_meta = vpkg->loadSurface(other_id);
                    if (other_surf_meta && other_surf_meta->path != seg_path) {
                        auto other_surf = other_surf_meta->surface();
                        if (other_surf) {
                            other_surfs.push_back(other_surf);
                        }
                    }
                }

                std::cout << seg_name << ": Comparing against " << other_surfs.size() << " other surfaces" << std::endl;

                // Generate unique mask
                render_unique_mask(surf, other_surfs, mask, img,
                                 volume ? volume->zarrDataset(0) : nullptr,
                                 cache, 1.0f);

                // Output to unique_mask.tif
                mask_path = seg_path / "unique_mask.tif";
            }
            else {
                std::cout << seg_name << ": Generating mask..." << std::endl;

                // Generate regular mask
                if (volume && cache) {
                    render_surface_image(surf, mask, img,
                                       volume->zarrDataset(0),
                                       cache, 1.0f);
                } else {
                    cv::Mat_<cv::Vec3f> coords;
                    render_binary_mask(surf, mask, coords, 1.0f);
                }
            }

            // Save the mask
            if (volume && cache && !img.empty()) {
                cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);
                std::vector<cv::Mat> layers = {mask, img};
                if (!cv::imwritemulti(mask_path.string(), layers)) {
                    std::cerr << seg_name << ": Error writing mask to " << mask_path << std::endl;
                    delete surf;
                    continue;
                }
            } else {
                if (!cv::imwrite(mask_path.string(), mask)) {
                    std::cerr << seg_name << ": Error writing mask to " << mask_path << std::endl;
                    delete surf;
                    continue;
                }
            }

            // Report statistics
            int valid_count = cv::countNonZero(mask);
            int total_count = mask.rows * mask.cols;
            float valid_percent = (float)valid_count / total_count * 100.0f;

            std::cout << seg_name << ": Mask generated successfully at " << mask_path << std::endl;
            std::cout << "  Dimensions: " << mask.size() << std::endl;
            std::cout << "  Valid pixels: " << valid_count << " / " << total_count
                      << " (" << std::fixed << std::setprecision(1) << valid_percent << "%)" << std::endl;

            delete surf;
            success_count++;
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing " << seg_path << ": " << e.what() << std::endl;
        }
    }

    if (cache) delete cache;

    std::cout << "\nSummary: " << success_count << " masks generated, "
              << skip_count << " skipped" << std::endl;

    return success_count > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
