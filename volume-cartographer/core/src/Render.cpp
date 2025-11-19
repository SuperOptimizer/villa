#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include <opencv2/imgproc.hpp>


void render_binary_mask(QuadSurface* surf,
                         cv::Mat_<uint8_t>& mask,
                         cv::Mat_<cv::Vec3f>& coords_out,
                         float scale) {

    // Get raw mesh vertices - this is the actual defined surface
    cv::Mat_<cv::Vec3f> rawPts = surf->rawPoints();
    cv::Size rawSize = rawPts.size();

    // Calculate target size: scale the raw size by the user's requested scale
    cv::Size targetSize(
        static_cast<int>(std::round(rawSize.width * scale)),
        static_cast<int>(std::round(rawSize.height * scale))
    );

    std::cout << "render_binary_mask: rawSize=" << rawSize
              << " targetSize=" << targetSize
              << " scale=" << scale << std::endl;

    // Create mask from raw points at their native resolution
    cv::Mat_<uint8_t> rawMask(rawSize);
    int rawValid = 0;

#pragma omp parallel for schedule(dynamic, 1) reduction(+:rawValid)
    for(int j = 0; j < rawSize.height; j++) {
        for(int i = 0; i < rawSize.width; i++) {
            const cv::Vec3f& pt = rawPts(j, i);
            // Check for undefined vertices: either NaN/inf OR the sentinel value [-1, -1, -1]
            bool isValid = std::isfinite(pt[0]) && std::isfinite(pt[1]) && std::isfinite(pt[2]) &&
                          !(pt[0] == -1.0f && pt[1] == -1.0f && pt[2] == -1.0f);
            rawMask(j, i) = isValid ? 255 : 0;
            if (isValid) rawValid++;
        }
    }

    // Upscale the mask using nearest neighbor to target resolution
    cv::resize(rawMask, mask, targetSize, 0, 0, cv::INTER_NEAREST);

    // Generate coords at target resolution for rendering
    cv::Vec3f ptr = surf->pointer();
    cv::Vec3f offset(-rawSize.width/2.0f, -rawSize.height/2.0f, 0);
    surf->gen(&coords_out, nullptr, targetSize, ptr, 1.0f / scale, offset);

    int finalValid = cv::countNonZero(mask);
    std::cout << "  rawValid=" << rawValid << "/" << (rawSize.width * rawSize.height)
              << " (" << (100.0 * rawValid / (rawSize.width * rawSize.height)) << "%)"
              << " targetValid=" << finalValid << "/" << (targetSize.width * targetSize.height)
              << " (" << (100.0 * finalValid / (targetSize.width * targetSize.height)) << "%)" << std::endl;
}

void render_image_from_coords(const cv::Mat_<cv::Vec3f>& coords,
                              cv::Mat_<uint8_t>& img,
                              z5::Dataset* ds,
                              ChunkCache<uint8_t>* cache) {
    if (!ds || !cache) {
        throw std::runtime_error("Dataset or cache is null in render_image_from_coords");
    }

    readInterpolated3D(img, ds, coords, cache);
    std::cout << "render_image_from_coords: completed" << std::endl;
}

// Render surface - generates both mask and image
void render_surface_image(QuadSurface* surf,
                         cv::Mat_<uint8_t>& mask,
                         cv::Mat_<uint8_t>& img,
                         z5::Dataset* ds,
                         ChunkCache<uint8_t>* cache,
                         float scale) {

    cv::Mat_<cv::Vec3f> coords;
    render_binary_mask(surf, mask, coords, scale);
    render_image_from_coords(coords, img, ds, cache);

    std::cout << "render_surface_image: completed" << std::endl;
}

// Generate unique mask - blacks out areas already traced by other surfaces
void render_unique_mask(QuadSurface* inspect_surf,
                       const std::vector<QuadSurface*>& other_surfs,
                       cv::Mat_<uint8_t>& unique_mask,
                       cv::Mat_<uint8_t>& img,
                       z5::Dataset* ds,
                       ChunkCache<uint8_t>* cache,
                       float scale) {

    cv::Mat_<cv::Vec3f> coords;

    // First, generate the base mask for the inspect surface
    render_binary_mask(inspect_surf, unique_mask, coords, scale);

    std::cout << "render_unique_mask: checking against " << other_surfs.size() << " other surfaces" << std::endl;

    // Now black out any pixels that are covered by other surfaces
    int blackedOut = 0;
    int total = 0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:blackedOut, total)
    for(int j = 0; j < unique_mask.rows; j++) {
        for(int i = 0; i < unique_mask.cols; i++) {
            // Skip pixels that are already black (invalid in the inspect surface)
            if (unique_mask(j, i) == 0) {
                continue;
            }

            total++;
            const cv::Vec3f& point = coords(j, i);

            // Skip invalid coordinates
            if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]) ||
                (point[0] == -1.0f && point[1] == -1.0f && point[2] == -1.0f)) {
                continue;
            }

            // Check if this point is contained in any other surface
            for (const auto* other : other_surfs) {
                if (other && other->containsPoint(point, 2.0f)) {
                    unique_mask(j, i) = 0;  // Black out this pixel
                    blackedOut++;
                    break;
                }
            }
        }
    }

    std::cout << "  Blacked out " << blackedOut << " / " << total << " pixels"
              << " (" << (total > 0 ? (100.0 * blackedOut / total) : 0.0) << "%)" << std::endl;

    // Generate the image if requested
    if (ds && cache) {
        render_image_from_coords(coords, img, ds, cache);
    }

    std::cout << "render_unique_mask: completed" << std::endl;
}