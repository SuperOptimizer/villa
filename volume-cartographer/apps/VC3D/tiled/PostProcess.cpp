#include "PostProcess.hpp"
#include "VolumeViewerCmaps.hpp"

#include <opencv2/imgproc.hpp>

cv::Mat applyPostProcess(const cv::Mat_<uint8_t>& gray,
                         const PostProcessParams& params)
{
    // Steps 1-4: delegate to core grayscale pipeline
    cv::Mat_<uint8_t> img = gray.clone();
    vc::applyPostProcess(img, params.toCoreParams());

    // Step 5: Colormap (Qt-dependent)
    cv::Mat color;
    if (!params.colormapId.empty()) {
        const auto& spec = volume_viewer_cmaps::resolve(params.colormapId);
        color = volume_viewer_cmaps::makeColors(img, spec);
    } else {
        if (img.channels() == 1) {
            cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
        } else {
            color = img.clone();
        }
    }

    return color;
}
