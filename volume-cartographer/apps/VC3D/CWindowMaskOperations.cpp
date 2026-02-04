/**
 * @file CWindowMaskOperations.cpp
 * @brief Mask editing operations extracted from CWindow
 *
 * This file contains methods for editing and appending to surface masks.
 * Extracted from CWindow.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"

#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"
#include "ViewerManager.hpp"

#include <QDesktopServices>
#include <QMessageBox>
#include <QStatusBar>
#include <QUrl>

#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Render.hpp"

void CWindow::onEditMaskPressed()
{
    auto surf = _surf_weak.lock();
    if (!surf)
        return;

    std::filesystem::path path = surf->path/"mask.tif";

    if (!std::filesystem::exists(path)) {
        cv::Mat_<uint8_t> mask;
        cv::Mat_<cv::Vec3f> coords; // Not used after generation

        // Generate the binary mask at raw points resolution
        render_binary_mask(surf.get(), mask, coords, 1.0f);

        // Save just the mask as single layer
        cv::imwrite(path.string(), mask);

        // Update metadata
        (*surf->meta)["date_last_modified"] = get_surface_time_str();
        surf->save_meta();
    }

    QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));
}

void CWindow::onAppendMaskPressed()
{
    auto surf = _surf_weak.lock();
    if (!surf || !currentVolume) {
        if (!surf) {
            QMessageBox::warning(this, tr("Error"), tr("No surface selected."));
        } else {
            QMessageBox::warning(this, tr("Error"), tr("No volume loaded."));
        }
        return;
    }

    std::filesystem::path path = surf->path/"mask.tif";

    cv::Mat_<uint8_t> mask;
    cv::Mat_<uint8_t> img;
    std::vector<cv::Mat> existing_layers;

    z5::Dataset* ds = currentVolume->zarrDataset(0);

    try {
        // Find the segmentation viewer and check if composite is enabled
        CVolumeViewer* segViewer = segmentationViewer();
        bool useComposite = segViewer && segViewer->isCompositeEnabled();

        // Check if mask.tif exists
        if (std::filesystem::exists(path)) {
            // Load existing mask
            cv::imreadmulti(path.string(), existing_layers, cv::IMREAD_UNCHANGED);

            if (existing_layers.empty()) {
                QMessageBox::warning(this, tr("Error"), tr("Could not read existing mask file."));
                return;
            }

            // Use the first layer as the mask
            mask = existing_layers[0];
            cv::Size maskSize = mask.size();

            if (useComposite) {
                // Use composite rendering from the segmentation viewer
                img = segViewer->renderCompositeForSurface(surf, maskSize);
            } else {
                // Original single-layer rendering - use same approach as render_binary_mask
                cv::Size rawSize = surf->rawPointsPtr()->size();
                cv::Vec3f ptr = surf->pointer();
                cv::Vec3f offset(-rawSize.width/2.0f, -rawSize.height/2.0f, 0);

                // Use surface's scale so sx = _scale/_scale = 1.0, sampling 1:1 from raw points
                float surfScale = surf->scale()[0];
                cv::Mat_<cv::Vec3f> coords;
                surf->gen(&coords, nullptr, maskSize, ptr, surfScale, offset);

                std::cout << "[AppendMask non-composite] rawSize: " << rawSize.width << "x" << rawSize.height
                          << ", maskSize: " << maskSize.width << "x" << maskSize.height
                          << ", coords size: " << coords.cols << "x" << coords.rows
                          << ", surface._scale: " << surf->scale()[0] << std::endl;

                // Sample a few coords to verify they're in native voxel space
                if (coords.rows > 4 && coords.cols > 4) {
                    std::cout << "[AppendMask non-composite] coords[0,0]: " << coords(4,4)
                              << ", coords[center]: " << coords(coords.rows/2, coords.cols/2)
                              << ", coords[end]: " << coords(coords.rows-5, coords.cols-5) << std::endl;
                }

                render_image_from_coords(coords, img, ds, chunk_cache);
            }
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            std::cout << "[AppendMask] maskSize: " << maskSize.width << "x" << maskSize.height
                      << ", img size: " << img.cols << "x" << img.rows
                      << ", useComposite: " << useComposite << std::endl;

            // Append the new image layer to existing layers
            existing_layers.push_back(img);

            // Save all layers
            imwritemulti(path.string(), existing_layers);

            QString message = useComposite ?
                tr("Appended composite surface image to existing mask (now %1 layers)").arg(existing_layers.size()) :
                tr("Appended surface image to existing mask (now %1 layers)").arg(existing_layers.size());
            statusBar()->showMessage(message, 3000);

        } else {
            // No existing mask, generate both mask and image at raw points resolution
            cv::Mat_<cv::Vec3f> coords;
            render_binary_mask(surf.get(), mask, coords, 1.0f);
            cv::Size maskSize = mask.size();

            if (useComposite) {
                // Use composite rendering for image
                img = segViewer->renderCompositeForSurface(surf, maskSize);
            } else {
                // Original rendering
                render_surface_image(surf.get(), mask, img, ds, chunk_cache, 1.0f);
            }
            cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);

            // Save as new multi-layer TIFF
            std::vector<cv::Mat> layers = {mask, img};
            imwritemulti(path.string(), layers);

            QString message = useComposite ?
                tr("Created new surface mask with composite image data") :
                tr("Created new surface mask with image data");
            statusBar()->showMessage(message, 3000);
        }

        // Update metadata
        (*surf->meta)["date_last_modified"] = get_surface_time_str();
        surf->save_meta();

        QDesktopServices::openUrl(QUrl::fromLocalFile(path.string().c_str()));

    } catch (const std::exception& e) {
        QMessageBox::critical(this, tr("Error"),
                            tr("Failed to render surface: %1").arg(e.what()));
    }
}
