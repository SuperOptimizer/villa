/**
 * @file CWindowAreaCalculation.cpp
 * @brief Area calculation and surface metrics extracted from CWindow
 *
 * This file contains helper functions and the recalcAreaForSegments method
 * for computing surface area from mesh data and masks.
 * Extracted from CWindow.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"

#include <QMessageBox>
#include <QStatusBar>
#include <QTreeWidgetItemIterator>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>

namespace
{

// --- Small config knobs (can be lifted to QSettings later) ------------------
constexpr bool kDeactivateWhenZero = true;     // mask 0 => deactivate; flip if workflow differs
constexpr double kTauDeactivate = 0.50;        // fraction of deactivating pixels needed to drop a quad
constexpr bool kBackfaceCullFolds = false;     // reduce double-count in folds by culling backfaces
constexpr double kCullDotEps = 1e-12;          // tolerance for backface culling
constexpr int kNormalDecimateMax = 128;        // sampling grid for global normal estimation

// --- Utilities ---------------------------------------------------------------
inline bool isFinite3(const cv::Vec3d& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Triangle area (standard "notorious" cross-product formula)
inline double tri_area3D(const cv::Vec3d& a, const cv::Vec3d& b, const cv::Vec3d& c)
{
    return 0.5 * cv::norm((b - a).cross(c - a));
}

// Triangle area with simple backface culling vs. a reference normal
inline double tri_area3D_culled(const cv::Vec3d& a,
                                const cv::Vec3d& b,
                                const cv::Vec3d& c,
                                const cv::Vec3d& refN,
                                double dot_eps)
{
    const cv::Vec3d n = (b - a).cross(c - a);
    const double dot = n.dot(refN);
    if (dot <= dot_eps) {
        return 0.0;  // backfacing or near parallel -> culled
    }
    return 0.5 * cv::norm(n);
}

// Choose largest image (by pixel count) among multi-page TIFFs
int choose_largest_page(const std::vector<cv::Mat>& pages)
{
    int bestIdx = -1;
    size_t bestPix = 0;
    for (int i = 0; i < static_cast<int>(pages.size()); ++i) {
        const size_t pix = static_cast<size_t>(pages[i].rows) * static_cast<size_t>(pages[i].cols);
        if (pix > bestPix) {
            bestPix = pix;
            bestIdx = i;
        }
    }
    return bestIdx;
}

// Robustly binarize an 8/16/32-bit single-channel mask to {0,1}
//  - fast path if already {0,255} (or {0,1})
//  - else Otsu
void binarize_mask(const cv::Mat& srcAnyDepth, cv::Mat1b& mask01)
{
    cv::Mat m;
    if (srcAnyDepth.channels() != 1) {
        cv::Mat gray;
        cv::cvtColor(srcAnyDepth, gray, cv::COLOR_BGR2GRAY);
        m = gray;
    } else {
        m = srcAnyDepth;
    }

    // Convert to 8U (preserving dynamic range)
    if (m.type() != CV_8U) {
        double minv = 0.0;
        double maxv = 0.0;
        cv::minMaxLoc(m, &minv, &maxv);
        if (std::abs(maxv - minv) < 1e-12) {
            mask01 = cv::Mat1b(m.size(), 0);
            return;
        }
        cv::Mat m8;
        m.convertTo(m8, CV_8U, 255.0 / (maxv - minv), (-minv) * 255.0 / (maxv - minv));
        m = m8;
    }

    // Fast path: already binary?
    int nz = cv::countNonZero(m);
    if (nz == 0) {
        mask01 = cv::Mat1b(m.size(), 0);
        return;
    }
    if (nz == m.rows * m.cols) {
        mask01 = cv::Mat1b(m.size(), 1);
        return;
    }

    // Check if unique values are (0,255) or (0,1)
    // (cheap test using bitwise ops)
    cv::Mat1b tmp;
    cv::threshold(m, tmp, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(m != tmp) == 0) {
        // values are {0, something}; normalize to {0,1}
        mask01 = (tmp > 0) / 255;
        return;
    }

    // Otsu threshold to {0,1}
    cv::Mat1b otsu;
    cv::threshold(m, otsu, 0, 1, cv::THRESH_BINARY | cv::THRESH_OTSU);
    mask01 = otsu;
}

// Load single-channel TIFF -> CV_32F
bool load_tif_as_float(const std::filesystem::path& file, cv::Mat1f& out)
{
    cv::Mat raw = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty() || raw.channels() != 1) {
        return false;
    }

    switch (raw.type()) {
        case CV_32FC1:
            out = raw;
            return true;
        case CV_64FC1:
            raw.convertTo(out, CV_32F);
            return true;
        default:
            raw.convertTo(out, CV_32F);
            return true;
    }
}

// 64-bit (double) integral image for 0/1 maps.
// ii has size (H+1, W+1), type CV_64F
inline double sumRect01d(const cv::Mat1d& ii, int x0, int y0, int x1, int y1)
{
    // rectangle is [x0,x1) × [y0,y1)
    return ii(y1, x1) - ii(y0, x1) - ii(y1, x0) + ii(y0, x0);
}

// Estimate a global reference normal from sparse samples of the grid
cv::Vec3d estimate_global_normal(const cv::Mat1f& X, const cv::Mat1f& Y, const cv::Mat1f& Z)
{
    const int H = X.rows;
    const int W = X.cols;
    const int sy = std::max(1, H / kNormalDecimateMax);
    const int sx = std::max(1, W / kNormalDecimateMax);

    cv::Vec3d acc(0, 0, 0);
    for (int y = 0; y + sy < H; y += sy) {
        for (int x = 0; x + sx < W; x += sx) {
            const cv::Vec3d A(X(y, x), Y(y, x), Z(y, x));
            const cv::Vec3d B(X(y, x + sx), Y(y, x + sx), Z(y, x + sx));
            const cv::Vec3d C(X(y + sy, x), Y(y + sy, x), Z(y + sy, x));
            if (!isFinite3(A) || !isFinite3(B) || !isFinite3(C)) {
                continue;
            }
            acc += (B - A).cross(C - A);
        }
    }
    const double nrm = cv::norm(acc);
    if (nrm < 1e-20) {
        return cv::Vec3d(0, 0, 1);  // fallback (rare)
    }
    return acc / nrm;
}

// Core: area from kept quads using original X/Y/Z grids, fractional mask rule, 64-bit integral,
// and optional backface culling against a global normal to reduce fold double-counting.
double area_from_mesh_and_mask(const cv::Mat1f& X,
                               const cv::Mat1f& Y,
                               const cv::Mat1f& Z,
                               const cv::Mat1b& mask01)
{
    const int Hq = X.rows;
    const int Wq = X.cols;
    if (Hq < 2 || Wq < 2) {
        return 0.0;
    }

    const int Hm = mask01.rows;
    const int Wm = mask01.cols;
    if (Hm <= 0 || Wm <= 0) {
        return 0.0;
    }

    // Build "deactivation" map: 1 when a pixel should deactivate, 0 otherwise
    cv::Mat1b deact;
    if (kDeactivateWhenZero) {
        deact = (mask01 == 0);
    } else {
        deact = (mask01 != 0);
    }

    // 64-bit integral image (double) -> no overflow for huge images
    cv::Mat1d ii;
    cv::integral(deact, ii, CV_64F);

    // Linear mapping from quad cells to mask pixels
    const double sx = static_cast<double>(Wm) / static_cast<double>(Wq - 1);
    const double sy = static_cast<double>(Hm) / static_cast<double>(Hq - 1);

    // Optional global normal for backface culling
    const cv::Vec3d refN = kBackfaceCullFolds ? estimate_global_normal(X, Y, Z) : cv::Vec3d(0, 0, 0);

    double total = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total) schedule(static)
#endif
    for (int qy = 0; qy < Hq - 1; ++qy) {
        for (int qx = 0; qx < Wq - 1; ++qx) {
            // Map UV cell [qx,qx+1)×[qy,qy+1) → mask rect [x0,x1)×[y0,y1)
            int x0 = static_cast<int>(std::floor(qx * sx));
            int y0 = static_cast<int>(std::floor(qy * sy));
            int x1 = static_cast<int>(std::ceil((qx + 1) * sx));  // A3 fix: ceil end
            int y1 = static_cast<int>(std::ceil((qy + 1) * sy));

            // Clamp and ensure ≥1 pixel extent
            x0 = std::clamp(x0, 0, Wm - 1);
            y0 = std::clamp(y0, 0, Hm - 1);
            x1 = std::clamp(x1, x0 + 1, Wm);
            y1 = std::clamp(y1, y0 + 1, Hm);

            const int rectPix = (x1 - x0) * (y1 - y0);
            if (rectPix <= 0) {
                continue;
            }

            const double deactCount = sumRect01d(ii, x0, y0, x1, y1);
            const double fracDeact = deactCount / static_cast<double>(rectPix);

            // Fractional rule (Brittle ANY-pixel fixed) -> robust fraction rule
            if (fracDeact >= kTauDeactivate) {
                continue;  // drop quad
            }

            // 3D corners
            const cv::Vec3d A(X(qy, qx), Y(qy, qx), Z(qy, qx));
            const cv::Vec3d B(X(qy, qx + 1), Y(qy, qx + 1), Z(qy, qx + 1));
            const cv::Vec3d C(X(qy + 1, qx), Y(qy + 1, qx), Z(qy + 1, qx));
            const cv::Vec3d D(X(qy + 1, qx + 1), Y(qy + 1, qx + 1), Z(qy + 1, qx + 1));
            if (!isFinite3(A) || !isFinite3(B) || !isFinite3(C) || !isFinite3(D)) {
                continue;
            }

            if (kBackfaceCullFolds) {
                // Count only front-facing triangles vs. global refN (C4 mitigation)
                total += tri_area3D_culled(A, B, D, refN, kCullDotEps);
                total += tri_area3D_culled(A, D, C, refN, kCullDotEps);
            } else {
                // No culling: fixed diagonal (deterministic) is fine for area
                total += tri_area3D(A, B, D) + tri_area3D(A, D, C);
            }
        }
    }

    return total;
}

}  // namespace

void CWindow::recalcAreaForSegments(const std::vector<std::string>& ids)
{
    if (!fVpkg) {
        return;
    }

    // Linear voxel size (µm/voxel) for cm² conversion
    float voxelsize = 1.0f;
    try {
        if (currentVolume) {
            voxelsize = static_cast<float>(currentVolume->voxelSize());
        }
    } catch (...) {
        voxelsize = 1.0f;
    }
    if (!std::isfinite(voxelsize) || voxelsize <= 0.f) {
        voxelsize = 1.0f;
    }

    int okCount = 0;
    int failCount = 0;
    QStringList updatedIds;
    QStringList skippedIds;

    for (const auto& id : ids) {
        auto sm = fVpkg->getSurface(id);
        if (!sm) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (missing surface)";
            continue;
        }
        auto* surf = sm.get();  // QuadSurface* - sm is already shared_ptr<QuadSurface>

        // --- Load mask (robust multi-page handling) ----------------------
        const std::filesystem::path maskPath = sm->path / "mask.tif";
        if (!std::filesystem::exists(maskPath)) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (no mask.tif)";
            continue;
        }

        cv::Mat1b mask01;
        {
            std::vector<cv::Mat> pages;
            if (cv::imreadmulti(maskPath.string(), pages, cv::IMREAD_UNCHANGED) && !pages.empty()) {
                int best = choose_largest_page(pages);
                if (best < 0) {
                    ++failCount;
                    skippedIds << QString::fromStdString(id) + " (mask pages invalid)";
                    continue;
                }
                binarize_mask(pages[best], mask01);
            } else {
                // Fallback: single-page read
                cv::Mat m = cv::imread(maskPath.string(), cv::IMREAD_UNCHANGED);
                if (m.empty()) {
                    ++failCount;
                    skippedIds << QString::fromStdString(id) + " (mask read error)";
                    continue;
                }
                binarize_mask(m, mask01);
            }
        }
        if (mask01.empty()) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (empty mask)";
            continue;
        }

        // --- Load ORIGINAL quadmesh (no resampling; lower memory) --------
        cv::Mat1f X;
        cv::Mat1f Y;
        cv::Mat1f Z;
        if (!load_tif_as_float(sm->path / "x.tif", X) || !load_tif_as_float(sm->path / "y.tif", Y) ||
            !load_tif_as_float(sm->path / "z.tif", Z)) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (bad or missing x/y/z.tif)";
            continue;
        }
        if (X.size() != Y.size() || X.size() != Z.size() || X.rows < 2 || X.cols < 2) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (xyz size mismatch)";
            continue;
        }

        // --- Area from kept quads --------------
        double area_vx2 = 0.0;
        try {
            area_vx2 = area_from_mesh_and_mask(X, Y, Z, mask01);
        } catch (...) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (area compute error)";
            continue;
        }
        if (!std::isfinite(area_vx2)) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (non-finite area)";
            continue;
        }

        // --- Convert voxel^2 → cm^2 -----------------------------------------
        const double area_cm2 = area_vx2 * static_cast<double>(voxelsize) * static_cast<double>(voxelsize) / 1e8;
        if (!std::isfinite(area_cm2)) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (non-finite cm²)";
            continue;
        }

        // --- Persist & UI update --------------------------------------------
        try {
            if (!surf->meta) {
                surf->meta = std::make_unique<nlohmann::json>();
            }
            (*surf->meta)["area_vx2"] = area_vx2;
            (*surf->meta)["area_cm2"] = area_cm2;
            (*surf->meta)["date_last_modified"] = get_surface_time_str();
            surf->save_meta();
            okCount++;
            updatedIds << QString::fromStdString(id);
        } catch (...) {
            ++failCount;
            skippedIds << QString::fromStdString(id) + " (meta save failed)";
            continue;
        }

        // Update the Surfaces tree (Area column)
        QTreeWidgetItemIterator it(treeWidgetSurfaces);
        while (*it) {
            if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == id) {
                (*it)->setText(2, QString::number(area_cm2, 'f', 3));
                break;
            }
            ++it;
        }
    }

    if (okCount > 0) {
        statusBar()->showMessage(tr("Recalculated area (triangulated kept quads) for %1 segment(s).").arg(okCount),
                                 5000);
    }
    if (failCount > 0) {
        QMessageBox::warning(this,
                             tr("Area Recalculation"),
                             tr("Updated: %1\nSkipped: %2\n\n%3")
                                 .arg(okCount)
                                 .arg(failCount)
                                 .arg(skippedIds.join("\n")));
    }
}
