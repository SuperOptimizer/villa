#include "CVolumeViewer.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Intersection.hpp"
#include "CSurfaceCollection.hpp"

#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QGraphicsScene>
#include <omp.h>

// Color definitions for intersection lines
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

void CVolumeViewer::renderIntersections()
{
    if (!volume || !volume->zarrDataset() || !_surf)
        return;

    // Clear previous intersection rendering
    clearOverlayGroup("intersections");

    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);

    if (plane) {
        cv::Rect plane_roi = {curr_img_area.x()/_scale, curr_img_area.y()/_scale, curr_img_area.width()/_scale, curr_img_area.height()/_scale};

        cv::Vec3f corner = plane->coord(cv::Vec3f(0,0,0), {plane_roi.x, plane_roi.y, 0.0});
        Rect3D view_bbox = {corner, corner};
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.br().x, plane_roi.y, 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.x, plane_roi.br().y, 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.br().x, plane_roi.br().y, 0}));

        std::vector<std::string> intersect_cands;
        std::set<Surface*> seen_surfaces;

        for (auto key : _intersect_tgts) {
            if (dynamic_cast<QuadSurface*>(_surf_col->surface(key))) {
                QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));

                // Skip if we've already seen this surface pointer (same surface, different name)
                if (seen_surfaces.count(segmentation))
                    continue;
                seen_surfaces.insert(segmentation);

                if (intersect(view_bbox, segmentation->bbox())) {
                    intersect_cands.push_back(key);
                }
            }
        }

        std::vector<std::vector<std::vector<cv::Vec3f>>> intersections(intersect_cands.size());
        std::hash<std::string> str_hasher;

        // NOTE: Cannot use parallel for here because srand()/rand() are not thread-safe
        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];
            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));

            // Set deterministic seed BEFORE computing intersections
            size_t seed = str_hasher(key);
            srand(seed);

            std::vector<std::vector<cv::Vec2f>> xy_seg_;
            if (key == "segmentation") {
                find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi, 4/_scale, 1000);
            }
            else
                find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi, 4/_scale);

        }

        std::vector<QGraphicsItem*> all_items;

        std::cout << "Rendering " << intersect_cands.size() << " surfaces in view " << _surf_name << std::endl;

        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];

            if (!intersections.size()) {
                continue;
            }

            std::cout << "  Surface '" << key << "' has " << intersections[n].size() << " intersection segments" << std::endl;

            size_t seed = str_hasher(key);
            srand(seed);

            int prim = rand() % 3;
            cv::Vec3i cvcol = {100 + rand() % 255, 100 + rand() % 255, 100 + rand() % 255};
            cvcol[prim] = 200 + rand() % 55;

            QColor col(cvcol[0],cvcol[1],cvcol[2]);
            float width = 2;
            int z_value = 5;

            if (key == "segmentation") {
                col =
                    (_surf_name == "seg yz"   ? COLOR_SEG_YZ
                     : _surf_name == "seg xz" ? COLOR_SEG_XZ
                                              : COLOR_SEG_XY);
                width = 3;
                z_value = 20;
            }

            std::cout << "    Color: " << col.name().toStdString() << " for view " << _surf_name << std::endl;

            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(intersect_cands[n]));

            int len = 0;
            for (auto seg : intersections[n]) {
                QPainterPath path;

                bool first = true;
                cv::Vec3f last = {-1,-1,-1};
                for (auto wp : seg)
                {
                    len++;
                    cv::Vec3f p = plane->project(wp, 1.0, _scale);

                    if (last[0] != -1 && cv::norm(p-last) >= 8) {
                        auto item = fGraphicsView->scene()->addPath(path, QPen(col, width));
                        item->setZValue(z_value);
                        item->setOpacity(_intersectionOpacity);
                        all_items.push_back(item);
                        first = true;
                    }
                    last = p;

                    if (first)
                        path.moveTo(p[0],p[1]);
                    else
                        path.lineTo(p[0],p[1]);
                    first = false;
                }
                auto item = fGraphicsView->scene()->addPath(path, QPen(col, width));
                item->setZValue(z_value);
                item->setOpacity(_intersectionOpacity);
                all_items.push_back(item);
            }
        }

        setOverlayGroup("intersections", all_items);
    }
}
