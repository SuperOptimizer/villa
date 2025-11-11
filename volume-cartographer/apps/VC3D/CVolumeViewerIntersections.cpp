#include "CVolumeViewer.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/core/util/Surface.hpp"

#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QPainterPath>
#include <QPen>
#include <QColor>

#include <opencv2/core.hpp>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

// Color constants for segmentation intersection rendering
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

void CVolumeViewer::invalidateIntersect(const std::string &name)
{
    if (!name.size() || name == _surf_name) {
        for(auto &pair : _intersect_items) {
            for(auto &item : pair.second) {
                fScene->removeItem(item);
                delete item;
            }
        }
        _intersect_items.clear();
    }
    else if (_intersect_items.count(name)) {
        for(auto &item : _intersect_items[name]) {
            fScene->removeItem(item);
            delete item;
        }
        _intersect_items.erase(name);
    }
}

void CVolumeViewer::onIntersectionChanged(std::string a, std::string b, Intersection *intersection)
{
    if (_ignore_intersect_change && intersection == _ignore_intersect_change)
        return;

    if (!_intersect_tgts.count(a) || !_intersect_tgts.count(b))
        return;

    //FIXME fix segmentation vs visible_segmentation naming and usage ..., think about dependency chain ..
    if (a == _surf_name || (_surf_name == "segmentation" && a == "visible_segmentation"))
        invalidateIntersect(b);
    else if (b == _surf_name || (_surf_name == "segmentation" && b == "visible_segmentation"))
        invalidateIntersect(a);

    renderIntersections();
}

void CVolumeViewer::setIntersects(const std::set<std::string> &set)
{
    _intersect_tgts = set;
    renderIntersections();
}

void CVolumeViewer::setIntersectionOpacity(float opacity)
{
    _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f);
    for (auto& pair : _intersect_items) {
        for (auto* item : pair.second) {
            if (item) {
                item->setOpacity(_intersectionOpacity);
            }
        }
    }
}

void CVolumeViewer::renderIntersections()
{
    auto total_start = std::chrono::high_resolution_clock::now();

    if (!volume || !volume->zarrDataset() || !_surf) {
        return;
    }

    std::vector<std::string> remove;
    for (auto &pair : _intersect_items)
        if (!_intersect_tgts.count(pair.first)) {
            for(auto &item : pair.second) {
                fScene->removeItem(item);
                delete item;
            }
            remove.push_back(pair.first);
        }
    for(auto key : remove)
        _intersect_items.erase(key);

    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);


    if (plane) {
        cv::Rect plane_roi = {curr_img_area.x()/_scale, curr_img_area.y()/_scale, curr_img_area.width()/_scale, curr_img_area.height()/_scale};

        // Clear all cached intersections for plane surfaces since they depend on the view ROI
        // When the view changes (pan/zoom), we need to recompute all intersections
        for (auto &pair : _intersect_items) {
            for (auto &item : pair.second) {
                fScene->removeItem(item);
                delete item;
            }
        }
        _intersect_items.clear();

        // Build candidate list from all targets
        std::vector<std::string> intersect_cands;
        for (auto key : _intersect_tgts) {
            if (dynamic_cast<QuadSurface*>(_surf_col->surface(key))) {
                intersect_cands.push_back(key);
            }
        }

        std::vector<std::vector<std::vector<cv::Vec3f>>> intersections(intersect_cands.size());

        auto compute_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];
            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));

            std::vector<std::vector<cv::Vec2f>> xy_seg_;
            // Use min_tries=1000 for all segments to ensure we find intersections
            find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi);
        }

        auto compute_end = std::chrono::high_resolution_clock::now();
        double compute_time = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();

        auto render_start = std::chrono::high_resolution_clock::now();
        std::hash<std::string> str_hasher;

        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];

            if (!intersections.size()) {
                _intersect_items[key] = {};
                continue;
            }

            if (!intersections[n].size()) {
                _intersect_items[key] = {};
                continue;
            }

            // Generate deterministic color from segment name using hash
            size_t hash = str_hasher(key);

            // Use hash bits directly to generate RGB values (more deterministic than srand)
            int r = 100 + ((hash >> 0) % 156);   // 100-255
            int g = 100 + ((hash >> 8) % 156);   // 100-255
            int b = 100 + ((hash >> 16) % 156);  // 100-255

            // Make one channel brighter to ensure distinct colors
            int prim = hash % 3;
            if (prim == 0) r = 200 + (hash % 56);  // 200-255
            if (prim == 1) g = 200 + (hash % 56);
            if (prim == 2) b = 200 + (hash % 56);

            QColor col(r, g, b);
            float width = 6;  // 3x thicker than original 2
            int z_value = 5;

            // Check if this segment is the currently active "segmentation" surface
            // The segment might have a different name (e.g., "auto_grown_...") but be aliased as "segmentation"
            Surface* seg_surface = _surf_col->surface("segmentation");
            Surface* current_surface = _surf_col->surface(key);
            bool is_current_segmentation = (seg_surface && current_surface && seg_surface == current_surface);

            if (is_current_segmentation && (_surf_name == "seg yz" || _surf_name == "seg xz" ||
                                           _surf_name.find("plane") != std::string::npos)) {
                // Assign special colors for segmentation in plane viewers
                col =
                    (_surf_name == "seg yz"   ? COLOR_SEG_YZ
                     : _surf_name == "seg xz" ? COLOR_SEG_XZ
                                              : COLOR_SEG_XY);
                width = 9;  // 3x thicker than original 3
                z_value = 20;
            }


            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(intersect_cands[n]));
            std::vector<QGraphicsItem*> items;

            int len = 0;
            for (auto seg : intersections[n]) {
                QPainterPath path;

                bool first = true;
                for (auto wp : seg)
                {
                    len++;
                    cv::Vec3f p = plane->project(wp, 1.0, _scale);

                    if (first)
                        path.moveTo(p[0],p[1]);
                    else
                        path.lineTo(p[0],p[1]);
                    first = false;
                }
                auto item = fGraphicsView->scene()->addPath(path, QPen(col, width));
                item->setZValue(z_value);
                item->setOpacity(_intersectionOpacity);
                items.push_back(item);
            }
            _intersect_items[key] = items;
            _ignore_intersect_change = new Intersection({intersections[n]});
            _surf_col->setIntersection(_surf_name, key, _ignore_intersect_change);
            _ignore_intersect_change = nullptr;
        }

        auto render_end = std::chrono::high_resolution_clock::now();
        double render_time = std::chrono::duration<double, std::milli>(render_end - render_start).count();

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "RENDER PROFILE: compute=" << std::fixed << std::setprecision(1) << compute_time
                  << "ms, qt_render=" << render_time
                  << "ms, TOTAL=" << total_time << "ms" << std::endl;
    }
    else if (_surf_name == "segmentation" /*&& dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"))*/) {
        // QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"));

        //TODO make configurable, for now just show everything!
        std::vector<std::pair<std::string,std::string>> intersects = _surf_col->intersections("segmentation");
        for(auto pair : intersects) {
            std::string key = pair.first;
            if (key == "segmentation")
                key = pair.second;

            if (_intersect_items.count(key) || !_intersect_tgts.count(key)) {
                continue;
            }

            std::unordered_map<cv::Vec3f,cv::Vec3f,vec3f_hash> location_cache;
            std::vector<cv::Vec3f> src_locations;

            for (auto seg : _surf_col->intersection(pair.first, pair.second)->lines)
                for (auto wp : seg)
                    src_locations.push_back(wp);

#pragma omp parallel
            {
                // SurfacePointer *ptr = crop->pointer();
                auto ptr = _surf->pointer();
#pragma omp for
                for (auto wp : src_locations) {
                    // float res = crop->pointTo(ptr, wp, 2.0, 100);
                    // cv::Vec3f p = crop->loc(ptr)*_ds_scale + cv::Vec3f(_vis_center[0],_vis_center[1],0);
                    float res = _surf->pointTo(ptr, wp, 2.0, 100);
                    cv::Vec3f p = _surf->loc(ptr)*_scale ;//+ cv::Vec3f(_vis_center[0],_vis_center[1],0);
                    //FIXME still happening?
                    if (res >= 2.0)
                        p = {-1,-1,-1};
                        // std::cout << "WARNING pointTo() high residual in renderIntersections()" << std::endl;
#pragma omp critical
                    location_cache[wp] = p;
                }
            }

            std::vector<QGraphicsItem*> items;
            int line_count = 0;
            for (auto seg : _surf_col->intersection(pair.first, pair.second)->lines) {
                line_count++;
                QPainterPath path;

                bool first = true;
                int point_count = 0;
                for (auto wp : seg)
                {
                    cv::Vec3f p = location_cache[wp];

                    if (p[0] == -1)
                        continue;

                    point_count++;
                    if (first)
                        path.moveTo(p[0],p[1]);
                    else
                        path.lineTo(p[0],p[1]);
                    first = false;
                }
                QColor col = (key == "seg yz" ? COLOR_SEG_YZ : COLOR_SEG_XZ);
                auto item = fGraphicsView->scene()->addPath(path, QPen(col, 6));  // 3x thicker
                item->setZValue(5);
                item->setOpacity(_intersectionOpacity);
                items.push_back(item);
            }
            _intersect_items[key] = items;
        }
    }
}
