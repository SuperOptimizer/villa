#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"

#include <QtWidgets>
#include <functional>
#include <opencv2/core.hpp>
#include <chrono>
#include <iostream>

#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

// Timing helper macros
#define TIME_START(name) auto name##_start = std::chrono::high_resolution_clock::now()
#define TIME_END(name) do { \
    auto name##_end = std::chrono::high_resolution_clock::now(); \
    auto name##_duration = std::chrono::duration_cast<std::chrono::microseconds>(name##_end - name##_start).count(); \
    std::cout << "[TIMING] " << #name << ": " << (name##_duration / 1000.0) << " ms" << std::endl; \
} while(0)

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
    // Only rebuild index if the set actually changed
    if (_intersect_tgts != set) {
        _intersect_tgts = set;
        rebuildIntersectionIndex();
    }

    renderIntersections();
}

void CVolumeViewer::rebuildIntersectionIndex()
{
    TIME_START(rebuild_index_total);

    std::cout << "[INTERSECT] rebuildIntersectionIndex: " << _intersect_tgts.size() << " targets" << std::endl;

    // Log if "segmentation" is in the target set
    bool has_segmentation = _intersect_tgts.count("segmentation") > 0;
    std::cout << "[INTERSECT] 'segmentation' in targets: " << (has_segmentation ? "YES" : "NO") << std::endl;

    // Clear the index
    _intersect_index = MultiSurfaceIndex(100.0f); // 100 voxel cell size
    _intersect_index_map.clear();

    if (!_surf_col) {
        std::cout << "[INTERSECT] No surface collection" << std::endl;
        return;
    }

    // Log all surfaces in the collection for debugging
    std::cout << "[INTERSECT] Listing surfaces in collection:" << std::endl;
    for (const auto& target_name : _intersect_tgts) {
        Surface* s = _surf_col->surface(target_name);
        std::cout << "[INTERSECT]   '" << target_name << "': " << (s ? "EXISTS" : "NULL") << std::endl;
    }

    // Build index from all intersection targets (limit to 100)
    TIME_START(build_index);
    int idx = 0;
    for (const auto& key : _intersect_tgts) {
        if (idx >= 100) {
            break;
        }

        TIME_START(get_surface);
        Surface* base_surf = _surf_col->surface(key);
        QuadSurface* surf = dynamic_cast<QuadSurface*>(base_surf);
        TIME_END(get_surface);

        if (surf) {
            TIME_START(add_patch);
            _intersect_index.addPatch(idx, surf);
            TIME_END(add_patch);
            _intersect_index_map[key] = idx;
            if (key == "segmentation") {
                std::cout << "[INTERSECT] Added 'segmentation' to index at idx=" << idx << std::endl;
            }
            idx++;
        } else {
            if (key == "segmentation") {
                std::cout << "[INTERSECT] WARNING: Could not get QuadSurface for 'segmentation'" << std::endl;
                std::cout << "[INTERSECT]   base_surf=" << base_surf << ", surf=" << surf << std::endl;
                if (base_surf) {
                    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(base_surf);
                    std::cout << "[INTERSECT]   Is PlaneSurface? " << (plane ? "YES" : "NO") << std::endl;
                }
            }
        }
    }
    TIME_END(build_index);

    std::cout << "[INTERSECT] Built index with " << idx << " segments" << std::endl;
    TIME_END(rebuild_index_total);
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

void CVolumeViewer::setIntersectionLineWidth(float width)
{
    _intersectionLineWidth = std::clamp(width, 1.0f, 10.0f);
    // Trigger re-render to apply new line width
    renderIntersections();
}

void CVolumeViewer::renderIntersections()
{
    TIME_START(render_intersections_total);

    std::cout << "[INTERSECT] renderIntersections called, _surf_name='" << _surf_name << "'" << std::endl;

    if (!volume || !volume->zarrDataset() || !_surf) {
        std::cout << "[INTERSECT] renderIntersections: no volume/dataset/surf" << std::endl;
        return;
    }

    TIME_START(cleanup_old_items);
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
    TIME_END(cleanup_old_items);

    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);


    if (plane) {
        TIME_START(setup_bbox);
        cv::Rect plane_roi = {curr_img_area.x()/_scale, curr_img_area.y()/_scale, curr_img_area.width()/_scale, curr_img_area.height()/_scale};

        cv::Vec3f corner = plane->coord(cv::Vec3f(0,0,0), {plane_roi.x, plane_roi.y, 0.0});
        Rect3D view_bbox = {corner, corner};
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.br().x, plane_roi.y, 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.x, plane_roi.br().y, 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {plane_roi.br().x, plane_roi.br().y, 0}));
        TIME_END(setup_bbox);

        std::vector<std::string> intersect_cands;
        std::set<std::string> candidate_set;

        TIME_START(spatial_query);
        // Query the spatial index with view bbox corners to find candidate segments
        // Calculate bbox center and diagonal for tolerance
        cv::Vec3f bbox_center = {
            (view_bbox.low[0] + view_bbox.high[0]) / 2.0f,
            (view_bbox.low[1] + view_bbox.high[1]) / 2.0f,
            (view_bbox.low[2] + view_bbox.high[2]) / 2.0f
        };
        float bbox_diagonal = cv::norm(view_bbox.high - view_bbox.low);

        // Query the index at the bbox center with tolerance = half diagonal
        std::vector<int> candidate_indices = _intersect_index.getCandidatePatches(bbox_center, bbox_diagonal / 2.0f);
        std::cout << "[INTERSECT] Spatial query returned " << candidate_indices.size() << " candidates" << std::endl;
        TIME_END(spatial_query);

        TIME_START(filter_candidates);
        // Build reverse map: index -> segment name
        std::unordered_map<int, std::string> index_to_name;
        for (const auto& pair : _intersect_index_map) {
            index_to_name[pair.second] = pair.first;
        }

        // Convert indices to segment names and filter by bbox intersection
        std::vector<std::string> intersect_tgts_v;
        int cached_count = 0;
        int bbox_fail_count = 0;
        for (int idx : candidate_indices) {
            auto it = index_to_name.find(idx);
            if (it != index_to_name.end()) {
                const std::string& key = it->second;

                // Skip if already cached
                if (_intersect_items.count(key)) {
                    cached_count++;
                    if (key == "segmentation") {
                        std::cout << "[INTERSECT] 'segmentation' already cached, items: " << _intersect_items[key].size() << std::endl;
                    }
                    continue;
                }

                QuadSurface* segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));
                if (segmentation && intersect(view_bbox, segmentation->bbox())) {
                    intersect_tgts_v.push_back(key);
                    candidate_set.insert(key);
                    if (key == "segmentation") {
                        std::cout << "[INTERSECT] 'segmentation' ADDED to render candidates" << std::endl;
                    }
                } else {
                    bbox_fail_count++;
                    if (key == "segmentation") {
                        std::cout << "[INTERSECT] 'segmentation' FAILED: surf=" << (segmentation ? "valid" : "null")
                                  << ", bbox_intersect=" << (segmentation ? (intersect(view_bbox, segmentation->bbox()) ? "true" : "false") : "N/A") << std::endl;
                    }
                }
            }
        }
        std::cout << "[INTERSECT] Filter results: " << cached_count << " cached, " << bbox_fail_count << " bbox failed" << std::endl;

        // Swap to use the candidate list
        intersect_cands = intersect_tgts_v;
        std::cout << "[INTERSECT] After filtering: " << intersect_cands.size() << " segments need rendering" << std::endl;

        // Log if segmentation is in candidates
        bool seg_in_cands = false;
        for (const auto& cand : intersect_cands) {
            if (cand == "segmentation") {
                seg_in_cands = true;
                break;
            }
        }
        std::cout << "[INTERSECT] 'segmentation' in candidates list: " << (seg_in_cands ? "YES" : "NO") << std::endl;

        TIME_END(filter_candidates);

        std::vector<std::vector<std::vector<cv::Vec3f>>> intersections(intersect_cands.size());

        TIME_START(compute_intersections);
#pragma omp parallel for
        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];
            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));

            std::vector<std::vector<cv::Vec2f>> xy_seg_;
            if (key == "segmentation") {
                find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi, 4/_scale, 1000);
            }
            else
                find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi, 4/_scale);

        }
        TIME_END(compute_intersections);
        std::cout << "[INTERSECT] Computed intersections for " << intersect_cands.size() << " segments" << std::endl;

        // Log intersection counts for each segment
        for (int n = 0; n < intersect_cands.size(); n++) {
            if (intersect_cands[n] == "segmentation") {
                std::cout << "[INTERSECT] 'segmentation' computed " << intersections[n].size() << " intersection line segments" << std::endl;
                if (intersections[n].size() > 0) {
                    int total_pts = 0;
                    for (const auto& seg : intersections[n]) {
                        total_pts += seg.size();
                    }
                    std::cout << "[INTERSECT] 'segmentation' total points: " << total_pts << std::endl;
                }
            }
        }

        std::hash<std::string> str_hasher;

        // Helper lambda to generate deterministic color from UUID hash
        auto generateColorFromUUID = [&str_hasher](const std::string& uuid) -> QColor {
            size_t hash = str_hasher(uuid);

            // Use different parts of hash for R, G, B
            unsigned char r = 100 + ((hash >> 0) % 156);   // 100-255
            unsigned char g = 100 + ((hash >> 8) % 156);   // 100-255
            unsigned char b = 100 + ((hash >> 16) % 156);  // 100-255

            // Make one channel dominant for better visibility
            int prim = (hash >> 24) % 3;
            if (prim == 0) r = 200 + (hash % 56);        // 200-255
            else if (prim == 1) g = 200 + (hash % 56);
            else b = 200 + (hash % 56);

            return QColor(r, g, b);
        };

        TIME_START(render_graphics);
        int total_items = 0;
        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];

            if (!intersections[n].size()) {
                _intersect_items[key] = {};
                if (key == "segmentation") {
                    std::cout << "[INTERSECT] 'segmentation' has 0 intersections, skipping" << std::endl;
                }
                continue;
            }

            if (key == "segmentation") {
                std::cout << "[INTERSECT] 'segmentation' has " << intersections[n].size() << " intersection segments" << std::endl;
            }

            QColor col = generateColorFromUUID(key);
            float width = _intersectionLineWidth;
            int z_value = 5;

            if (key == "segmentation") {
                col =
                    (_surf_name == "seg yz"   ? COLOR_SEG_YZ
                     : _surf_name == "seg xz" ? COLOR_SEG_XZ
                                              : COLOR_SEG_XY);
                width = _intersectionLineWidth * 1.5f;  // Slightly thicker for segmentation
                z_value = 20;
            }


            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(intersect_cands[n]));
            std::vector<QGraphicsItem*> items;

            if (key == "segmentation") {
                int total_points = 0;
                for (const auto& seg : intersections[n]) {
                    total_points += seg.size();
                }
                std::cout << "[INTERSECT] 'segmentation' processing: " << intersections[n].size()
                          << " segments with " << total_points << " total points" << std::endl;
            }

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
                        items.push_back(item);
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
                items.push_back(item);
            }
            _intersect_items[key] = items;
            total_items += items.size();

            if (key == "segmentation") {
                std::cout << "[INTERSECT] 'segmentation' final: created " << items.size() << " graphics items" << std::endl;
            }

            _ignore_intersect_change = new Intersection({intersections[n]});
            _surf_col->setIntersection(_surf_name, key, _ignore_intersect_change);
            _ignore_intersect_change = nullptr;
        }
        TIME_END(render_graphics);
        std::cout << "[INTERSECT] Rendered " << total_items << " graphics items" << std::endl;
    }
    else if (_surf_name == "segmentation" /*&& dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"))*/) {
        // QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"));

        //TODO make configurable, for now just show everything!
        std::vector<std::pair<std::string,std::string>> intersects = _surf_col->intersections("segmentation");
        for(auto pair : intersects) {
            std::string key = pair.first;
            if (key == "segmentation")
                key = pair.second;

            if (_intersect_items.count(key) || !_intersect_tgts.count(key))
                continue;

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
            for (auto seg : _surf_col->intersection(pair.first, pair.second)->lines) {
                QPainterPath path;

                bool first = true;
                cv::Vec3f last = {-1,-1,-1};
                for (auto wp : seg)
                {
                    cv::Vec3f p = location_cache[wp];

                    if (p[0] == -1)
                        continue;

                    if (last[0] != -1 && cv::norm(p-last) >= 8) {
                        auto item = fGraphicsView->scene()->addPath(path, QPen(key == "seg yz" ? COLOR_SEG_YZ: COLOR_SEG_XZ, _intersectionLineWidth * 1.5f));
                        item->setZValue(5);
                        item->setOpacity(_intersectionOpacity);
                        items.push_back(item);
                        first = true;
                    }
                    last = p;

                    if (first)
                        path.moveTo(p[0],p[1]);
                    else
                        path.lineTo(p[0],p[1]);
                    first = false;
                }
                auto item = fGraphicsView->scene()->addPath(path, QPen(key == "seg yz" ? COLOR_SEG_YZ: COLOR_SEG_XZ, _intersectionLineWidth * 1.5f));
                item->setZValue(5);
                item->setOpacity(_intersectionOpacity);
                items.push_back(item);
            }
            _intersect_items[key] = items;
        }
    }

    TIME_END(render_intersections_total);
    std::cout << "[INTERSECT] === renderIntersections() complete ===" << std::endl;
}
