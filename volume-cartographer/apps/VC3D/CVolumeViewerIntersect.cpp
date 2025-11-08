#include "CVolumeViewer.hpp"

// Intersection rendering is currently stubbed out for future rewrite

void CVolumeViewer::invalidateIntersect(const std::string &name)
{
    // Stub: Clear all intersection items
    for(auto &pair : _intersect_items) {
        for(auto &item : pair.second) {
            if (fScene && item) {
                fScene->removeItem(item);
                delete item;
            }
        }
    }
    _intersect_items.clear();
}


void CVolumeViewer::onIntersectionChanged(std::string a, std::string b, Intersection *intersection)
{
    // Stub: Do nothing
}

void CVolumeViewer::setIntersects(const std::set<std::string> &set)
{
    // Stub: Just store the set
    _intersect_tgts = set;
}

void CVolumeViewer::setIntersectionOpacity(float opacity)
{
    // Stub: Just store the value
    _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f);
}

void CVolumeViewer::setIntersectionLineWidth(float width)
{
    // Stub: Just store the value
    _intersectionLineWidth = std::clamp(width, 1.0f, 10.0f);
}

void CVolumeViewer::renderIntersections()
{
    // Stub: Do nothing
}
