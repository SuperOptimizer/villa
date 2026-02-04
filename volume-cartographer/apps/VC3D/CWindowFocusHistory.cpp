/**
 * @file CWindowFocusHistory.cpp
 * @brief Focus history navigation extracted from CWindow
 *
 * This file contains methods for managing focus history,
 * allowing navigation back/forward through previously visited locations.
 * Extracted from CWindow.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"

#include "CSurfaceCollection.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>

void CWindow::recordFocusHistory(const POI& poi)
{
    if (_navigatingFocusHistory) {
        return;
    }

    FocusHistoryEntry entry;
    entry.position = poi.p;
    entry.normal = poi.n;
    entry.surfaceId = poi.surfaceId;

    if (_focusHistoryIndex >= 0 &&
        _focusHistoryIndex < static_cast<int>(_focusHistory.size())) {
        const auto& current = _focusHistory[_focusHistoryIndex];
        const float positionDelta = cv::norm(current.position - entry.position);
        const float normalDelta = cv::norm(current.normal - entry.normal);
        if (positionDelta < 1e-4f && normalDelta < 1e-4f && current.surfaceId == entry.surfaceId) {
            return;
        }
    }

    if (_focusHistoryIndex >= 0 &&
        _focusHistoryIndex + 1 < static_cast<int>(_focusHistory.size())) {
        _focusHistory.erase(_focusHistory.begin() + _focusHistoryIndex + 1,
                            _focusHistory.end());
    }

    _focusHistory.push_back(entry);

    if (_focusHistory.size() > 10) {
        _focusHistory.pop_front();
        if (_focusHistoryIndex > 0) {
            --_focusHistoryIndex;
        }
    }

    _focusHistoryIndex = static_cast<int>(_focusHistory.size()) - 1;
}

bool CWindow::stepFocusHistory(int direction)
{
    if (_focusHistory.empty() || direction == 0 || _focusHistoryIndex < 0) {
        return false;
    }

    const int lastIndex = static_cast<int>(_focusHistory.size()) - 1;
    int targetIndex = _focusHistoryIndex + direction;
    targetIndex = std::max(0, std::min(targetIndex, lastIndex));

    if (targetIndex == _focusHistoryIndex) {
        return false;
    }

    _focusHistoryIndex = targetIndex;
    _navigatingFocusHistory = true;
    const auto& entry = _focusHistory[_focusHistoryIndex];
    centerFocusAt(entry.position, entry.normal, entry.surfaceId, false);
    _navigatingFocusHistory = false;
    return true;
}

bool CWindow::centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId, bool addToHistory)
{
    if (!_surf_col) {
        return false;
    }

    POI* focus = _surf_col->poi("focus");
    if (!focus) {
        focus = new POI;
    }

    focus->p = position;
    if (cv::norm(normal) > 0.0) {
        focus->n = normal;
    }
    if (!sourceId.empty()) {
        focus->surfaceId = sourceId;
    } else if (focus->surfaceId.empty()) {
        focus->surfaceId = "segmentation";
    }

    _surf_col->setPOI("focus", focus);

    if (addToHistory) {
        recordFocusHistory(*focus);
    }

    // Get surface for orientation - look up by ID
    Surface* orientationSource = _surf_col->surfaceRaw(focus->surfaceId);
    if (!orientationSource) {
        orientationSource = _surf_col->surfaceRaw("segmentation");
    }
    applySlicePlaneOrientation(orientationSource);

    return true;
}

bool CWindow::centerFocusOnCursor()
{
    if (!_surf_col) {
        return false;
    }

    POI* cursor = _surf_col->poi("cursor");
    if (!cursor) {
        return false;
    }

    return centerFocusAt(cursor->p, cursor->n, cursor->surfaceId, true);
}
