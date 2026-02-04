/**
 * @file SegmentationModule_ApprovalMask.cpp
 * @brief Approval mask methods extracted from SegmentationModule
 *
 * This file contains methods for approval mask display and editing.
 * Extracted from SegmentationModule.cpp to improve parallel compilation.
 */

#include "SegmentationModule.hpp"

#include "ApprovalMaskBrushTool.hpp"
#include "CSurfaceCollection.hpp"
#include "SegmentationCorrections.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QLoggingCategory>
#include <QTimer>

Q_DECLARE_LOGGING_CATEGORY(lcSegModule)

void SegmentationModule::setShowApprovalMask(bool enabled)
{
    if (_showApprovalMask == enabled) {
        return;
    }

    _showApprovalMask = enabled;
    qCInfo(lcSegModule) << "=== Show Approval Mask:" << (enabled ? "ENABLED" : "DISABLED") << "===";

    if (_showApprovalMask) {
        // Showing approval mask - load it for display
        QuadSurface* surface = nullptr;
        std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
        if (_editManager && _editManager->hasSession()) {
            qCInfo(lcSegModule) << "  Loading approval mask (has active session)";
            surface = _editManager->baseSurface().get();
        } else if (_surfaces) {
            qCInfo(lcSegModule) << "  Loading approval mask (from surfaces collection)";
            surfaceHolder = _surfaces->surface("segmentation");
            surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
        }

        if (surface && _overlay) {
            _overlay->loadApprovalMaskImage(surface);
            qCInfo(lcSegModule) << "  Loaded approval mask into QImage";
        }
    }

    refreshOverlay();
}

void SegmentationModule::onActiveSegmentChanged(QuadSurface* newSurface)
{
    qCInfo(lcSegModule) << "Active segment changed";

    // Flush any pending approval mask saves and clear images BEFORE turning off editing
    // loadApprovalMaskImage(nullptr) does both:
    // 1. Saves pending changes to _approvalSaveSurface (the previous segment)
    // 2. Clears the mask images so subsequent saveApprovalMaskToDisk() has nothing to save
    // This prevents the old mask from being incorrectly saved to the new segment
    if (_overlay) {
        _overlay->loadApprovalMaskImage(nullptr);
    }

    // Turn off any approval mask editing when switching segments
    if (isEditingApprovalMask()) {
        qCInfo(lcSegModule) << "  Turning off approval mask editing";
        if (_editApprovedMask) {
            setEditApprovedMask(false);
            if (_widget) {
                _widget->setEditApprovedMask(false);
            }
        }
        if (_editUnapprovedMask) {
            setEditUnapprovedMask(false);
            if (_widget) {
                _widget->setEditUnapprovedMask(false);
            }
        }
    }

    // Sync show approval mask state from widget (handles restored settings case)
    if (_widget && _widget->showApprovalMask() != _showApprovalMask) {
        qCInfo(lcSegModule) << "  Syncing showApprovalMask from widget:" << _widget->showApprovalMask();
        _showApprovalMask = _widget->showApprovalMask();
    }

    // Check if new surface has an approval mask
    bool hasApprovalMask = false;
    if (newSurface) {
        cv::Mat approvalChannel = newSurface->channel("approval", SURF_CHANNEL_NORESIZE);
        hasApprovalMask = !approvalChannel.empty();
        qCInfo(lcSegModule) << "  New surface has approval mask:" << hasApprovalMask;
    }

    if (_showApprovalMask) {
        if (hasApprovalMask && newSurface && _overlay) {
            // Load the new surface's approval mask
            qCInfo(lcSegModule) << "  Loading approval mask for new surface";
            _overlay->loadApprovalMaskImage(newSurface);
        } else {
            // No approval mask on new surface - turn off show mode
            qCInfo(lcSegModule) << "  No approval mask on new surface, turning off show mode";
            _showApprovalMask = false;
            if (_widget) {
                _widget->setShowApprovalMask(false);
            }
            if (_overlay) {
                _overlay->loadApprovalMaskImage(nullptr);  // Clear the mask
            }
        }
    }

    // Save corrections for old segment and load for new segment
    if (_pointCollection) {
        // Save pending corrections for the old segment
        if (_correctionsSaveTimer && _correctionsSaveTimer->isActive()) {
            _correctionsSaveTimer->stop();
        }
        if (!_correctionsSegmentPath.empty()) {
            qCInfo(lcSegModule) << "  Saving correction points for previous segment";
            _pointCollection->saveToSegmentPath(_correctionsSegmentPath);
        }

        // Load corrections for new segment
        if (newSurface && !newSurface->path.empty()) {
            qCInfo(lcSegModule) << "  Loading correction points for new segment:"
                                << QString::fromStdString(newSurface->path.string());
            _pointCollection->loadFromSegmentPath(newSurface->path);
            _correctionsSegmentPath = newSurface->path;
        } else {
            // No valid path - clear anchored collections and path
            qCInfo(lcSegModule) << "  No segment path, clearing anchored corrections";
            _pointCollection->loadFromSegmentPath({});  // Clears anchored collections
            _correctionsSegmentPath.clear();
        }
    }

    refreshOverlay();
}

void SegmentationModule::setEditApprovedMask(bool enabled)
{
    if (_editApprovedMask == enabled) {
        return;
    }

    // If enabling, ensure unapproved mode is off (mutual exclusion)
    if (enabled && _editUnapprovedMask) {
        setEditUnapprovedMask(false);
    }

    const bool wasEditing = isEditingApprovalMask();
    _editApprovedMask = enabled;
    qCInfo(lcSegModule) << "=== Edit Approved Mask:" << (enabled ? "ENABLED" : "DISABLED") << "===";

    if (_editApprovedMask) {
        // Entering approval mask editing mode (approve)
        qCInfo(lcSegModule) << "  Activating approval brush tool (approve mode)";
        if (_approvalTool) {
            _approvalTool->setActive(true);
            _approvalTool->setPaintMode(ApprovalMaskBrushTool::PaintMode::Approve);

            // Set surface on approval tool - prefer surface from collection since it has
            // the most up-to-date approval mask (preserved after tracer growth)
            QuadSurface* surface = nullptr;
            std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
            if (_surfaces) {
                surfaceHolder = _surfaces->surface("segmentation");
                surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
            }
            if (!surface && _editManager && _editManager->hasSession()) {
                surface = _editManager->baseSurface().get();
            }

            if (surface) {
                _approvalTool->setSurface(surface);
                // Reload approval mask image to ensure dimensions match current surface
                if (_overlay) {
                    _overlay->loadApprovalMaskImage(surface);
                }
            }
        }

        // Deactivate regular editing tools
        clearLineDragStroke();
        stopAllPushPull();
    } else if (!isEditingApprovalMask()) {
        // Exiting all approval mask editing - save to disk
        qCInfo(lcSegModule) << "  Deactivating approval brush tool and saving";
        if (_approvalTool) {
            _approvalTool->setActive(false);
        }

        // Save changes to disk when exiting edit mode
        if (wasEditing) {
            saveApprovalMaskToDisk();
        }
    }

    refreshOverlay();
}

void SegmentationModule::setEditUnapprovedMask(bool enabled)
{
    if (_editUnapprovedMask == enabled) {
        return;
    }

    // If enabling, ensure approved mode is off (mutual exclusion)
    if (enabled && _editApprovedMask) {
        setEditApprovedMask(false);
    }

    const bool wasEditing = isEditingApprovalMask();
    _editUnapprovedMask = enabled;
    qCInfo(lcSegModule) << "=== Edit Unapproved Mask:" << (enabled ? "ENABLED" : "DISABLED") << "===";

    if (_editUnapprovedMask) {
        // Entering approval mask editing mode (unapprove)
        qCInfo(lcSegModule) << "  Activating approval brush tool (unapprove mode)";
        if (_approvalTool) {
            _approvalTool->setActive(true);
            _approvalTool->setPaintMode(ApprovalMaskBrushTool::PaintMode::Unapprove);

            // Set surface on approval tool - prefer surface from collection since it has
            // the most up-to-date approval mask (preserved after tracer growth)
            QuadSurface* surface = nullptr;
            std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
            if (_surfaces) {
                surfaceHolder = _surfaces->surface("segmentation");
                surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
            }
            if (!surface && _editManager && _editManager->hasSession()) {
                surface = _editManager->baseSurface().get();
            }

            if (surface) {
                _approvalTool->setSurface(surface);
                // Reload approval mask image to ensure dimensions match current surface
                if (_overlay) {
                    _overlay->loadApprovalMaskImage(surface);
                }
            }
        }

        // Deactivate regular editing tools
        clearLineDragStroke();
        stopAllPushPull();
    } else if (!isEditingApprovalMask()) {
        // Exiting all approval mask editing - save to disk
        qCInfo(lcSegModule) << "  Deactivating approval brush tool and saving";
        if (_approvalTool) {
            _approvalTool->setActive(false);
        }

        // Save changes to disk when exiting edit mode
        if (wasEditing) {
            saveApprovalMaskToDisk();
        }
    }

    refreshOverlay();
}

void SegmentationModule::saveApprovalMaskToDisk()
{
    qCInfo(lcSegModule) << "Saving approval mask to disk...";

    QuadSurface* surface = nullptr;
    std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
    if (_editManager && _editManager->hasSession()) {
        surface = _editManager->baseSurface().get();
    } else if (_surfaces) {
        surfaceHolder = _surfaces->surface("segmentation");
        surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
    }

    if (_overlay && surface) {
        _overlay->saveApprovalMaskToSurface(surface);
        emit statusMessageRequested(tr("Saved approval mask."), kStatusShort);
        qCInfo(lcSegModule) << "  Approval mask saved to disk";

        // Emit signal so CWindow can mark this segment as recently edited
        // (to prevent inotify from triggering unwanted removals/reloads)
        if (!surface->id.empty()) {
            emit approvalMaskSaved(surface->id);
        }
    }
}

void SegmentationModule::setApprovalMaskBrushRadius(float radiusSteps)
{
    _approvalMaskBrushRadius = std::max(1.0f, radiusSteps);
}

void SegmentationModule::setApprovalBrushDepth(float depth)
{
    _approvalBrushDepth = std::clamp(depth, 1.0f, 500.0f);
}

void SegmentationModule::setApprovalBrushColor(const QColor& color)
{
    if (color.isValid()) {
        _approvalBrushColor = color;
    }
}

void SegmentationModule::undoApprovalStroke()
{
    qCInfo(lcSegModule) << "Undoing last approval stroke...";
    if (!_overlay) {
        qCWarning(lcSegModule) << "  No overlay controller available";
        return;
    }

    if (!_overlay->canUndoApprovalMaskPaint()) {
        qCInfo(lcSegModule) << "  Nothing to undo";
        emit statusMessageRequested(tr("Nothing to undo."), kStatusShort);
        return;
    }

    if (_overlay->undoLastApprovalMaskPaint()) {
        refreshOverlay();
        emit statusMessageRequested(tr("Undid last approval stroke."), kStatusShort);
        qCInfo(lcSegModule) << "  Approval stroke undone";
    }
}
