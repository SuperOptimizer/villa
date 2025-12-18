#pragma once

#include "SegmentationTool.hpp"
#include "SegmentationEditManager.hpp"

#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

class CSurfaceCollection;
class CVolumeViewer;
class QuadSurface;
class SegmentationModule;
class SegmentationOverlayController;
class QTimer;

class SegmentationLassoPushPullTool : public SegmentationTool
{
public:
    enum class State
    {
        Idle,            // Not active
        Drawing,         // Drawing the lasso
        SelectionActive  // Lasso complete, waiting for push/pull
    };

    struct LassoSelection
    {
        std::vector<cv::Vec3f> polygonWorld;       // Lasso points in world coords
        std::vector<cv::Vec2f> polygonGrid;        // Lasso points in grid coords (col, row)
        cv::Vec3f centroidWorld{0, 0, 0};          // Centroid of selected vertices
        std::pair<int, int> centroidGrid{0, 0};    // Grid coords of centroid (row, col)
        cv::Vec3f averageNormal{0, 0, 1};          // Average normal of selected region
        std::vector<SegmentationEditManager::DragSample> samples;  // Selected vertices
    };

    SegmentationLassoPushPullTool(SegmentationModule& module,
                                   SegmentationEditManager* editManager,
                                   SegmentationOverlayController* overlay,
                                   CSurfaceCollection* surfaces);

    void setDependencies(SegmentationEditManager* editManager,
                         SegmentationOverlayController* overlay,
                         CSurfaceCollection* surfaces);

    // Lasso drawing interface
    void startLasso(const cv::Vec3f& worldPos);
    void extendLasso(const cv::Vec3f& worldPos, bool forceSample);
    bool finishLasso();  // Returns true if valid selection created

    // Push/pull interface (called when A/D pressed after lasso complete)
    bool startPushPull(int direction);
    void stopPushPull(int direction);
    void stopAllPushPull();
    bool applyPushPullStep();

    // Clear selection and return to idle
    void clearSelection();

    // State queries
    [[nodiscard]] State state() const { return _state; }
    [[nodiscard]] bool isDrawing() const { return _state == State::Drawing; }
    [[nodiscard]] bool hasSelection() const { return _state == State::SelectionActive; }
    [[nodiscard]] bool isPushPullActive() const { return _pushPullActive; }

    // Overlay data access
    [[nodiscard]] const std::vector<cv::Vec3f>& currentStrokePoints() const { return _currentStroke; }
    [[nodiscard]] const LassoSelection& selection() const { return _selection; }

    // SegmentationTool interface
    void cancel() override;
    [[nodiscard]] bool isActive() const override;

    // Configuration
    void setStepMultiplier(float multiplier);
    [[nodiscard]] float stepMultiplier() const { return _stepMultiplier; }

private:
    bool buildSelectionFromLasso();
    void computeCentroidAndNormal();
    bool pointInPolygon(float gridRow, float gridCol) const;
    void ensureTimer();
    bool applyStepInternal();
    void updateSampleBasePositions();

    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    SegmentationOverlayController* _overlay{nullptr};
    CSurfaceCollection* _surfaces{nullptr};

    State _state{State::Idle};

    // Lasso drawing state
    std::vector<cv::Vec3f> _currentStroke;
    cv::Vec3f _lastSample{0, 0, 0};
    bool _hasLastSample{false};
    static constexpr float kSampleSpacing = 2.0f;

    // Selection state
    LassoSelection _selection;

    // Push/pull state
    bool _pushPullActive{false};
    int _pushPullDirection{0};
    QTimer* _timer{nullptr};
    float _stepMultiplier{4.0f};
    bool _undoCaptured{false};
    static constexpr int kPushPullIntervalMs = 16;
};
