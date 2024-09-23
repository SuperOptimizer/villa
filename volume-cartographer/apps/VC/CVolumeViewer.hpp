// CVolumeViewer.h
// Chao Du 2015 April
#pragma once

#include <QtWidgets>
#include <opencv2/core/core.hpp>

#include "vc/core/types/VolumePkg.hpp"

class PlaneCoords;
class ChunkCache;
class ControlPointSegmentator;
class QGraphicsScene;

namespace ChaoVis
{

class CVolumeViewerView;
class SegmentationStruct;

class CVolumeViewer : public QWidget
{
    Q_OBJECT

public:
    enum EViewState {
        ViewStateEdit,  // edit mode
        ViewStateDraw,  // draw mode
        ViewStateIdle   // idle mode
    };

    QPushButton* fNextBtn;
    QPushButton* fPrevBtn;
    CVolumeViewer(QWidget* parent = 0);
    ~CVolumeViewer(void);
    virtual void SetButtonsEnabled(bool state);

    void SetViewState(EViewState nViewState) { fViewState = nViewState; }
    EViewState GetViewState(void) { return fViewState; }
    void Reset();

    virtual void SetImage(const QImage& nSrc);
    void SetImageIndex(int nImageIndex);
    void SetNumSlices(int num);
    void SetRotation(int degress);
    void Rotate(int delta);
    void ResetRotation();
    void setCache(ChunkCache *cache);
    void loadSlice();
    void setSlice(PlaneCoords *slice);
    cv::Mat getCoordSlice();
    void renderVisible(bool force = false);
    void currRoi(cv::Rect &roi, float &render_scale, float &coord_scale, int &sd_idx) const;
    cv::Mat render_area();
    void addIntersectVisSlice(PlaneCoords *slice_);
    void setSegTool(ControlPointSegmentator *tool);
    
    CVolumeViewerView* fGraphicsView;

protected:
    // bool eventFilter(QObject* watched, QEvent* event);

public slots:
    void OnVolumeChanged(volcart::Volume::Pointer vol);
    void onVolumeClicked(QPointF scene_loc,Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void OnSliceChanged();
    void onScrolled();
    void onZoom(int steps);

signals:
    void SendSignalSliceShift(int shift, int axis);
    void SendSignalStatusMessageAvailable(QString text, int timeout);
    void sendVolumeClicked(cv::Vec3f vol_loc,Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void sendShiftNormal(cv::Vec3f step);

protected:
    void ScaleImage(double nFactor);
    void CenterOn(const QPointF& point);

protected:
    // widget components
    // CVolumeViewerView* fGraphicsView;
    QGraphicsScene* fScene;

    QLabel* fCanvas;
    QScrollArea* fScrollArea;
    QPushButton* fZoomInBtn;
    QPushButton* fZoomOutBtn;
    QPushButton* fResetBtn;
    QSpinBox* fImageRotationSpin;
    QHBoxLayout* fButtonsLayout;
    // QComboBox* fAxisCombo;

    // data
    EViewState fViewState;
    QImage* fImgQImage;
    double fScaleFactor;
    int sliceIndexToolStart{-1};
    int fScanRange;  // how many slices a mouse wheel step will jump
    // Required to be able to reset the rotation without also resetting the scaling
    int currentRotation{0};

    // user settings
    bool fCenterOnZoomEnabled;
    int fScrollSpeed{-1};
    bool fSkipImageFormatConv;

    QGraphicsPixmapItem* fBaseImageItem;
    
    volcart::Volume::Pointer volume = nullptr;
    PlaneCoords *slice = nullptr;
    int axis = 0;
    int loc[3] = {0,0,0};
    
    ChunkCache *cache = nullptr;
    QRect curr_img_area = {0,0,1000,1000};
    float scale = 0.5;
    
    QGraphicsEllipseItem *center_marker = nullptr;
    std::vector<PlaneCoords*> other_slices;
    
    std::vector<QGraphicsItem*> other_slice_items; 
    ControlPointSegmentator *seg_tool = nullptr;
};  // class CVolumeViewer

}  // namespace ChaoVis
