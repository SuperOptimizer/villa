// CVolumeViewer.h
// Chao Du 2015 April
#pragma once

#include <QtWidgets>
#include <opencv2/core/core.hpp>

class ChunkCache;
class Surface;

class QGraphicsScene;

namespace volcart {
    class Volume;
}

namespace ChaoVis
{

class CVolumeViewerView;
class SegmentationStruct;
class CSurfaceCollection;
class POI;

class CVolumeViewer : public QWidget
{
    Q_OBJECT

public:
    QPushButton* fNextBtn;
    QPushButton* fPrevBtn;
    CVolumeViewer(CSurfaceCollection *col, QWidget* parent = 0);
    ~CVolumeViewer(void);

    void Reset();

    virtual void SetImage(const QImage& nSrc);
    void SetImageIndex(int nImageIndex);
    void SetNumSlices(int num);
    void SetRotation(int degress);
    void Rotate(int delta);
    void ResetRotation();
    void setCache(ChunkCache *cache);
    void loadSlice();
    void setSurface(const std::string &name);
    cv::Mat getCoordSlice();
    void renderVisible(bool force = false);
    cv::Mat render_area(const cv::Rect &roi);
    void invalidateVis();
    void invalidateIntersect();
    
    CVolumeViewerView* fGraphicsView;

public slots:
    void OnVolumeChanged(std::shared_ptr<volcart::Volume> vol);
    void onVolumeClicked(QPointF scene_loc,Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onSurfaceChanged(std::string name, Surface *surf);
    void onPOIChanged(std::string name, POI *poi);
    void onScrolled();
    void onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers);
    void onCursorMove(QPointF);

signals:
    void SendSignalSliceShift(int shift, int axis);
    void SendSignalStatusMessageAvailable(QString text, int timeout);
    void sendVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, cv::Vec3f surf_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
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
    QImage* fImgQImage;
    int sliceIndexToolStart{-1};
    int fScanRange;  // how many slices a mouse wheel step will jump
    // Required to be able to reset the rotation without also resetting the scaling
    int currentRotation{0};

    // user settings
    bool fCenterOnZoomEnabled;
    int fScrollSpeed{-1};
    bool fSkipImageFormatConv;

    QGraphicsPixmapItem* fBaseImageItem;
    
    std::shared_ptr<volcart::Volume> volume = nullptr;
    Surface *_surf = nullptr;
    std::string _surf_name;
    int axis = 0;
    int loc[3] = {0,0,0};
    
    ChunkCache *cache = nullptr;
    QRect curr_img_area = {0,0,1000,1000};
    float _scale = 0.5;
    float _scene_scale = 1.0;
    float _ds_scale = 0.5;
    int _ds_sd_idx = 1;
    float _max_scale = 1;
    float _min_scale = 1;

    float _z_off = 0.0;
    
    QGraphicsItem *_center_marker = nullptr;
    QGraphicsItem *_cursor = nullptr;
    
    bool _slice_vis_valid = false;
    std::vector<QGraphicsItem*> slice_vis_items; 
    bool _intersect_valid = false;
    std::vector<QGraphicsItem*> _intersect_items;;
    
    CSurfaceCollection *_surf_col = nullptr;
};  // class CVolumeViewer

}  // namespace ChaoVis
