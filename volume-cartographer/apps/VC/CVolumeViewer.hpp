// CVolumeViewer.h
// Chao Du 2015 April
#pragma once

#include <QtWidgets>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QDebug>

#include "vc/core/types/VolumePkg.hpp"

class CoordGenerator;
class ChunkCache;

namespace ChaoVis
{

class CVolumeViewerView : public QGraphicsView
{
    Q_OBJECT

    public:
        CVolumeViewerView(QWidget* parent = nullptr);

        void setup();

        void keyPressEvent(QKeyEvent* event) override;
        void keyReleaseEvent(QKeyEvent* event) override;

        bool isRangeKeyPressed() { return rangeKeyPressed; }
        bool isCurvePanKeyPressed() { return curvePanKeyPressed; }
        bool isRotateKyPressed() { return rotateKeyPressed; }

        void showTextAboveCursor(const QString& value, const QString& label, const QColor& color);
        void hideTextAboveCursor();

        void showCurrentImpactRange(int range);
        void showCurrentScanRange(int range);
        void showCurrentSliceIndex(int slice, bool highlight);

    signals:
    void sendScrolled();
    
    protected:
        bool rangeKeyPressed{false};
        bool curvePanKeyPressed{false};
        bool rotateKeyPressed{false};

        QGraphicsTextItem* textAboveCursor;
        QGraphicsRectItem* backgroundBehindText;
        QTimer* timerTextAboveCursor;
        
        void scrollContentsBy(int dx, int dy);
};

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
    void setSlice(CoordGenerator *slice);
    cv::Mat getCoordSlice();

protected:
    bool eventFilter(QObject* watched, QEvent* event);

public slots:
    void OnZoomInClicked(void);
    void OnZoomOutClicked(void);
    void OnResetClicked(void);
    void OnImageRotationSpinChanged(void);
    void OnViewAxisChanged(void);
    void OnLocChanged(int x_, int y_, int z_);
    void OnVolumeChanged(volcart::Volume::Pointer vol);
    void OnSliceChanged();
    void onScrolled();

signals:
    void SendSignalSliceShift(int shift, int axis);
    void SendSignalStatusMessageAvailable(QString text, int timeout);

protected:
    void ScaleImage(double nFactor);
    void CenterOn(const QPointF& point);
    virtual void UpdateButtons(void);

protected:
    // widget components
    CVolumeViewerView* fGraphicsView;
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
    CoordGenerator *slice = nullptr;
    int axis = 0;
    int loc[3] = {0,0,0};
    
    ChunkCache *cache = nullptr;
    QRect curr_img_area = {0,0,1000,1000};
};  // class CVolumeViewer

}  // namespace ChaoVis
