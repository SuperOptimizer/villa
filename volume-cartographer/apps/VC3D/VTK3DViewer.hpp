#pragma once

#ifdef VC_WITH_VTK

#include <QMainWindow>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkImageData.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkOutlineFilter.h>
#include <vtkPlaneSource.h>

#include <opencv2/core.hpp>
#include <memory>

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/ChunkCache.hpp"

class QuadSurface;
class QSlider;
class QSpinBox;
class QPushButton;
class QCheckBox;
class QLabel;

class VTK3DViewer : public QMainWindow
{
    Q_OBJECT

public:
    explicit VTK3DViewer(QWidget* parent = nullptr);
    ~VTK3DViewer() override;

    // Volume data
    void setVolume(std::shared_ptr<Volume> vol, ChunkCache<uint8_t>* cache);
    void clearVolume();

    // Focus-based ROI loading
    void setFocusPoint(const cv::Vec3f& point);
    void setROISize(int size);

    // Segmentation surface overlay
    void setSegmentationSurface(QuadSurface* surface);
    void clearSegmentationSurface();

    // Transfer function
    void setWindowLevel(int window, int level);

    // Slice planes (for showing where 2D viewers are cutting)
    void setSlicePositions(int xySlice, int xzSlice, int yzSlice);

    // Camera
    void resetCamera();

signals:
    void volumePointClicked(cv::Vec3f position);
    void windowClosed();

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onReloadROI();
    void onWindowChanged(int value);
    void onLevelChanged(int value);
    void onROISizeChanged(int value);
    void onShowSegmentationToggled(bool checked);
    void onSurfaceOpacityChanged(int value);
    void onShowBoundingBoxToggled(bool checked);
    void onShowSlicePlanesToggled(bool checked);
    void onXYPlaneOpacityChanged(int value);
    void onXZPlaneOpacityChanged(int value);
    void onYZPlaneOpacityChanged(int value);
    void onLayersFrontChanged(int value);
    void onLayersBehindChanged(int value);

private:
    void setupUI();
    void setupVTKPipeline();
    void loadROIAroundFocus();
    void updateSegmentationMesh();
    void updateTransferFunction();
    void updateBoundingBox();
    void updateSlicePlanes();

    // VTK widget
    QVTKOpenGLNativeWidget* _vtkWidget{nullptr};

    // VTK rendering components
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> _renderWindow;
    vtkSmartPointer<vtkRenderer> _renderer;

    // Volume rendering pipeline
    vtkSmartPointer<vtkImageData> _imageData;
    vtkSmartPointer<vtkGPUVolumeRayCastMapper> _volumeMapper;
    vtkSmartPointer<vtkVolume> _volumeActor;
    vtkSmartPointer<vtkVolumeProperty> _volumeProperty;
    vtkSmartPointer<vtkColorTransferFunction> _colorTF;
    vtkSmartPointer<vtkPiecewiseFunction> _opacityTF;

    // Segmentation mesh overlay
    vtkSmartPointer<vtkPolyData> _segmentationMesh;
    vtkSmartPointer<vtkPolyDataMapper> _meshMapper;
    vtkSmartPointer<vtkActor> _meshActor;

    // Bounding box
    vtkSmartPointer<vtkOutlineFilter> _outlineFilter;
    vtkSmartPointer<vtkPolyDataMapper> _outlineMapper;
    vtkSmartPointer<vtkActor> _outlineActor;

    // Slice planes (XY=orange, XZ=red, YZ=yellow)
    vtkSmartPointer<vtkPlaneSource> _xyPlaneSource;
    vtkSmartPointer<vtkPolyDataMapper> _xyPlaneMapper;
    vtkSmartPointer<vtkActor> _xyPlaneActor;

    vtkSmartPointer<vtkPlaneSource> _xzPlaneSource;
    vtkSmartPointer<vtkPolyDataMapper> _xzPlaneMapper;
    vtkSmartPointer<vtkActor> _xzPlaneActor;

    vtkSmartPointer<vtkPlaneSource> _yzPlaneSource;
    vtkSmartPointer<vtkPolyDataMapper> _yzPlaneMapper;
    vtkSmartPointer<vtkActor> _yzPlaneActor;

    // Data sources
    std::shared_ptr<Volume> _volume;
    ChunkCache<uint8_t>* _cache{nullptr};
    QuadSurface* _segSurface{nullptr};

    // ROI state
    cv::Vec3f _focusPoint{0, 0, 0};
    int _roiSize{256};
    cv::Vec3i _roiMin{0, 0, 0};
    cv::Vec3i _roiMax{0, 0, 0};
    bool _roiLoaded{false};

    // Slice positions (in volume coordinates)
    int _xySlice{0};  // Z position for XY plane
    int _xzSlice{0};  // Y position for XZ plane
    int _yzSlice{0};  // X position for YZ plane

    // Surface-relative sampling
    int _layersFront{32};   // Layers in front of (above) the surface
    int _layersBehind{32};  // Layers behind (below) the surface

    // Transfer function state
    int _window{255};
    int _level{128};

    // UI controls
    QSlider* _windowSlider{nullptr};
    QSlider* _levelSlider{nullptr};
    QSlider* _surfaceOpacitySlider{nullptr};
    QSpinBox* _roiSizeSpinBox{nullptr};
    QPushButton* _reloadButton{nullptr};
    QPushButton* _resetCameraButton{nullptr};
    QCheckBox* _showSegmentationCheckbox{nullptr};
    QCheckBox* _showBoundingBoxCheckbox{nullptr};
    QCheckBox* _showSlicePlanesCheckbox{nullptr};
    QSlider* _xyPlaneOpacitySlider{nullptr};
    QSlider* _xzPlaneOpacitySlider{nullptr};
    QSlider* _yzPlaneOpacitySlider{nullptr};
    QSpinBox* _layersFrontSpinBox{nullptr};
    QSpinBox* _layersBehindSpinBox{nullptr};
    QLabel* _statusLabel{nullptr};
};

#endif // VC_WITH_VTK
