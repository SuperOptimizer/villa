#ifdef VC_WITH_VTK

#include "VTK3DViewer.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QToolBar>
#include <QSlider>
#include <QSpinBox>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QCloseEvent>
#include <QStatusBar>
#include <QApplication>

#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xtensor.hpp)

#include <cstring>
#include <cmath>
#include <iostream>

VTK3DViewer::VTK3DViewer(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("3D Volume Viewer");
    setAttribute(Qt::WA_DeleteOnClose);
    resize(800, 600);

    setupUI();
    setupVTKPipeline();
}

VTK3DViewer::~VTK3DViewer()
{
    // VTK smart pointers handle cleanup
}

void VTK3DViewer::setupUI()
{
    // Central widget
    auto* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    auto* mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setContentsMargins(0, 0, 0, 0);

    // Create VTK widget
    _vtkWidget = new QVTKOpenGLNativeWidget(this);
    mainLayout->addWidget(_vtkWidget, 1);

    // Toolbar
    auto* toolbar = addToolBar("Controls");
    toolbar->setMovable(false);

    // ROI Size
    toolbar->addWidget(new QLabel("ROI Size:"));
    _roiSizeSpinBox = new QSpinBox();
    _roiSizeSpinBox->setRange(64, 512);
    _roiSizeSpinBox->setSingleStep(64);
    _roiSizeSpinBox->setValue(_roiSize);
    _roiSizeSpinBox->setToolTip("Size of the cubic region to load (in voxels)");
    toolbar->addWidget(_roiSizeSpinBox);
    connect(_roiSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &VTK3DViewer::onROISizeChanged);

    toolbar->addSeparator();

    // Layer controls for surface-relative sampling
    toolbar->addWidget(new QLabel("Front:"));
    _layersFrontSpinBox = new QSpinBox();
    _layersFrontSpinBox->setRange(0, 128);
    _layersFrontSpinBox->setSingleStep(8);
    _layersFrontSpinBox->setValue(_layersFront);
    _layersFrontSpinBox->setToolTip("Layers in front of (above) the surface");
    toolbar->addWidget(_layersFrontSpinBox);
    connect(_layersFrontSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &VTK3DViewer::onLayersFrontChanged);

    toolbar->addWidget(new QLabel("Behind:"));
    _layersBehindSpinBox = new QSpinBox();
    _layersBehindSpinBox->setRange(0, 128);
    _layersBehindSpinBox->setSingleStep(8);
    _layersBehindSpinBox->setValue(_layersBehind);
    _layersBehindSpinBox->setToolTip("Layers behind (below) the surface");
    toolbar->addWidget(_layersBehindSpinBox);
    connect(_layersBehindSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &VTK3DViewer::onLayersBehindChanged);

    // Reload button
    _reloadButton = new QPushButton("Reload");
    _reloadButton->setToolTip("Reload volume data at current focus point");
    toolbar->addWidget(_reloadButton);
    connect(_reloadButton, &QPushButton::clicked, this, &VTK3DViewer::onReloadROI);

    toolbar->addSeparator();

    // Window slider
    toolbar->addWidget(new QLabel("Window:"));
    _windowSlider = new QSlider(Qt::Horizontal);
    _windowSlider->setRange(1, 255);
    _windowSlider->setValue(_window);
    _windowSlider->setFixedWidth(100);
    _windowSlider->setToolTip("Adjust contrast window width");
    toolbar->addWidget(_windowSlider);
    connect(_windowSlider, &QSlider::valueChanged, this, &VTK3DViewer::onWindowChanged);

    // Level slider
    toolbar->addWidget(new QLabel("Level:"));
    _levelSlider = new QSlider(Qt::Horizontal);
    _levelSlider->setRange(0, 255);
    _levelSlider->setValue(_level);
    _levelSlider->setFixedWidth(100);
    _levelSlider->setToolTip("Adjust brightness level");
    toolbar->addWidget(_levelSlider);
    connect(_levelSlider, &QSlider::valueChanged, this, &VTK3DViewer::onLevelChanged);

    toolbar->addSeparator();

    // Show segmentation checkbox
    _showSegmentationCheckbox = new QCheckBox("Segmentation");
    _showSegmentationCheckbox->setChecked(true);
    _showSegmentationCheckbox->setToolTip("Show/hide segmentation surface mesh");
    toolbar->addWidget(_showSegmentationCheckbox);
    connect(_showSegmentationCheckbox, &QCheckBox::toggled,
            this, &VTK3DViewer::onShowSegmentationToggled);

    // Surface opacity slider
    toolbar->addWidget(new QLabel("Opacity:"));
    _surfaceOpacitySlider = new QSlider(Qt::Horizontal);
    _surfaceOpacitySlider->setRange(0, 100);
    _surfaceOpacitySlider->setValue(50);
    _surfaceOpacitySlider->setFixedWidth(80);
    _surfaceOpacitySlider->setToolTip("Segmentation surface opacity");
    toolbar->addWidget(_surfaceOpacitySlider);
    connect(_surfaceOpacitySlider, &QSlider::valueChanged,
            this, &VTK3DViewer::onSurfaceOpacityChanged);

    // Show bounding box checkbox
    _showBoundingBoxCheckbox = new QCheckBox("Bounds");
    _showBoundingBoxCheckbox->setChecked(true);
    _showBoundingBoxCheckbox->setToolTip("Show/hide ROI bounding box");
    toolbar->addWidget(_showBoundingBoxCheckbox);
    connect(_showBoundingBoxCheckbox, &QCheckBox::toggled,
            this, &VTK3DViewer::onShowBoundingBoxToggled);

    toolbar->addSeparator();

    // Show slice planes checkbox
    _showSlicePlanesCheckbox = new QCheckBox("Slices");
    _showSlicePlanesCheckbox->setChecked(true);
    _showSlicePlanesCheckbox->setToolTip("Show/hide slice indicator planes");
    toolbar->addWidget(_showSlicePlanesCheckbox);
    connect(_showSlicePlanesCheckbox, &QCheckBox::toggled,
            this, &VTK3DViewer::onShowSlicePlanesToggled);

    // XY plane opacity (orange)
    auto* xyLabel = new QLabel("XY:");
    xyLabel->setStyleSheet("QLabel { color: rgb(255, 140, 0); font-weight: bold; }");
    toolbar->addWidget(xyLabel);
    _xyPlaneOpacitySlider = new QSlider(Qt::Horizontal);
    _xyPlaneOpacitySlider->setRange(0, 100);
    _xyPlaneOpacitySlider->setValue(30);
    _xyPlaneOpacitySlider->setFixedWidth(50);
    _xyPlaneOpacitySlider->setToolTip("XY slice plane opacity (orange)");
    toolbar->addWidget(_xyPlaneOpacitySlider);
    connect(_xyPlaneOpacitySlider, &QSlider::valueChanged,
            this, &VTK3DViewer::onXYPlaneOpacityChanged);

    // XZ plane opacity (red)
    auto* xzLabel = new QLabel("XZ:");
    xzLabel->setStyleSheet("QLabel { color: red; font-weight: bold; }");
    toolbar->addWidget(xzLabel);
    _xzPlaneOpacitySlider = new QSlider(Qt::Horizontal);
    _xzPlaneOpacitySlider->setRange(0, 100);
    _xzPlaneOpacitySlider->setValue(30);
    _xzPlaneOpacitySlider->setFixedWidth(50);
    _xzPlaneOpacitySlider->setToolTip("XZ slice plane opacity (red)");
    toolbar->addWidget(_xzPlaneOpacitySlider);
    connect(_xzPlaneOpacitySlider, &QSlider::valueChanged,
            this, &VTK3DViewer::onXZPlaneOpacityChanged);

    // YZ plane opacity (yellow)
    auto* yzLabel = new QLabel("YZ:");
    yzLabel->setStyleSheet("QLabel { color: yellow; font-weight: bold; }");
    toolbar->addWidget(yzLabel);
    _yzPlaneOpacitySlider = new QSlider(Qt::Horizontal);
    _yzPlaneOpacitySlider->setRange(0, 100);
    _yzPlaneOpacitySlider->setValue(30);
    _yzPlaneOpacitySlider->setFixedWidth(50);
    _yzPlaneOpacitySlider->setToolTip("YZ slice plane opacity (yellow)");
    toolbar->addWidget(_yzPlaneOpacitySlider);
    connect(_yzPlaneOpacitySlider, &QSlider::valueChanged,
            this, &VTK3DViewer::onYZPlaneOpacityChanged);

    toolbar->addSeparator();

    // Reset camera button
    _resetCameraButton = new QPushButton("Reset Camera");
    toolbar->addWidget(_resetCameraButton);
    connect(_resetCameraButton, &QPushButton::clicked, this, &VTK3DViewer::resetCamera);

    // Status bar
    _statusLabel = new QLabel("No volume loaded");
    statusBar()->addWidget(_statusLabel);
}

void VTK3DViewer::setupVTKPipeline()
{
    // Create render window
    _renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    _vtkWidget->setRenderWindow(_renderWindow);

    // Create renderer
    _renderer = vtkSmartPointer<vtkRenderer>::New();
    _renderer->SetBackground(0.1, 0.1, 0.15);
    _renderWindow->AddRenderer(_renderer);

    // Set up interactor style
    auto interactorStyle = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    _renderWindow->GetInteractor()->SetInteractorStyle(interactorStyle);

    // Initialize image data
    _imageData = vtkSmartPointer<vtkImageData>::New();

    // Volume rendering pipeline
    _volumeMapper = vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();
    _volumeMapper->SetInputData(_imageData);
    _volumeMapper->SetBlendModeToComposite();

    // Transfer functions
    _colorTF = vtkSmartPointer<vtkColorTransferFunction>::New();
    _opacityTF = vtkSmartPointer<vtkPiecewiseFunction>::New();
    updateTransferFunction();

    // Volume property
    _volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
    _volumeProperty->SetColor(_colorTF);
    _volumeProperty->SetScalarOpacity(_opacityTF);
    _volumeProperty->ShadeOn();
    _volumeProperty->SetInterpolationTypeToLinear();
    _volumeProperty->SetAmbient(0.4);
    _volumeProperty->SetDiffuse(0.6);
    _volumeProperty->SetSpecular(0.2);

    // Volume actor
    _volumeActor = vtkSmartPointer<vtkVolume>::New();
    _volumeActor->SetMapper(_volumeMapper);
    _volumeActor->SetProperty(_volumeProperty);
    _volumeActor->VisibilityOff();  // Hidden until data loaded
    _renderer->AddVolume(_volumeActor);

    // Segmentation mesh pipeline
    _segmentationMesh = vtkSmartPointer<vtkPolyData>::New();
    _meshMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    _meshMapper->SetInputData(_segmentationMesh);
    _meshActor = vtkSmartPointer<vtkActor>::New();
    _meshActor->SetMapper(_meshMapper);
    _meshActor->GetProperty()->SetColor(0.2, 0.8, 0.3);  // Green
    _meshActor->GetProperty()->SetOpacity(0.5);
    _meshActor->GetProperty()->SetRepresentationToSurface();
    _meshActor->VisibilityOff();
    _renderer->AddActor(_meshActor);

    // Bounding box outline
    _outlineFilter = vtkSmartPointer<vtkOutlineFilter>::New();
    _outlineFilter->SetInputData(_imageData);
    _outlineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    _outlineMapper->SetInputConnection(_outlineFilter->GetOutputPort());
    _outlineActor = vtkSmartPointer<vtkActor>::New();
    _outlineActor->SetMapper(_outlineMapper);
    _outlineActor->GetProperty()->SetColor(1.0, 1.0, 0.0);  // Yellow
    _outlineActor->GetProperty()->SetLineWidth(2.0);
    _outlineActor->VisibilityOff();
    _renderer->AddActor(_outlineActor);

    // XY slice plane (orange) - horizontal plane at Z position
    _xyPlaneSource = vtkSmartPointer<vtkPlaneSource>::New();
    _xyPlaneMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    _xyPlaneMapper->SetInputConnection(_xyPlaneSource->GetOutputPort());
    _xyPlaneActor = vtkSmartPointer<vtkActor>::New();
    _xyPlaneActor->SetMapper(_xyPlaneMapper);
    _xyPlaneActor->GetProperty()->SetColor(1.0, 0.55, 0.0);  // Orange (255, 140, 0)
    _xyPlaneActor->GetProperty()->SetOpacity(0.3);
    _xyPlaneActor->GetProperty()->SetLighting(false);
    _xyPlaneActor->VisibilityOff();
    _renderer->AddActor(_xyPlaneActor);

    // XZ slice plane (red) - vertical plane at Y position
    _xzPlaneSource = vtkSmartPointer<vtkPlaneSource>::New();
    _xzPlaneMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    _xzPlaneMapper->SetInputConnection(_xzPlaneSource->GetOutputPort());
    _xzPlaneActor = vtkSmartPointer<vtkActor>::New();
    _xzPlaneActor->SetMapper(_xzPlaneMapper);
    _xzPlaneActor->GetProperty()->SetColor(1.0, 0.0, 0.0);  // Red
    _xzPlaneActor->GetProperty()->SetOpacity(0.3);
    _xzPlaneActor->GetProperty()->SetLighting(false);
    _xzPlaneActor->VisibilityOff();
    _renderer->AddActor(_xzPlaneActor);

    // YZ slice plane (yellow) - vertical plane at X position
    _yzPlaneSource = vtkSmartPointer<vtkPlaneSource>::New();
    _yzPlaneMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    _yzPlaneMapper->SetInputConnection(_yzPlaneSource->GetOutputPort());
    _yzPlaneActor = vtkSmartPointer<vtkActor>::New();
    _yzPlaneActor->SetMapper(_yzPlaneMapper);
    _yzPlaneActor->GetProperty()->SetColor(1.0, 1.0, 0.0);  // Yellow
    _yzPlaneActor->GetProperty()->SetOpacity(0.3);
    _yzPlaneActor->GetProperty()->SetLighting(false);
    _yzPlaneActor->VisibilityOff();
    _renderer->AddActor(_yzPlaneActor);
}

void VTK3DViewer::setVolume(std::shared_ptr<Volume> vol, ChunkCache<uint8_t>* cache)
{
    _volume = vol;
    _cache = cache;
    _roiLoaded = false;

    if (_volume) {
        auto shape = _volume->shape();
        _statusLabel->setText(QString("Volume: %1 x %2 x %3")
            .arg(shape[0]).arg(shape[1]).arg(shape[2]));
    } else {
        _statusLabel->setText("No volume loaded");
        _volumeActor->VisibilityOff();
        _outlineActor->VisibilityOff();
        _renderWindow->Render();
    }
}

void VTK3DViewer::clearVolume()
{
    _volume = nullptr;
    _cache = nullptr;
    _roiLoaded = false;
    _volumeActor->VisibilityOff();
    _outlineActor->VisibilityOff();
    _statusLabel->setText("No volume loaded");
    _renderWindow->Render();
}

void VTK3DViewer::setFocusPoint(const cv::Vec3f& point)
{
    _focusPoint = point;
    // Auto-reload when focus changes
    if (_volume && _cache) {
        loadROIAroundFocus();
    }
}

void VTK3DViewer::setROISize(int size)
{
    _roiSize = size;
    _roiSizeSpinBox->setValue(size);
}

void VTK3DViewer::setSegmentationSurface(QuadSurface* surface)
{
    _segSurface = surface;
    updateSegmentationMesh();
}

void VTK3DViewer::clearSegmentationSurface()
{
    _segSurface = nullptr;
    _meshActor->VisibilityOff();
    _renderWindow->Render();
}

void VTK3DViewer::setWindowLevel(int window, int level)
{
    _window = window;
    _level = level;
    _windowSlider->setValue(window);
    _levelSlider->setValue(level);
    updateTransferFunction();
}

void VTK3DViewer::setSlicePositions(int xySlice, int xzSlice, int yzSlice)
{
    _xySlice = xySlice;
    _xzSlice = xzSlice;
    _yzSlice = yzSlice;
    if (_roiLoaded) {
        updateSlicePlanes();
    }
}

void VTK3DViewer::resetCamera()
{
    _renderer->ResetCamera();
    _renderWindow->Render();
}

void VTK3DViewer::closeEvent(QCloseEvent* event)
{
    emit windowClosed();
    event->accept();
}

void VTK3DViewer::onReloadROI()
{
    if (_volume && _cache) {
        loadROIAroundFocus();
    }
}

void VTK3DViewer::onWindowChanged(int value)
{
    _window = value;
    updateTransferFunction();
}

void VTK3DViewer::onLevelChanged(int value)
{
    _level = value;
    updateTransferFunction();
}

void VTK3DViewer::onROISizeChanged(int value)
{
    _roiSize = value;
}

void VTK3DViewer::onShowSegmentationToggled(bool checked)
{
    if (_segSurface && _roiLoaded) {
        _meshActor->SetVisibility(checked);
        _renderWindow->Render();
    }
}

void VTK3DViewer::onSurfaceOpacityChanged(int value)
{
    float opacity = value / 100.0f;
    _meshActor->GetProperty()->SetOpacity(opacity);
    if (_roiLoaded) {
        _renderWindow->Render();
    }
}

void VTK3DViewer::onShowBoundingBoxToggled(bool checked)
{
    if (_roiLoaded) {
        _outlineActor->SetVisibility(checked);
        _renderWindow->Render();
    }
}

void VTK3DViewer::onShowSlicePlanesToggled(bool checked)
{
    if (_roiLoaded) {
        _xyPlaneActor->SetVisibility(checked && _xyPlaneOpacitySlider->value() > 0);
        _xzPlaneActor->SetVisibility(checked && _xzPlaneOpacitySlider->value() > 0);
        _yzPlaneActor->SetVisibility(checked && _yzPlaneOpacitySlider->value() > 0);
        _renderWindow->Render();
    }
}

void VTK3DViewer::onXYPlaneOpacityChanged(int value)
{
    float opacity = value / 100.0f;
    _xyPlaneActor->GetProperty()->SetOpacity(opacity);
    if (_roiLoaded && _showSlicePlanesCheckbox->isChecked()) {
        _xyPlaneActor->SetVisibility(value > 0);
        _renderWindow->Render();
    }
}

void VTK3DViewer::onXZPlaneOpacityChanged(int value)
{
    float opacity = value / 100.0f;
    _xzPlaneActor->GetProperty()->SetOpacity(opacity);
    if (_roiLoaded && _showSlicePlanesCheckbox->isChecked()) {
        _xzPlaneActor->SetVisibility(value > 0);
        _renderWindow->Render();
    }
}

void VTK3DViewer::onYZPlaneOpacityChanged(int value)
{
    float opacity = value / 100.0f;
    _yzPlaneActor->GetProperty()->SetOpacity(opacity);
    if (_roiLoaded && _showSlicePlanesCheckbox->isChecked()) {
        _yzPlaneActor->SetVisibility(value > 0);
        _renderWindow->Render();
    }
}

void VTK3DViewer::onLayersFrontChanged(int value)
{
    _layersFront = value;
}

void VTK3DViewer::onLayersBehindChanged(int value)
{
    _layersBehind = value;
}

void VTK3DViewer::loadROIAroundFocus()
{
    if (!_volume || !_cache) return;

    _statusLabel->setText("Loading ROI...");
    QApplication::processEvents();

    // Focus point is in (x, y, z) order
    // readArea3D expects offset in (z, y, x) order
    int half = _roiSize / 2;
    int fx = static_cast<int>(_focusPoint[0]);
    int fy = static_cast<int>(_focusPoint[1]);
    int fz = static_cast<int>(_focusPoint[2]);

    // Volume shape is [width, height, slices] = [x, y, z]
    auto shape = _volume->shape();
    int volX = shape[0];
    int volY = shape[1];
    int volZ = shape[2];

    // Calculate ROI bounds in x, y, z
    int xMin = std::max(0, fx - half);
    int yMin = std::max(0, fy - half);
    int zMin = std::max(0, fz - half);
    int xMax = std::min(volX, fx + half);
    int yMax = std::min(volY, fy + half);
    int zMax = std::min(volZ, fz + half);

    int dimX = xMax - xMin;
    int dimY = yMax - yMin;
    int dimZ = zMax - zMin;

    if (dimX <= 0 || dimY <= 0 || dimZ <= 0) {
        _statusLabel->setText("Invalid ROI bounds");
        return;
    }

    // Store for mesh clipping (in x, y, z order)
    _roiMin = cv::Vec3i(xMin, yMin, zMin);
    _roiMax = cv::Vec3i(xMax, yMax, zMax);

    // readArea3D uses (z, y, x) order for offset and tensor shape
    cv::Vec3i readOffset(zMin, yMin, xMin);
    xt::xtensor<uint8_t, 3, xt::layout_type::column_major> data(
        {static_cast<size_t>(dimZ),
         static_cast<size_t>(dimY),
         static_cast<size_t>(dimX)}
    );

    readArea3D(data, readOffset, _volume->zarrDataset(), _cache);

    // VTK uses (x, y, z) dimensions, x varies fastest in memory
    _imageData->SetDimensions(dimX, dimY, dimZ);
    _imageData->SetOrigin(xMin, yMin, zMin);
    _imageData->SetSpacing(1.0, 1.0, 1.0);
    _imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 1);

    // Get VTK buffer pointer
    uint8_t* vtkPtr = static_cast<uint8_t*>(_imageData->GetScalarPointer());

    // First, copy ALL the voxel data to the VTK buffer
    // data is indexed as (z, y, x), VTK wants x varying fastest
    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                vtkPtr[x + y * dimX + z * dimX * dimY] = data(z, y, x);
            }
        }
    }

    // Debug: print some sample values to verify data is loaded
    std::cout << "[VTK3D] Loaded " << dimX << "x" << dimY << "x" << dimZ << " voxels" << std::endl;
    std::cout << "[VTK3D] Sample values: center=" << (int)data(dimZ/2, dimY/2, dimX/2)
              << " corner=" << (int)data(0, 0, 0) << std::endl;

    // Now, if we have a surface for surface-relative sampling, zero out voxels outside the range
    const bool useSurfaceSampling = _segSurface && (_layersFront > 0 || _layersBehind > 0);

    if (useSurfaceSampling) {
        // Use the surface's gen() method to generate coordinates for the XY extent of the ROI
        // This gives us surface Z values at each (x,y) position, then we keep voxels within layers of that Z

        // First, create a 2D array to store the surface Z coordinate and validity at each (x,y) in the ROI
        // We'll sample the surface at each XY position in the volume

        // Create a mask initialized to false
        std::vector<bool> keepMask(dimX * dimY * dimZ, false);

        // Get surface center and scale for coordinate transformation
        const cv::Vec3f surfCenter = _segSurface->center();
        const cv::Vec2f surfScale = _segSurface->scale();
        const cv::Mat_<cv::Vec3f>* rawPoints = _segSurface->rawPointsPtr();

        std::cout << "[VTK3D] Surface center: " << surfCenter
                  << ", scale: " << surfScale[0] << "," << surfScale[1] << std::endl;

        // Debug: show expected grid coord for ROI center
        float roiCenterX = (xMin + xMax) / 2.0f;
        float roiCenterY = (yMin + yMax) / 2.0f;
        std::cout << "[VTK3D] ROI center vol: (" << roiCenterX << "," << roiCenterY << ")" << std::endl;

        if (rawPoints && !rawPoints->empty()) {
            const int gridRows = rawPoints->rows;
            const int gridCols = rawPoints->cols;

            // Debug: show grid coords for ROI center
            float roiGridCol = (roiCenterX - surfCenter[0]) * surfScale[0] + gridCols / 2.0f;
            float roiGridRow = (roiCenterY - surfCenter[1]) * surfScale[1] + gridRows / 2.0f;
            std::cout << "[VTK3D] Grid size: " << gridCols << "x" << gridRows
                      << ", ROI center -> grid: (" << roiGridCol << "," << roiGridRow << ")" << std::endl;

            // For each (x,y) in the ROI, find the corresponding surface point
            // The rawPoints stores volume coordinates at each grid position
            //
            // Coordinate systems (from QuadSurface.cpp):
            // - Nominal: volume coordinates (what we have)
            // - Internal relative (ptr): coords where center is at 0/0
            // - Internal absolute (_points): grid coords where upper-left is at 0/0
            //
            // center = [gridCols/2/scale[0], gridRows/2/scale[1], 0] in nominal coords
            // So center represents where grid center is in volume space
            //
            // To convert volume (vx, vy) to grid (col, row):
            //   gridCol = (vx - center[0]) * scale[0] + gridCols/2
            //   gridRow = (vy - center[1]) * scale[1] + gridRows/2

            int keptCount = 0;
            int zeroedCount = 0;
            int validXY = 0;

            for (int ly = 0; ly < dimY; ++ly) {
                float vy = static_cast<float>(yMin + ly);

                for (int lx = 0; lx < dimX; ++lx) {
                    float vx = static_cast<float>(xMin + lx);

                    // Convert volume (x,y) to grid coordinates
                    float gridCol = (vx - surfCenter[0]) * surfScale[0] + gridCols / 2.0f;
                    float gridRow = (vy - surfCenter[1]) * surfScale[1] + gridRows / 2.0f;

                    int gc = static_cast<int>(std::round(gridCol));
                    int gr = static_cast<int>(std::round(gridRow));

                    // Check if within grid bounds
                    if (gc < 0 || gc >= gridCols || gr < 0 || gr >= gridRows) {
                        // Surface doesn't cover this XY - zero out all Z at this position
                        for (int lz = 0; lz < dimZ; ++lz) {
                            // Already false in keepMask
                        }
                        continue;
                    }

                    const cv::Vec3f& surfPt = (*rawPoints)(gr, gc);

                    // Check if valid surface point
                    if (surfPt[0] == -1.f) {
                        continue;
                    }

                    validXY++;
                    if (validXY == 1) {
                        std::cout << "[VTK3D] First valid XY: vol(" << vx << "," << vy
                                  << ") -> grid(" << gc << "," << gr
                                  << ") -> surfZ=" << surfPt[2] << std::endl;
                    }

                    // Get the normal at this surface point
                    cv::Vec3f normal = _segSurface->gridNormal(gr, gc);
                    float normalLen = cv::norm(normal);
                    if (normalLen < 0.001f) {
                        normal = cv::Vec3f(0, 0, 1);
                    } else {
                        normal /= normalLen;
                    }

                    // For each Z in the ROI, check if it's within layer bounds of this surface point
                    for (int lz = 0; lz < dimZ; ++lz) {
                        float vz = static_cast<float>(zMin + lz);

                        // Compute signed distance along normal
                        cv::Vec3f voxelPt(vx, vy, vz);
                        cv::Vec3f diff = voxelPt - surfPt;
                        float signedDist = diff.dot(normal);

                        // Check if within layer bounds
                        if (signedDist >= -static_cast<float>(_layersBehind) &&
                            signedDist <= static_cast<float>(_layersFront)) {
                            int idx = lx + ly * dimX + lz * dimX * dimY;
                            keepMask[idx] = true;
                        }
                    }
                }
            }

            // Apply the mask
            for (int lz = 0; lz < dimZ; ++lz) {
                for (int ly = 0; ly < dimY; ++ly) {
                    for (int lx = 0; lx < dimX; ++lx) {
                        int idx = lx + ly * dimX + lz * dimX * dimY;
                        if (!keepMask[idx]) {
                            vtkPtr[idx] = 0;
                            zeroedCount++;
                        } else {
                            keptCount++;
                        }
                    }
                }
            }

            std::cout << "[VTK3D] Valid XY positions: " << validXY << "/" << (dimX * dimY)
                      << ", kept " << keptCount << " voxels, zeroed " << zeroedCount
                      << " (" << (100.0 * keptCount / (keptCount + zeroedCount)) << "% kept)" << std::endl;
        }
    }

    _imageData->Modified();

    // Get the actual scalar range of non-zero values for debugging
    double scalarRange[2];
    _imageData->GetScalarRange(scalarRange);
    std::cout << "[VTK3D] Scalar range: " << scalarRange[0] << " to " << scalarRange[1] << std::endl;
    _volumeActor->VisibilityOn();
    _roiLoaded = true;

    // Update bounding box
    updateBoundingBox();

    // Update slice planes
    updateSlicePlanes();

    // Update segmentation mesh if we have a surface
    if (_segSurface) {
        updateSegmentationMesh();
    }

    resetCamera();

    QString status = QString("ROI: %1x%2x%3 at (%4, %5, %6)")
        .arg(dimX).arg(dimY).arg(dimZ)
        .arg(xMin).arg(yMin).arg(zMin);
    if (useSurfaceSampling) {
        status += QString(" [%1 front, %2 behind]").arg(_layersFront).arg(_layersBehind);
    }
    _statusLabel->setText(status);
}

void VTK3DViewer::updateSegmentationMesh()
{
    if (!_segSurface || !_roiLoaded) {
        _meshActor->VisibilityOff();
        return;
    }

    auto points = vtkSmartPointer<vtkPoints>::New();
    auto triangles = vtkSmartPointer<vtkCellArray>::New();

    // Get surface points that fall within the ROI
    const cv::Mat_<cv::Vec3f>* rawPoints = _segSurface->rawPointsPtr();
    if (!rawPoints || rawPoints->empty()) {
        _meshActor->VisibilityOff();
        return;
    }

    // Build point index map for triangle connectivity
    std::unordered_map<int, vtkIdType> pointIndexMap;
    int rows = rawPoints->rows;
    int cols = rawPoints->cols;

    // First pass: collect all valid points within ROI
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& p = (*rawPoints)(r, c);
            if (p[0] == -1.f) continue;  // Invalid point

            // Check if point is within ROI (with some margin)
            bool inROI = (p[0] >= _roiMin[0] - 10 && p[0] <= _roiMax[0] + 10 &&
                         p[1] >= _roiMin[1] - 10 && p[1] <= _roiMax[1] + 10 &&
                         p[2] >= _roiMin[2] - 10 && p[2] <= _roiMax[2] + 10);

            if (inROI) {
                int flatIdx = r * cols + c;
                vtkIdType ptId = points->InsertNextPoint(p[0], p[1], p[2]);
                pointIndexMap[flatIdx] = ptId;
            }
        }
    }

    // Second pass: create triangles for quads where all 4 corners are in ROI
    for (int r = 0; r < rows - 1; ++r) {
        for (int c = 0; c < cols - 1; ++c) {
            int idx00 = r * cols + c;
            int idx01 = r * cols + (c + 1);
            int idx10 = (r + 1) * cols + c;
            int idx11 = (r + 1) * cols + (c + 1);

            // Check if all 4 corners are in our point map
            auto it00 = pointIndexMap.find(idx00);
            auto it01 = pointIndexMap.find(idx01);
            auto it10 = pointIndexMap.find(idx10);
            auto it11 = pointIndexMap.find(idx11);

            if (it00 != pointIndexMap.end() &&
                it01 != pointIndexMap.end() &&
                it10 != pointIndexMap.end() &&
                it11 != pointIndexMap.end()) {
                // Create two triangles for this quad
                triangles->InsertNextCell(3);
                triangles->InsertCellPoint(it00->second);
                triangles->InsertCellPoint(it01->second);
                triangles->InsertCellPoint(it11->second);

                triangles->InsertNextCell(3);
                triangles->InsertCellPoint(it00->second);
                triangles->InsertCellPoint(it11->second);
                triangles->InsertCellPoint(it10->second);
            }
        }
    }

    _segmentationMesh->SetPoints(points);
    _segmentationMesh->SetPolys(triangles);
    _segmentationMesh->Modified();

    if (points->GetNumberOfPoints() > 0 && _showSegmentationCheckbox->isChecked()) {
        _meshActor->VisibilityOn();
    } else {
        _meshActor->VisibilityOff();
    }

    _renderWindow->Render();
}

void VTK3DViewer::updateTransferFunction()
{
    // Calculate the intensity range based on window/level
    float minVal = _level - _window / 2.0f;
    float maxVal = _level + _window / 2.0f;

    // Color transfer function - grayscale
    _colorTF->RemoveAllPoints();
    _colorTF->AddRGBPoint(0.0, 0.0, 0.0, 0.0);   // Value 0 is black
    _colorTF->AddRGBPoint(1.0, 0.0, 0.0, 0.0);   // Value 1 is black (for masking transition)
    _colorTF->AddRGBPoint(minVal, 0.0, 0.0, 0.0);
    _colorTF->AddRGBPoint(maxVal, 1.0, 1.0, 1.0);

    // Opacity transfer function
    // Make value 0 fully transparent (for masked out voxels)
    // Make other values visible based on window/level
    _opacityTF->RemoveAllPoints();
    _opacityTF->AddPoint(0.0, 0.0);              // Value 0: fully transparent
    _opacityTF->AddPoint(1.0, 0.0);              // Value 1: transparent (sharp cutoff)
    _opacityTF->AddPoint(2.0, 0.05);             // Value 2+: start becoming visible
    _opacityTF->AddPoint(minVal + (_window * 0.2f), 0.1);
    _opacityTF->AddPoint(maxVal, 0.5);           // Max opacity at high values

    if (_roiLoaded) {
        _renderWindow->Render();
    }
}

void VTK3DViewer::updateBoundingBox()
{
    _outlineFilter->Update();
    if (_showBoundingBoxCheckbox->isChecked()) {
        _outlineActor->VisibilityOn();
    }
}

void VTK3DViewer::updateSlicePlanes()
{
    if (!_roiLoaded) return;

    // Get ROI bounds
    double xMin = _roiMin[0], xMax = _roiMax[0];
    double yMin = _roiMin[1], yMax = _roiMax[1];
    double zMin = _roiMin[2], zMax = _roiMax[2];

    // XY plane (orange) - horizontal at Z = _xySlice
    // Clamp slice position to ROI bounds
    double xyZ = std::clamp(static_cast<double>(_xySlice), zMin, zMax);
    _xyPlaneSource->SetOrigin(xMin, yMin, xyZ);
    _xyPlaneSource->SetPoint1(xMax, yMin, xyZ);
    _xyPlaneSource->SetPoint2(xMin, yMax, xyZ);
    _xyPlaneSource->Update();

    // XZ plane (red) - vertical at Y = _xzSlice
    double xzY = std::clamp(static_cast<double>(_xzSlice), yMin, yMax);
    _xzPlaneSource->SetOrigin(xMin, xzY, zMin);
    _xzPlaneSource->SetPoint1(xMax, xzY, zMin);
    _xzPlaneSource->SetPoint2(xMin, xzY, zMax);
    _xzPlaneSource->Update();

    // YZ plane (yellow) - vertical at X = _yzSlice
    double yzX = std::clamp(static_cast<double>(_yzSlice), xMin, xMax);
    _yzPlaneSource->SetOrigin(yzX, yMin, zMin);
    _yzPlaneSource->SetPoint1(yzX, yMax, zMin);
    _yzPlaneSource->SetPoint2(yzX, yMin, zMax);
    _yzPlaneSource->Update();

    // Show planes if checkbox is checked and opacity > 0
    if (_showSlicePlanesCheckbox->isChecked()) {
        _xyPlaneActor->SetVisibility(_xyPlaneOpacitySlider->value() > 0);
        _xzPlaneActor->SetVisibility(_xzPlaneOpacitySlider->value() > 0);
        _yzPlaneActor->SetVisibility(_yzPlaneOpacitySlider->value() > 0);
    }

    _renderWindow->Render();
}

#endif // VC_WITH_VTK
