// CVolumeViewer.cpp
// Chao Du 2015 April
#include "CVolumeViewer.hpp"
#include "UDataManipulateUtils.hpp"
#include "HBase.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>

#include "CVolumeViewerView.hpp"
#include "SegmentationStruct.hpp"

#include "vc/core/util/Slicing.hpp"

using namespace ChaoVis;
using qga = QGuiApplication;

#define BGND_RECT_MARGIN 8
#define DEFAULT_TEXT_COLOR QColor(255, 255, 120)
// #define ZOOM_FACTOR 1.148698354997035
#define ZOOM_FACTOR 2.0 //1.414213562373095

// Constructor
CVolumeViewer::CVolumeViewer(QWidget* parent)
    : QWidget(parent)
    , fCanvas(nullptr)
    // , fScrollArea(nullptr)
    , fGraphicsView(nullptr)
    , fZoomInBtn(nullptr)
    , fZoomOutBtn(nullptr)
    , fResetBtn(nullptr)
    , fNextBtn(nullptr)
    , fPrevBtn(nullptr)
    , fViewState(EViewState::ViewStateIdle)
    , fImgQImage(nullptr)
    , fBaseImageItem(nullptr)
    , fScanRange(1)
{
    // buttons
    // fZoomInBtn = new QPushButton(tr("Zoom In"), this);
    // fZoomOutBtn = new QPushButton(tr("Zoom Out"), this);
    // fResetBtn = new QPushButton(tr("Reset"), this);
    // fNextBtn = new QPushButton(tr("Next Slice"), this);
    // fPrevBtn = new QPushButton(tr("Previous Slice"), this);

    // fImageRotationSpin = new QSpinBox(this);
    // fImageRotationSpin->setMinimum(-360);
    // fImageRotationSpin->setMaximum(360);
    // fImageRotationSpin->setSuffix("°");
    // fImageRotationSpin->setEnabled(true);
    // connect(fImageRotationSpin, SIGNAL(editingFinished()), this, SLOT(OnImageRotationSpinChanged()));

    // fAxisCombo = new QComboBox(this);
    //data is the missing axis (but in inverted order ZYX)
    // fAxisCombo->addItem(QString::fromStdString("XY"), QVariant(0));
    // fAxisCombo->addItem(QString::fromStdString("XZ"), QVariant(1));
    // fAxisCombo->addItem(QString::fromStdString("YZ"), QVariant(2));
    // fAxisCombo->addItem(QString::fromStdString("slice"), QVariant(3));
    // connect(fAxisCombo, &QComboBox::currentIndexChanged, this, &CVolumeViewer::OnViewAxisChanged);

    fBaseImageItem = nullptr;

    // Create graphics view
    fGraphicsView = new CVolumeViewerView(this);
    
    
    fGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    fGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    
    fGraphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);
    
    fGraphicsView->setRenderHint(QPainter::Antialiasing);
    // setFocusProxy(fGraphicsView);
    connect(fGraphicsView, &CVolumeViewerView::sendScrolled, this, &CVolumeViewer::onScrolled);
    connect(fGraphicsView, &CVolumeViewerView::sendVolumeClicked, this, &CVolumeViewer::onVolumeClicked);
    connect(fGraphicsView, &CVolumeViewerView::sendZoom, this, &CVolumeViewer::onZoom);

    // Create graphics scene
    fScene = new QGraphicsScene({-2500,-2500,5000,5000}, this);

    // Set the scene
    fGraphicsView->setScene(fScene);
    // fGraphicsView->setup();

    // fGraphicsView->viewport()->installEventFilter(this);
    // fGraphicsView->setDragMode(QGraphicsView::ScrollHandDrag);

    // fButtonsLayout = new QHBoxLayout;
    // fButtonsLayout->addWidget(fZoomInBtn);
    // fButtonsLayout->addWidget(fZoomOutBtn);
    // fButtonsLayout->addWidget(fResetBtn);
    // fButtonsLayout->addWidget(fImageRotationSpin);
    // // fButtonsLayout->addWidget(fAxisCombo);
    // // Add some space between the slice spin box and the curve tools (color, checkboxes, ...)
    // fButtonsLayout->addSpacerItem(new QSpacerItem(1, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));

    // connect(fZoomInBtn, SIGNAL(clicked()), this, SLOT(OnZoomInClicked()));
    // connect(fZoomOutBtn, SIGNAL(clicked()), this, SLOT(OnZoomOutClicked()));
    // connect(fResetBtn, SIGNAL(clicked()), this, SLOT(OnResetClicked()));

    QSettings settings("VC.ini", QSettings::IniFormat);
    // fCenterOnZoomEnabled = settings.value("viewer/center_on_zoom", false).toInt() != 0;
    // fScrollSpeed = settings.value("viewer/scroll_speed", false).toInt();
    fSkipImageFormatConv = settings.value("perf/chkSkipImageFormatConvExp", false).toBool();

    QVBoxLayout* aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->addWidget(fGraphicsView);
    // aWidgetLayout->addLayout(fButtonsLayout);

    setLayout(aWidgetLayout);
}

// Destructor
CVolumeViewer::~CVolumeViewer(void)
{
    deleteNULL(fGraphicsView);
    deleteNULL(fScene);
    // deleteNULL(fScrollArea);
    // deleteNULL(fZoomInBtn);
    // deleteNULL(fZoomOutBtn);
    // deleteNULL(fResetBtn);
    // deleteNULL(fImageRotationSpin);
}

void CVolumeViewer::SetButtonsEnabled(bool state)
{
    // fZoomOutBtn->setEnabled(state);
    // fZoomInBtn->setEnabled(state);
    // fImageRotationSpin->setEnabled(state);
}

void CVolumeViewer::SetImage(const QImage& nSrc)
{
    if (fImgQImage == nullptr) {
        fImgQImage = new QImage(nSrc);
    } else {
        *fImgQImage = nSrc;
    }

    // Create a QPixmap from the QImage
    QPixmap pixmap = QPixmap::fromImage(*fImgQImage, fSkipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);

    // Add the QPixmap to the scene as a QGraphicsPixmapItem
    if (!fBaseImageItem) {
        fBaseImageItem = fScene->addPixmap(pixmap);
    } else {
        fBaseImageItem->setPixmap(pixmap);
    }
    update();
}

void round_scale(float &scale)
{
    if (abs(scale-round(log2(scale))) < 0.02)
        scale = pow(2,round(log2(scale)));
}

//get center of current visible area in scene coordinates
QPointF visible_center(QGraphicsView *view)
{
    QRectF bbox = view->mapToScene(view->viewport()->geometry()).boundingRect();
    return bbox.topLeft() + QPointF(bbox.width(),bbox.height())*0.5;
}

void CVolumeViewer::onZoom(int steps, QPointF scene_loc, Qt::KeyboardModifiers modifiers)
{
    //TODO don't invalidate if only _scene_scale chagned
    invalidateVis();
    
    if (modifiers & Qt::ShiftModifier) {
        slice->setOffsetZ(slice->offsetZ()+steps);
        renderVisible(true);
    }
    else {
        float zoom = pow(ZOOM_FACTOR, steps);
        
        _scale *= zoom;
        round_scale(_scale);
        
        if (_scale >= _max_scale) {
            _ds_scale = _max_scale;
            _ds_sd_idx = -log2(_ds_scale);
            _scene_scale = _scale/_ds_scale;
        }
        else if (_scale < _min_scale) {
            _ds_scale = _min_scale;
            _ds_sd_idx = -log2(_ds_scale);
            _scene_scale = _scale/_ds_scale;
        }
        else {
            _ds_sd_idx = -log2(_scale);
            _ds_scale = pow(2,-_ds_sd_idx);
            _scene_scale = _scale/_ds_scale;
        }
        
        curr_img_area = {0,0,0,0};
        QPointF center = visible_center(fGraphicsView) * zoom;
        
        //FIXME get correct size for slice!
        int max_size = std::max(volume->sliceWidth(), std::max(volume->numSlices(), volume->sliceHeight()))*_ds_scale + 512;
        fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
        
        fGraphicsView->centerOn(center);
        renderVisible();
    }
}

void CVolumeViewer::OnVolumeChanged(volcart::Volume::Pointer volume_)
{
    volume = volume_;
    
    printf("sizes %d %d %d\n", volume_->sliceWidth(), volume_->sliceHeight(), volume_->numSlices());
    
    int max_size = std::max(volume_->sliceWidth(), std::max(volume_->numSlices(), volume_->sliceHeight()))*_ds_scale + 512;
    printf("max size %d\n", max_size);
    fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
    
    //FIXME currently hardcoded
    _max_scale = 0.5;
    _min_scale = pow(2.0,1.-volume->numScales());

    renderVisible(true);
}

cv::Vec3f loc3d_at_imgpos(volcart::Volume *vol, CoordGenerator *slice, QPointF loc, float scale)
{
    xt::xarray<float> coords;
    
    int sd_idx = 1;
    
    float round_scale = 0.5;
    while (0.5*round_scale >= scale && sd_idx < vol->numScales()-1) {
        sd_idx++;
        round_scale *= 0.5;
    }
    
    slice->gen_coords(coords, loc.x()*round_scale/scale,loc.y()*round_scale/scale, 1, 1, scale/round_scale, round_scale);
    
    coords /= round_scale;
    
    return {coords(0,0,2),coords(0,0,1),coords(0,0,0)};
}

void CVolumeViewer::onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    xt::xarray<float> coords;
    slice->gen_coords(coords, {scene_loc.x(),scene_loc.y(),1,1}, 1.0, _ds_scale);
    
    coords /= _ds_scale;

    sendVolumeClicked({coords(0,0,2),coords(0,0,1),coords(0,0,0)}, buttons, modifiers);
}

void CVolumeViewer::setCache(ChunkCache *cache_)
{
    cache = cache_;
}

void CVolumeViewer::setSlice(CoordGenerator *slice_)
{
    slice = slice_;
    OnSliceChanged();
}


void CVolumeViewer::invalidateVis()
{
    _slice_vis_valid = false;    
    for(auto &item : slice_vis_items) {
        fScene->removeItem(item);
        delete item;
    }
    slice_vis_items.resize(0);
}

void CVolumeViewer::OnSliceChanged()
{
    invalidateVis();
    
    curr_img_area = {0,0,0,0};
    renderVisible();
}


// void calc_scales(float scale, float &render_scale, float &coord_scale, int &sd_idx, int max_idx)
// {
//     sd_idx = 1;
//     
//     while (0.5*coord_scale >= scale && sd_idx < max_idx-1) {
//         sd_idx++;
//         coord_scale *= 0.5;
//     }
//     
//     render_scale = scale/coord_scale;
// }

cv::Mat CVolumeViewer::render_area(const cv::Rect &roi)
{
    xt::xarray<float> coords;
    xt::xarray<uint8_t> img;

    slice->gen_coords(coords, roi, 1.0, _ds_scale);
    readInterpolated3D(img, volume->zarrDataset(_ds_sd_idx), coords, cache);
    cv::Mat m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    
    return m.clone();
}

// void CVolumeViewer::currRoi(cv::Rect &roi, float &render_scale, float &coord_scale, int &sd_idx) const
// {  
//     calc_scales(scale, render_scale, coord_scale, sd_idx, volume->numScales()-1);
//     
//     float m = coord_scale/scale;
//     
//     roi = {curr_img_area.left()*m, curr_img_area.top()*m, curr_img_area.width(), curr_img_area.height()};
// }

void CVolumeViewer::renderVisible(bool force)
{
    if (!volume || !volume->zarrDataset() || !slice)
        return;
    
    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();
    
    // printf("curr area %f %f \n", );
    
    //nothing to see her, move along
    if (!force && QRectF(curr_img_area).contains(bbox))
        return;
    
    curr_img_area = {bbox.left()-128,bbox.top()-128, bbox.width()+256, bbox.height()+256};
    
    cv::Mat img = render_area({curr_img_area.x(), curr_img_area.y(), curr_img_area.width(), curr_img_area.height()});
    
    QImage qimg = Mat2QImage(img);
    
    QPixmap pixmap = QPixmap::fromImage(qimg, fSkipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);
    //     
    // Add the QPixmap to the scene as a QGraphicsPixmapItem
    if (!fBaseImageItem)
        fBaseImageItem = fScene->addPixmap(pixmap);
    else
        fBaseImageItem->setPixmap(pixmap);
    
    if (!center_marker)
        center_marker = fScene->addEllipse({-10,-10,20,20}, QPen(Qt::yellow, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));

    
    center_marker->setParentItem(fBaseImageItem);
    
    fBaseImageItem->setOffset(curr_img_area.topLeft());
    
    /*for(auto other_plane : other_slices) {
        std::vector<std::vector<cv::Point2f>> segments_xy;
        
        find_intersect_segments(segments_xy, other_plane, slice, curr_img_area, render_scale, coord_scale);
        
        for (auto seg : segments_xy)
            for (auto p : seg)
            {f
                auto item = fGraphicsView->scene()->addEllipse({p.x-2,p.y-2,4,4}, QPen(Qt::yellow, 1));
                other_slice_items.push_back(item);
                item->setParentItem(fBaseImageItem);
            }
    }*/
    
    PlaneCoords *slice_plane = dynamic_cast<PlaneCoords*>(slice);
    if (!_slice_vis_valid && seg_tool && slice_plane) {
#pragma omp parallel for
        for (auto &wp : seg_tool->control_points) {
            float dist = slice_plane->pointDist(wp);
            
            if (dist > 0.5)
                continue;
            
            cv::Vec3f p = slice_plane->project(wp, 1.0, _ds_scale);
            
#pragma omp critical
            {
                auto item = fGraphicsView->scene()->addEllipse({p[0]-1,p[1]-1,2,2}, QPen(Qt::green, 1));
                //FIXME rename/clean
                slice_vis_items.push_back(item);
                item->setParentItem(fBaseImageItem);
            }
        }
        
        if (seg_tool->control_points.size())
            _slice_vis_valid = true;
    }
    
    update();
}

void CVolumeViewer::onScrolled()
{
    renderVisible();
}

cv::Mat CVolumeViewer::getCoordSlice()
{
    return cv::Mat();
}


void CVolumeViewer::addIntersectVisSlice(PlaneCoords *slice_)
{
    other_slices.push_back(slice_);
    
    OnSliceChanged();
}


void CVolumeViewer::setSegTool(ControlPointSegmentator *tool)
{
    seg_tool = tool;
}
