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
#define ZOOM_FACTOR 1.414213562373095

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
    , fScaleFactor(1.0)
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

    // UpdateButtons();
    update();
}

/*bool CVolumeViewer::eventFilter(QObject* watched, QEvent* event)
{
    // Wheel events
    if (watched == fGraphicsView || (fGraphicsView && watched == fGraphicsView->viewport()) && event->type() == QEvent::Wheel) {

        QWheelEvent* wheelEvent = static_cast<QWheelEvent*>(event);

        // Ctrl = Zoom in/out
        if (QApplication::keyboardModifiers() == Qt::ControlModifier) {
            int numDegrees = wheelEvent->angleDelta().y() / 8;

            if (numDegrees > 0) {
                OnZoomInClicked();
            } else if (numDegrees < 0) {
                OnZoomOutClicked();
            }
            return true;
        }
        // Shift = Scan through slices
        else if (QApplication::keyboardModifiers() == Qt::ShiftModifier) {
            int numDegrees = wheelEvent->angleDelta().y() / 8;

            sendShiftNormal(0.1*numDegrees*slice->normal);
            return true;
        }
    }
    return QWidget::eventFilter(watched, event);
}*/

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
    if (modifiers & Qt::ShiftModifier) {
        slice->setOffsetZ(slice->offsetZ()+steps);
        renderVisible(true);
    }
    
    else {
        float zoom = pow(ZOOM_FACTOR, steps);
        
        scale *= zoom;
        round_scale(scale);
        
        curr_img_area = {0,0,0,0};
        QPointF center = visible_center(fGraphicsView) * zoom;
        
        //FIXME get correct size for slice!
        int max_size = std::max(volume->sliceWidth(), std::max(volume->numSlices(), volume->sliceHeight()))*scale + 512;
        fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
        
        fGraphicsView->centerOn(center);
        renderVisible();
    }
}

void CVolumeViewer::OnVolumeChanged(volcart::Volume::Pointer volume_)
{
    volume = volume_;
    
    printf("sizes %d %d %d\n", volume_->sliceWidth(), volume_->sliceHeight(), volume_->numSlices());
    
    int max_size = std::max(volume_->sliceWidth(), std::max(volume_->numSlices(), volume_->sliceHeight()))*scale + 512;
    printf("max size %d\n", max_size);
    fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);

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
    cv::Vec3f vol_loc = loc3d_at_imgpos(volume.get(), slice, scene_loc, scale);

    sendVolumeClicked(vol_loc, buttons, modifiers);
}

void CVolumeViewer::setCache(ChunkCache *cache_)
{
    cache = cache_;
}

void CVolumeViewer::loadSlice()
{
//     QImage aImgQImage;
//     cv::Mat aImgMat;
//     
//     if (volume && volume->zarrDataset()) {
//         aImgMat = getCoordSlice();
// 
//         if (aImgMat.isContinuous() && aImgMat.type() == CV_16U) {
//             // create QImage directly backed by cv::Mat buffer
//             aImgQImage = QImage(
//                 aImgMat.ptr(), aImgMat.cols, aImgMat.rows, aImgMat.step,
//                 QImage::Format_Grayscale16);
//         } else
//             aImgQImage = Mat2QImage(aImgMat);
//             
//         SetImage(aImgQImage);
//     }
//     
//     UpdateButtons();
}

void CVolumeViewer::setSlice(CoordGenerator *slice_)
{
    slice = slice_;
    OnSliceChanged();
}

void CVolumeViewer::OnSliceChanged()
{
    curr_img_area = {0,0,0,0};
    renderVisible();
}


void calc_scales(float scale, float &render_scale, float &coord_scale, int &sd_idx, int max_idx)
{
    sd_idx = 1;
    
    coord_scale = 0.5;
    while (0.5*coord_scale >= scale && sd_idx < max_idx-1) {
        sd_idx++;
        coord_scale *= 0.5;
    }
    
    render_scale = scale/coord_scale;
}

cv::Mat CVolumeViewer::render_area()
{   
    xt::xarray<float> coords;
    xt::xarray<uint8_t> img;

    int sd_idx;
    float render_scale, coord_scale;
    cv::Rect roi;

    currRoi(roi, render_scale, coord_scale, sd_idx);

    slice->gen_coords(coords, roi, render_scale, coord_scale);
    readInterpolated3D(img, volume->zarrDataset(sd_idx), coords, cache);
    cv::Mat m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    
    return m.clone();
}

void CVolumeViewer::currRoi(cv::Rect &roi, float &render_scale, float &coord_scale, int &sd_idx) const
{  
    calc_scales(scale, render_scale, coord_scale, sd_idx, volume->numScales()-1);
    
    float m = coord_scale/scale;
    
    roi = {curr_img_area.left()*m, curr_img_area.top()*m, curr_img_area.width(), curr_img_area.height()};
}

void CVolumeViewer::renderVisible(bool force)
{
    if (!volume || !volume->zarrDataset() || !slice)
        return;
    
    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();
    
    //nothing to see her, move along
    if (!force && QRectF(curr_img_area).contains(bbox))
        return;
    
    curr_img_area = {bbox.left()-128,bbox.top()-128, bbox.width()+256, bbox.height()+256};
    
    cv::Mat img = render_area();
    
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
    
    for(auto &item : other_slice_items) {
        fScene->removeItem(item);
        delete item;
    }
    other_slice_items.resize(0);
    
    cv::Rect roi;
    int sd;
    float render_scale, coord_scale;
    currRoi(roi, render_scale, coord_scale, sd);

    for(auto other_plane : other_slices) {
        std::vector<std::vector<cv::Point2f>> segments_xy;
        
        
        if (!roi.width || !roi.height)
            return;

        find_intersect_segments(segments_xy, other_plane, slice, roi, render_scale, coord_scale);
        
        for (auto seg : segments_xy)
            for (auto p : seg)
            {
                auto item = fGraphicsView->scene()->addEllipse({p.x-2,p.y-2,4,4}, QPen(Qt::yellow, 1));
                other_slice_items.push_back(item);
                item->setParentItem(fBaseImageItem);
            }
    }
    
    PlaneCoords *slice_plane = dynamic_cast<PlaneCoords*>(slice);
    if (seg_tool && slice_plane) {
#pragma omp parallel for
        for (auto &wp : seg_tool->control_points) {
            float dist = slice_plane->pointDist(wp);
            
            if (dist > 0.5)
                continue;
            
            cv::Vec3f p = slice_plane->project(wp, roi, render_scale, coord_scale);
            
#pragma omp critical
            {
                auto item = fGraphicsView->scene()->addEllipse({p[0]-1,p[1]-1,2,2}, QPen(Qt::green, 1));
                //FIXME rename/clean
                other_slice_items.push_back(item);
                item->setParentItem(fBaseImageItem);
            }
        }
    }
    
    update();
}

void CVolumeViewer::onScrolled()
{
    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();
    QRect viewrect = fGraphicsView->viewport()->geometry();
    // printf("scrolled to scene pos %f x %f bbox %f %f %f %f from %d %d %d %d\n", visible_center(fGraphicsView).x(), visible_center(fGraphicsView).y(), bbox.left() , bbox.top() , bbox.width(), bbox.height(), viewrect.left(), viewrect.top(), viewrect.width(), viewrect.height());
    
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
