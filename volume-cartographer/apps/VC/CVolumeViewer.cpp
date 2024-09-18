// CVolumeViewer.cpp
// Chao Du 2015 April
#include "CVolumeViewer.hpp"
#include "UDataManipulateUtils.hpp"
#include "HBase.hpp"

#include "vc/core/util/Slicing.hpp"

using namespace ChaoVis;
using qga = QGuiApplication;

#define BGND_RECT_MARGIN 8
#define DEFAULT_TEXT_COLOR QColor(255, 255, 120)
#define ZOOM_FACTOR 1.15

// Constructor
CVolumeViewerView::CVolumeViewerView(QWidget* parent)
: QGraphicsView(parent)
{
    timerTextAboveCursor = new QTimer(this);
    connect(timerTextAboveCursor, &QTimer::timeout, this, &CVolumeViewerView::hideTextAboveCursor);
    timerTextAboveCursor->setSingleShot(true);
}

void CVolumeViewerView::setup()
{
    textAboveCursor = new QGraphicsTextItem("", 0);
    textAboveCursor->setFlag(QGraphicsItem::ItemIgnoresTransformations);
    textAboveCursor->setZValue(100);
    textAboveCursor->setVisible(false);
    textAboveCursor->setDefaultTextColor(DEFAULT_TEXT_COLOR);
    scene()->addItem(textAboveCursor);

    QFont f;
    f.setPointSize(f.pointSize() + 2);
    textAboveCursor->setFont(f);

    backgroundBehindText = new QGraphicsRectItem();
    backgroundBehindText->setFlag(QGraphicsItem::ItemIgnoresTransformations);
    backgroundBehindText->setPen(Qt::NoPen);
    backgroundBehindText->setZValue(99);
    scene()->addItem(backgroundBehindText);
}

void CVolumeViewerView::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_W) {
        rangeKeyPressed = true;
        event->accept();
    } else if (event->key() == Qt::Key_R) {
        curvePanKeyPressed = true;
        event->accept();
    } else if (event->key() == Qt::Key_S) {
        rotateKeyPressed = true;
        event->accept();
    }
}

void CVolumeViewerView::keyReleaseEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_W)  {
        rangeKeyPressed = false;
        event->accept();
    } else if (event->key() == Qt::Key_R) {
        curvePanKeyPressed = false;
        event->accept();
    } else if (event->key() == Qt::Key_S) {
        rotateKeyPressed = false;
        event->accept();
    }
}

void CVolumeViewerView::showTextAboveCursor(const QString& value, const QString& label, const QColor& color)
{
    // Without this check, when you start VC with auto-load the initial slice will not be in the center of
    // volume viewer, because during loading the initilization of the impact range slider and its callback slots
    // will already move the position/scrollbars of the viewer and therefore the image is no longer centered.
    if (!isVisible()) {
        return;
    }

    timerTextAboveCursor->start(150);

    QFontMetrics fm(textAboveCursor->font());
    QPointF p = mapToScene(mapFromGlobal(QPoint(QCursor::pos().x() + 10, QCursor::pos().y())));

    textAboveCursor->setVisible(true);
    textAboveCursor->setHtml("<b>" + value + "</b><br>" + label);
    textAboveCursor->setPos(p);
    textAboveCursor->setDefaultTextColor(color);

    backgroundBehindText->setVisible(true);
    backgroundBehindText->setPos(p);
    backgroundBehindText->setRect(0, 0, fm.horizontalAdvance((label.isEmpty() ? value : label)) + BGND_RECT_MARGIN, fm.height() * (label.isEmpty() ? 1 : 2) + BGND_RECT_MARGIN);
    backgroundBehindText->setBrush(QBrush(QColor(
        (2 * 125 + color.red())   / 3,
        (2 * 125 + color.green()) / 3,
        (2 * 125 + color.blue())  / 3,
    200)));
}

void CVolumeViewerView::hideTextAboveCursor()
{
    textAboveCursor->setVisible(false);
    backgroundBehindText->setVisible(false);
}

void CVolumeViewerView::showCurrentImpactRange(int range)
{
    showTextAboveCursor(QString::number(range), "", QColor(255, 120, 110)); // tr("Impact Range")
}

void CVolumeViewerView::showCurrentScanRange(int range)
{
    showTextAboveCursor(QString::number(range), "", QColor(160, 180, 255)); // tr("Scan Range")
}

void CVolumeViewerView::showCurrentSliceIndex(int slice, bool highlight)
{
    showTextAboveCursor(QString::number(slice), "", (highlight ? QColor(255, 50, 20) : QColor(255, 220, 30))); // tr("Slice")
}

// Constructor
CVolumeViewer::CVolumeViewer(QWidget* parent)
    : QWidget(parent)
    , fCanvas(nullptr)
    , fScrollArea(nullptr)
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
    fZoomInBtn = new QPushButton(tr("Zoom In"), this);
    fZoomOutBtn = new QPushButton(tr("Zoom Out"), this);
    fResetBtn = new QPushButton(tr("Reset"), this);
    fNextBtn = new QPushButton(tr("Next Slice"), this);
    fPrevBtn = new QPushButton(tr("Previous Slice"), this);

    fImageRotationSpin = new QSpinBox(this);
    fImageRotationSpin->setMinimum(-360);
    fImageRotationSpin->setMaximum(360);
    fImageRotationSpin->setSuffix("°");
    fImageRotationSpin->setEnabled(true);
    connect(fImageRotationSpin, SIGNAL(editingFinished()), this, SLOT(OnImageRotationSpinChanged()));

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
    fGraphicsView->setRenderHint(QPainter::Antialiasing);
    setFocusProxy(fGraphicsView);

    // Create graphics scene
    fScene = new QGraphicsScene(this);

    // Set the scene
    fGraphicsView->setScene(fScene);
    fGraphicsView->setup();

    fGraphicsView->viewport()->installEventFilter(this);
    fGraphicsView->setDragMode(QGraphicsView::ScrollHandDrag);

    fButtonsLayout = new QHBoxLayout;
    fButtonsLayout->addWidget(fZoomInBtn);
    fButtonsLayout->addWidget(fZoomOutBtn);
    fButtonsLayout->addWidget(fResetBtn);
    fButtonsLayout->addWidget(fImageRotationSpin);
    // fButtonsLayout->addWidget(fAxisCombo);
    // Add some space between the slice spin box and the curve tools (color, checkboxes, ...)
    fButtonsLayout->addSpacerItem(new QSpacerItem(1, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));

    connect(fZoomInBtn, SIGNAL(clicked()), this, SLOT(OnZoomInClicked()));
    connect(fZoomOutBtn, SIGNAL(clicked()), this, SLOT(OnZoomOutClicked()));
    connect(fResetBtn, SIGNAL(clicked()), this, SLOT(OnResetClicked()));

    QSettings settings("VC.ini", QSettings::IniFormat);
    fCenterOnZoomEnabled = settings.value("viewer/center_on_zoom", false).toInt() != 0;
    fScrollSpeed = settings.value("viewer/scroll_speed", false).toInt();
    fSkipImageFormatConv = settings.value("perf/chkSkipImageFormatConvExp", false).toBool();

    QVBoxLayout* aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->addWidget(fGraphicsView);
    aWidgetLayout->addLayout(fButtonsLayout);

    setLayout(aWidgetLayout);
    
    UpdateButtons();
}

// Destructor
CVolumeViewer::~CVolumeViewer(void)
{
    deleteNULL(fGraphicsView);
    deleteNULL(fScene);
    deleteNULL(fScrollArea);
    deleteNULL(fZoomInBtn);
    deleteNULL(fZoomOutBtn);
    deleteNULL(fResetBtn);
    deleteNULL(fImageRotationSpin);
}

void CVolumeViewer::SetButtonsEnabled(bool state)
{
    fZoomOutBtn->setEnabled(state);
    fZoomInBtn->setEnabled(state);
    fImageRotationSpin->setEnabled(state);
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

    UpdateButtons();
    update();
}

bool CVolumeViewer::eventFilter(QObject* watched, QEvent* event)
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

            if (fCenterOnZoomEnabled) {
                CenterOn(fGraphicsView->mapToScene(wheelEvent->position().toPoint()));
            }

            return true;
        }
        // Shift = Scan through slices
        else if (QApplication::keyboardModifiers() == Qt::ShiftModifier) {
            int numDegrees = wheelEvent->angleDelta().y() / 8;

            if (numDegrees > 0) {
                SendSignalSliceShift(fScanRange, axis);
            } else if (numDegrees < 0) {
                SendSignalSliceShift(-fScanRange, axis);
            }
            return true;
        }
        // Rotate key pressed
        else if (fGraphicsView->isRotateKyPressed()) {
            int delta = wheelEvent->angleDelta().y() / 22;
            fGraphicsView->rotate(delta);
            currentRotation += delta;
            currentRotation = currentRotation % 360;
            fImageRotationSpin->setValue(currentRotation);
            return true;
        } 
        // View scrolling
        else {
            // If there is no valid scroll speed override value set, we rely
            // on the default handling of Qt, so we pass on the event.
            if (fScrollSpeed > 0) {
                // We have to add the two values since when pressing AltGr as the modifier, 
                // the X component seems to be set by Qt
                int delta = wheelEvent->angleDelta().x() + wheelEvent->angleDelta().y();
                if (delta == 0) {
                    return true;
                }

                // Taken from QGraphicsView Qt source logic
                const bool horizontal = qAbs(wheelEvent->angleDelta().x()) > qAbs(wheelEvent->angleDelta().y());
                if (QApplication::keyboardModifiers() == Qt::AltModifier || horizontal) {
                    fGraphicsView->horizontalScrollBar()->setValue(
                        fGraphicsView->horizontalScrollBar()->value() + fScrollSpeed * ((delta < 0) ? 1 : -1));
                } else {
                    fGraphicsView->verticalScrollBar()->setValue(
                        fGraphicsView->verticalScrollBar()->value() + fScrollSpeed * ((delta < 0) ? 1 : -1));
                }
                return true;
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void CVolumeViewer::ScaleImage(double nFactor)
{
    fScaleFactor *= nFactor;
    fGraphicsView->scale(nFactor, nFactor);

    UpdateButtons();
}

void CVolumeViewer::CenterOn(const QPointF& point)
{
    fGraphicsView->centerOn(point);
}

void CVolumeViewer::SetRotation(int degrees)
{
    if (currentRotation != degrees) {
        auto delta = (currentRotation - degrees) * -1;
        fGraphicsView->rotate(delta);
        currentRotation += delta;
        currentRotation = currentRotation % 360;
        fImageRotationSpin->setValue(currentRotation);
    }
}

void CVolumeViewer::Rotate(int delta)
{
    SetRotation(currentRotation + delta);
}

void CVolumeViewer::ResetRotation()
{
    fGraphicsView->rotate(-currentRotation);
    currentRotation = 0;
    fImageRotationSpin->setValue(currentRotation);
}

// Handle zoom in click
void CVolumeViewer::OnZoomInClicked(void)
{
    if (fZoomInBtn->isEnabled()) {
        ScaleImage(ZOOM_FACTOR);
    }
}

// Handle zoom out click
void CVolumeViewer::OnZoomOutClicked(void)
{
    if (fZoomOutBtn->isEnabled()) {
        ScaleImage(1 / ZOOM_FACTOR);
    }
}

void CVolumeViewer::OnResetClicked(void)
{
    fGraphicsView->resetTransform();
    fScaleFactor = 1.0;
    currentRotation = 0;
    fImageRotationSpin->setValue(currentRotation);

    UpdateButtons();
}

// Handle image rotation change
void CVolumeViewer::OnImageRotationSpinChanged(void)
{
    SetRotation(fImageRotationSpin->value());
}

void CVolumeViewer::OnViewAxisChanged(void)
{    
    // axis = fAxisCombo->currentData().toInt();
    
    // loadSlice();
}

void CVolumeViewer::OnLocChanged(int x_, int y_, int z_)
{
//     bool have_change = false;
//     int slice_index = 0;
//     
//     if (loc[0] != x_ && axis == 0)
//         have_change = true;
//     if (loc[1] != y_ && axis == 1)
//         have_change = true;
//     if (loc[2] != z_ && axis == 2)
//         have_change = true;
//     
//     loc[0] = x_;
//     loc[1] = y_;
//     loc[2] = z_;
//     
//     if (have_change)
        loadSlice();
}

// Update the status of the buttons
void CVolumeViewer::UpdateButtons(void)
{
    fZoomInBtn->setEnabled(fImgQImage != nullptr && fScaleFactor < 10.0);
    fZoomOutBtn->setEnabled(fImgQImage != nullptr && fScaleFactor > 0.05);
    fResetBtn->setEnabled(fImgQImage != nullptr && fabs(fScaleFactor - 1.0) > 1e-6);
}

// Reset the viewer
void CVolumeViewer::Reset()
{
    if (fBaseImageItem) {
        delete fBaseImageItem;
        fBaseImageItem = nullptr;
    }

    OnResetClicked(); // to reset zoom
}

void CVolumeViewer::OnVolumeChanged(volcart::Volume::Pointer volume_)
{
    volume = volume_;
    loadSlice();
}

void CVolumeViewer::setCache(ChunkCache *cache_)
{
    cache = cache_;
}

void CVolumeViewer::loadSlice()
{
    QImage aImgQImage;
    cv::Mat aImgMat;
    
    if (volume && volume->zarrDataset()) {
        aImgMat = getCoordSlice();

        if (aImgMat.isContinuous() && aImgMat.type() == CV_16U) {
            // create QImage directly backed by cv::Mat buffer
            aImgQImage = QImage(
                aImgMat.ptr(), aImgMat.cols, aImgMat.rows, aImgMat.step,
                QImage::Format_Grayscale16);
        } else
            aImgQImage = Mat2QImage(aImgMat);
            
        SetImage(aImgQImage);
    }
    
    UpdateButtons();
}

void CVolumeViewer::setSlice(CoordGenerator *slice_)
{
    slice = slice_;
    OnSliceChanged();
}

void CVolumeViewer::OnSliceChanged()
{
    //TODO update slice if we are in slice view!
    if (axis == 3)
        loadSlice();
}

cv::Mat CVolumeViewer::getCoordSlice()
{
    xt::xarray<float> coords;
    xt::xarray<uint8_t> img;
    
    slice->gen_coords(coords, 1000, 1000);
    std::cout << "start read" << cache << std::endl;
    readInterpolated3D(img, volume->zarrDataset() ,coords, cache);
    std::cout << "done read" << cache << std::endl;
    // readInterpolated3D(img,ds,coords);
    cv::Mat m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    
    return m.clone();
}
