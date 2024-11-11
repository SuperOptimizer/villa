// CVolumeViewer.cpp
// Chao Du 2015 April
#include "CVolumeViewer.hpp"
#include "UDataManipulateUtils.hpp"
#include "HBase.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>

#include "CVolumeViewerView.hpp"
#include "SegmentationStruct.hpp"
#include "CSurfaceCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include "OpChain.hpp"

using namespace ChaoVis;
using qga = QGuiApplication;

#define BGND_RECT_MARGIN 8
#define DEFAULT_TEXT_COLOR QColor(255, 255, 120)
// #define ZOOM_FACTOR 1.148698354997035
#define ZOOM_FACTOR 2.0 //1.414213562373095

CVolumeViewer::CVolumeViewer(CSurfaceCollection *col, QWidget* parent)
    : QWidget(parent)
    , fGraphicsView(nullptr)
    , fBaseImageItem(nullptr)
    , _surf_col(col)
{
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
    connect(fGraphicsView, &CVolumeViewerView::sendCursorMove, this, &CVolumeViewer::onCursorMove);
    connect(fGraphicsView, &CVolumeViewerView::sendPanRelease, this, &CVolumeViewer::onPanRelease);

    // Create graphics scene
    fScene = new QGraphicsScene({-2500,-2500,5000,5000}, this);

    // Set the scene
    fGraphicsView->setScene(fScene);

    QSettings settings("VC.ini", QSettings::IniFormat);
    // fCenterOnZoomEnabled = settings.value("viewer/center_on_zoom", false).toInt() != 0;
    // fScrollSpeed = settings.value("viewer/scroll_speed", false).toInt();
    fSkipImageFormatConv = settings.value("perf/chkSkipImageFormatConvExp", false).toBool();

    QVBoxLayout* aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->addWidget(fGraphicsView);

    setLayout(aWidgetLayout);


    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : white; }");
    _lbl->move(10,5);
}

// Destructor
CVolumeViewer::~CVolumeViewer(void)
{
    deleteNULL(fGraphicsView);
    deleteNULL(fScene);
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


void scene2vol(cv::Vec3f &p, cv::Vec3f &n, Surface *_surf, const std::string &_surf_name, CSurfaceCollection *_surf_col, const QPointF &scene_loc, const cv::Vec2f &_vis_center, float _ds_scale)
{
    //for PlaneSurface we work with absolute coordinates only
    if (dynamic_cast<PlaneSurface*>(_surf)) {
        cv::Vec3f surf_loc = {scene_loc.x()/_ds_scale, scene_loc.y()/_ds_scale,0};
        
        SurfacePointer *ptr = _surf->pointer();
        
        n = _surf->normal(ptr, surf_loc);
        p = _surf->coord(ptr, surf_loc);
    }
    //FIXME quite some assumptions ...
    else if (_surf_name == "segmentation") {
        // assert(_ptr);
        assert(dynamic_cast<OpChain*>(_surf));
        
        QuadSurface* crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation")); 
        
        cv::Vec3f delta = {(scene_loc.x()-_vis_center[0])/_ds_scale, (scene_loc.y()-_vis_center[1])/_ds_scale,0};
        
        //NOTE crop center and original scene _ptr are off by < 0.5 voxels?
        SurfacePointer *ptr = crop->pointer();
        n = crop->normal(ptr, delta);
        p = crop->coord(ptr, delta);
    }
}

void CVolumeViewer::onCursorMove(QPointF scene_loc)
{
    if (!_surf)
        return;

    POI *cursor = _surf_col->poi("cursor");
    if (!cursor)
        cursor = new POI;
    
    cv::Vec3f p, n;
    scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _ds_scale);
    cursor->p = p;
    
    _surf_col->setPOI("cursor", cursor);
}

void CVolumeViewer::onZoom(int steps, QPointF scene_loc, Qt::KeyboardModifiers modifiers)
{
    //TODO don't invalidate if only _scene_scale chagned
    invalidateVis();
    invalidateIntersect();
    
    if (!_surf)
        return;
    
    if (modifiers & Qt::ShiftModifier) {
        _z_off += steps;
        renderVisible(true);
    }
    else {
        float zoom = pow(ZOOM_FACTOR, steps);
        
        _scale *= zoom;
        round_scale(_scale);

        if (dynamic_cast<PlaneSurface*>(_surf))
            _min_scale = pow(2.0,1.-volume->numScales());
        else
            _min_scale = std::max(pow(2.0,1.-volume->numScales()), 0.5);
        
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
        
        QTransform M = fGraphicsView->transform();
        if (_scene_scale != M.m11()) {
            double delta_scale = _scene_scale/M.m11();
            M.scale(delta_scale,delta_scale);
            fGraphicsView->setTransform(M);
        }
        
        curr_img_area = {0,0,0,0};
        QPointF center = visible_center(fGraphicsView) * zoom;
        
        //FIXME get correct size for slice!
        int max_size = std::max(volume->sliceWidth(), std::max(volume->numSlices(), volume->sliceHeight()))*_ds_scale + 512;
        fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
        
        fGraphicsView->centerOn(center);
        renderVisible();
    }

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));
    
    renderIntersections();
}

void CVolumeViewer::OnVolumeChanged(volcart::Volume::Pointer volume_)
{
    volume = volume_;
    
    // printf("sizes %d %d %d\n", volume_->sliceWidth(), volume_->sliceHeight(), volume_->numSlices());
    
    int max_size = std::max(volume_->sliceWidth(), std::max(volume_->numSlices(), volume_->sliceHeight()))*_ds_scale + 512;
    // printf("max size %d\n", max_size);
    fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
    
    if (volume->numScales() >= 2) {
        //FIXME currently hardcoded
        _max_scale = 0.5;
        _min_scale = pow(2.0,1.-volume->numScales());
    }
    else {
        //FIXME currently hardcoded
        _max_scale = 1.0;
        _min_scale = 1.0;
    }

    _ds_scale = _max_scale;
    _ds_sd_idx = -log2(_ds_scale);
    _scene_scale = _scale/_ds_scale;

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));

    renderVisible(true);
}

void CVolumeViewer::onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    if (!_surf)
        return;
    
    cv::Vec3f p, n;
    scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _ds_scale);

    //for PlaneSurface we work with absolute coordinates only
    if (dynamic_cast<PlaneSurface*>(_surf))
        sendVolumeClicked(p, n, _surf, buttons, modifiers);
    //FIXME quite some assumptions ...
    else if (_surf_name == "segmentation")
        sendVolumeClicked(p, n, _surf_col->surface("visible_segmentation"), buttons, modifiers);
    else
        std::cout << "FIXME: onVolumeClicked()" << std::endl;
}

void CVolumeViewer::setCache(ChunkCache *cache_)
{
    cache = cache_;
}

void CVolumeViewer::setSurface(const std::string &name)
{
    _surf_name = name;
    _surf = nullptr;
    _ptr = nullptr;
    onSurfaceChanged(name, _surf_col->surface(name));
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

void CVolumeViewer::invalidateIntersect(const std::string &name)
{
    if (!name.size() || name == _surf_name) {
        for(auto &pair : _intersect_items) {
            for(auto &item : pair.second) {
                fScene->removeItem(item);
                delete item;
            }
        }
        _intersect_items.clear();
    }
    else if (_intersect_items.count(name)) {
        for(auto &item : _intersect_items[name]) {
            fScene->removeItem(item);
            delete item;
        }
        _intersect_items.erase(name);
    }
}


void CVolumeViewer::onIntersectionChanged(std::string a, std::string b, Intersection *intersection)
{
    if (_ignore_intersect_change && intersection == _ignore_intersect_change)
        return;

    if (!_intersect_tgts.count(a) || !_intersect_tgts.count(b))
        return;

    //FIXME fix segmentation vs visible_segmentation naming and usage ..., think about dependency chain ..
    if (a == _surf_name || (_surf_name == "segmentation" && a == "visible_segmentation"))
        invalidateIntersect(b);
    else if (b == _surf_name || (_surf_name == "segmentation" && b == "visible_segmentation"))
        invalidateIntersect(a);
    
    renderIntersections();
}


std::set<std::string> CVolumeViewer::intersects()
{
    return _intersect_tgts;
}

void CVolumeViewer::setIntersects(const std::set<std::string> &set)
{
    _intersect_tgts = set;
    
    renderIntersections();
}

void CVolumeViewer::onSurfaceChanged(std::string name, Surface *surf)
{
    if (_surf_name == name) {
        _surf = surf;
        if (!_surf)
            fScene->clear();
        else
            invalidateVis();
    }

    //FIXME do not re-render surf if only segmentation changed?
    if (name == _surf_name) {
        curr_img_area = {0,0,0,0};
        renderVisible();
    }

    invalidateIntersect(name);
    renderIntersections();
}

QGraphicsItem *cursorItem()
{
    QPen pen(QBrush(Qt::cyan), 3);
    QGraphicsLineItem *parent = new QGraphicsLineItem(-10, 0, -5, 0);
    parent->setZValue(10);
    parent->setPen(pen);
    QGraphicsLineItem *line = new QGraphicsLineItem(10, 0, 5, 0, parent);
    line->setPen(pen);
    line = new QGraphicsLineItem(0, -10, 0, -5, parent);
    line->setPen(pen);
    line = new QGraphicsLineItem(0, 10, 0, 5, parent);
    line->setPen(pen);
    
    return parent;
}

QGraphicsItem *crossItem()
{
    QPen pen(QBrush(Qt::red), 1);
    QGraphicsLineItem *parent = new QGraphicsLineItem(-5, -5, 5, 5);
    parent->setZValue(10);
    parent->setPen(pen);
    QGraphicsLineItem *line = new QGraphicsLineItem(-5, 5, 5, -5, parent);
    line->setPen(pen);
    
    return parent;
}

//TODO make poi tracking optional and configurable
void CVolumeViewer::onPOIChanged(std::string name, POI *poi)
{    
    if (!poi)
        return;
    
    if (name == "focus") {
        PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);
        
        if (!plane)
            return;
        
        fGraphicsView->centerOn(0,0);
        
        if (poi->p == plane->origin())
            return;
        
        plane->setOrigin(poi->p);
        
        _surf_col->setSurface(_surf_name, plane);
    }
    else if (name == "cursor") {
        PlaneSurface *slice_plane = dynamic_cast<PlaneSurface*>(_surf);
        QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"));
        
        cv::Vec3f sp;
        float dist = -1;
        if (slice_plane) {            
            dist = slice_plane->pointDist(poi->p);
            sp = slice_plane->project(poi->p, 1.0, _ds_scale);
        }
        else if (_surf_name == "segmentation" && crop)
        {
            SurfacePointer *ptr = crop->pointer();
            dist = crop->pointTo(ptr, poi->p, 20.0);
            sp = crop->loc(ptr)*_ds_scale + cv::Vec3f(_vis_center[0],_vis_center[1],0);
        }
        
        if (!_cursor) {
            _cursor = cursorItem();
            fScene->addItem(_cursor);
        }
        
        if (dist != -1) {
            if (dist < 20.0/_scale) {
                _cursor->setPos(sp[0], sp[1]);
                _cursor->setOpacity(1.0-dist*_scale/20.0);
            }
            else
                _cursor->setOpacity(0.0);
        }
    }
}

cv::Mat CVolumeViewer::render_area(const cv::Rect &roi)
{
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> img;

    //PlaneSurface use absolute positioning to simplify intersection logic
    if (dynamic_cast<PlaneSurface*>(_surf)) {
        _surf->gen(&coords, nullptr, roi.size(), nullptr, _ds_scale, {roi.x, roi.y, _z_off});
    }
    else {
        cv::Vec2f roi_c = {roi.x+roi.width/2, roi.y + roi.height/2};

        if (!_ptr) {
            _ptr = _surf->pointer();
            _vis_center = roi_c;
        }
        else {
            cv::Vec3f diff = {roi_c[0]-_vis_center[0],roi_c[1]-_vis_center[1],0};
            _surf->move(_ptr, diff/_ds_scale);
            _vis_center = roi_c;
        }

        _surf->gen(&coords, nullptr, roi.size(), _ptr, _ds_scale, {-roi.width/2, -roi.height/2, _z_off});
        
        if (_surf_name == "segmentation") {
            invalidateIntersect();

            QuadSurface *old_crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"));
            
            QuadSurface *crop = new QuadSurface(coords.clone(), {_ds_scale, _ds_scale});
            _surf_col->setSurface("visible_segmentation", crop);
            if (old_crop)
                delete old_crop;
        }
    }

    readInterpolated3D(img, volume->zarrDataset(_ds_sd_idx), coords*_ds_scale, cache);
    
    return img;
}

class LifeTime
{
public:
    LifeTime(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~LifeTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

void CVolumeViewer::renderVisible(bool force)
{
    if (!volume || !volume->zarrDataset() || !_surf)
        return;
    
    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();
    
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
    
    if (!_center_marker) {
        _center_marker = fScene->addEllipse({-10,-10,20,20}, QPen(Qt::yellow, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _center_marker->setZValue(11);
    }

    _center_marker->setParentItem(fBaseImageItem);
    
    fBaseImageItem->setOffset(curr_img_area.topLeft());

    invalidateIntersect();
    renderIntersections();
}

struct vec3f_hash {
    size_t operator()(cv::Vec3f p) const
    {
        size_t hash1 = std::hash<float>{}(p[0]);
        size_t hash2 = std::hash<float>{}(p[1]);
        size_t hash3 = std::hash<float>{}(p[2]);
        
        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        return hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
};

void CVolumeViewer::renderIntersections()
{
    if (!volume || !volume->zarrDataset() || !_surf)
        return;
    
    std::vector<std::string> remove;
    for (auto &pair : _intersect_items)
        if (!_intersect_tgts.count(pair.first)) {
            for(auto &item : pair.second) {
                fScene->removeItem(item);
                delete item;
            }
            remove.push_back(pair.first);
        }
    for(auto key : remove)
        _intersect_items.erase(key);

    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);
    
    if (_z_off)
        return;
    
    if (plane) {
        cv::Rect plane_roi = {curr_img_area.x()/_ds_scale, curr_img_area.y()/_ds_scale, curr_img_area.width()/_ds_scale, curr_img_area.height()/_ds_scale};

        cv::Vec3f corner = plane->coord(nullptr, {plane_roi.x, plane_roi.y, 0.0});
        Rect3D view_bbox = {corner, corner};
        view_bbox = expand_rect(view_bbox, plane->coord(nullptr, {plane_roi.br().x, plane_roi.y, 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(nullptr, {plane_roi.x, plane_roi.br().y, 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(nullptr, {plane_roi.br().x, plane_roi.br().y, 0}));

        std::vector<std::string> intersect_cands;
        std::vector<std::string> intersect_tgts_v;

        for (auto key : _intersect_tgts)
            intersect_tgts_v.push_back(key);

#pragma omp parallel for
        for(int n=0;n<intersect_tgts_v.size();n++) {
            std::string key = intersect_tgts_v[n];
            if (!_intersect_items.count(key) && dynamic_cast<QuadSurface*>(_surf_col->surface(key))) {
                QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));

                if (intersect(view_bbox, segmentation->bbox()))
#pragma omp critical
                    intersect_cands.push_back(key);
                else
#pragma omp critical
                    _intersect_items[key] = {};
            }
        }

        std::vector<std::vector<std::vector<cv::Vec3f>>> intersections(intersect_cands.size());

#pragma omp parallel for
        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];
            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(key));

            std::vector<std::vector<cv::Vec2f>> xy_seg_;
            if (key == "segmentation")
                find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi, 4/_ds_scale, 100);
            else
                find_intersect_segments(intersections[n], xy_seg_, segmentation->rawPoints(), plane, plane_roi, 4/_ds_scale);

        }

        std::hash<std::string> str_hasher;

        for(int n=0;n<intersect_cands.size();n++) {
            std::string key = intersect_cands[n];

            if (!intersections.size()) {
                _intersect_items[key] = {};
                continue;
            }

            size_t seed = str_hasher(key);
            srand(seed);

            int prim = rand() % 3;
            cv::Vec3i cvcol = {100 + rand() % 255, 100 + rand() % 255, 100 + rand() % 255};
            cvcol[prim] = 200 + rand() % 55;

            QColor col(cvcol[0],cvcol[1],cvcol[2]);
            float width = 2/_scene_scale;
            int z_value = 5;

            if (key == "segmentation") {
                col = Qt::yellow;
                width = 4/_scene_scale;
                z_value = 20;
            }


            QuadSurface *segmentation = dynamic_cast<QuadSurface*>(_surf_col->surface(intersect_cands[n]));
            std::vector<QGraphicsItem*> items;

            int len = 0;
            for (auto seg : intersections[n]) {
                QPainterPath path;

                bool first = true;
                for (auto wp : seg)
                {
                    len++;
                    cv::Vec3f p = plane->project(wp, 1.0, _ds_scale);
                    if (first)
                        path.moveTo(p[0],p[1]);
                    else
                        path.lineTo(p[0],p[1]);
                    first = false;
                }
                auto item = fGraphicsView->scene()->addPath(path, QPen(col, width));
                item->setZValue(z_value);
                items.push_back(item);
            }
            _intersect_items[key] = items;
            _ignore_intersect_change = new Intersection({intersections[n]});
            _surf_col->setIntersection(_surf_name, key, _ignore_intersect_change);
            _ignore_intersect_change = nullptr;
        }
    }
    else if (_surf_name == "segmentation" && dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"))) {
        QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"));

        //TODO make configurable, for now just show everything!
        std::vector<std::pair<std::string,std::string>> intersects = _surf_col->intersections("segmentation");
        for(auto pair : intersects) {
            std::string key = pair.first;
            if (key == "segmentation")
                key = pair.second;
            
            if (_intersect_items.count(key) || !_intersect_tgts.count(key))
                continue;
            
            std::unordered_map<cv::Vec3f,cv::Vec3f,vec3f_hash> location_cache;
            std::vector<cv::Vec3f> src_locations;
            SurfacePointer *ptrs[omp_get_max_threads()] = {};
            
            for (auto seg : _surf_col->intersection(pair.first, pair.second)->lines)
                for (auto wp : seg)
                    src_locations.push_back(wp);
            
#pragma omp parallel
            {
                SurfacePointer *ptr = crop->pointer();
#pragma omp for
                for (auto wp : src_locations) {
                    float res = crop->pointTo(ptr, wp, 1.0, 100);
                    cv::Vec3f p = crop->loc(ptr)*_ds_scale + cv::Vec3f(_vis_center[0],_vis_center[1],0);
                    //FIXME still happening?
                    // if (res >= 1.0)
                        // std::cout << "WARNING pointTo() high residual in renderIntersections()" << std::endl;
#pragma omp critical
                    location_cache[wp] = p;
                }
            }
            
            std::vector<QGraphicsItem*> items;
            for (auto seg : _surf_col->intersection(pair.first, pair.second)->lines) {
                QPainterPath path;
                
                // SurfacePointer *ptr = crop->pointer();
                bool first = true;
                for (auto wp : seg)
                {
                    // float res = crop->pointTo(ptr, wp, 1.0);
                    // cv::Vec3f p = crop->loc(ptr)*_ds_scale + cv::Vec3f(_vis_center[0],_vis_center[1],0);
                    cv::Vec3f p = location_cache[wp];
                    if (first)
                        path.moveTo(p[0],p[1]);
                    else
                        path.lineTo(p[0],p[1]);
                    first = false;
                }
                auto item = fGraphicsView->scene()->addPath(path, QPen(Qt::yellow, 2/_scene_scale));
                item->setZValue(5);
                items.push_back(item);
            }
            _intersect_items[key] = items;
        }
    }
}


void CVolumeViewer::onPanRelease(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    renderVisible();
}

void CVolumeViewer::onScrolled()
{
    // if (!dynamic_cast<OpChain*>(_surf) && !dynamic_cast<OpChain*>(_surf)->slow() && _min_scale == 1.0)
        // renderVisible();
    // if ((!dynamic_cast<OpChain*>(_surf) || !dynamic_cast<OpChain*>(_surf)->slow()) && _min_scale < 1.0)
        // renderVisible();
}
