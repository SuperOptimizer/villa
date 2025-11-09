#include "vc/core/util/Surface.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <algorithm>
#include <vector>


// Use libtiff for BigTIFF; fall back to OpenCV if not present.
#include <tiffio.h>


//TODO check if this actually works?!
static void set_block(cv::Mat_<uint8_t> &block, const cv::Vec3f &last_loc, const cv::Vec3f &loc, const cv::Rect &roi, float step)
{
    int x1 = (loc[0]-roi.x)/step;
    int y1 = (loc[1]-roi.y)/step;
    int x2 = (last_loc[0]-roi.x)/step;
    int y2 = (last_loc[1]-roi.y)/step;

    if (x1 < 0 || y1 < 0 || x1 >= block.cols || y1 >= block.rows)
        return;
    if (x2 < 0 || y2 < 0 || x2 >= block.cols || y2 >= block.rows)
        return;

    if (x1 == x2 && y1 == y2)
        block(y1, x1) = 1;
    else
        cv::line(block, {x1,y1},{x2,y2}, 3);
}

static uint8_t get_block(const cv::Mat_<uint8_t> &block, const cv::Vec3f &loc, const cv::Rect &roi, float step)
{
    int x = (loc[0]-roi.x)/step;
    int y = (loc[1]-roi.y)/step;

    if (x < 0 || y < 0 || x >= block.cols || y >= block.rows)
        return 1;

    return block(y, x);
}


void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step, int min_tries)
{
    //start with random points and search for a plane intersection

    float block_step = 0.5*step;

    cv::Mat_<uint8_t> block(cv::Size(plane_roi.width/block_step, plane_roi.height/block_step), 0);

    cv::Rect grid_bounds(1,1,points.cols-2,points.rows-2);

    std::vector<std::vector<cv::Vec3f>> seg_vol_raw;
    std::vector<std::vector<cv::Vec2f>> seg_grid_raw;

    for(int r=0;r<std::max(min_tries, std::max(points.cols,points.rows)/100);r++) {
        std::vector<cv::Vec3f> seg;
        std::vector<cv::Vec2f> seg_loc;
        std::vector<cv::Vec3f> seg2;
        std::vector<cv::Vec2f> seg_loc2;
        cv::Vec2f loc;
        cv::Vec2f loc2;
        cv::Vec2f loc3;
        cv::Vec3f point;
        cv::Vec3f point2;
        cv::Vec3f point3;
        cv::Vec3f plane_loc;
        cv::Vec3f last_plane_loc;
        float dist = -1;


        //initial points
        for(int i=0;i<std::max(min_tries, std::max(points.cols,points.rows)/100);i++) {
            loc = {std::rand() % (points.cols-1), std::rand() % (points.rows-1)};
            point = at_int(points, loc);

            plane_loc = plane->project(point);
            if (!plane_roi.contains(cv::Point(plane_loc[0],plane_loc[1])))
                continue;

                dist = min_loc(points, loc, point, {}, {}, plane, std::min(points.cols,points.rows)*0.1, 0.01);

                plane_loc = plane->project(point);
                if (!plane_roi.contains(cv::Point(plane_loc[0],plane_loc[1])))
                    dist = -1;

                if (get_block(block, plane_loc, plane_roi, block_step))
                    dist = -1;

            if (dist >= 0 && dist <= 1 || !loc_valid_xy(points, loc))
                break;
        }


        if (dist < 0 || dist > 1)
            continue;

        seg.push_back(point);
        seg_loc.push_back(loc);

        //point2
        loc2 = loc;
        //search point at distance of 1 to init point
        dist = min_loc(points, loc2, point2, {point}, {1}, plane, 0.01, 0.0001);

        if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc))
            continue;

        seg.push_back(point2);
        seg_loc.push_back(loc2);

        last_plane_loc = plane->project(point);
        plane_loc = plane->project(point2);
        set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
        last_plane_loc = plane_loc;

        //go one direction
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains(cv::Point(loc3)))
                break;

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.01, 0.0001);

                //then refine
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.01, 0.0001);

                if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc))
                    break;

            seg.push_back(point3);
            seg_loc.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            plane_loc = plane->project(point3);
            if (get_block(block, plane_loc, plane_roi, block_step))
                break;

            set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
            last_plane_loc = plane_loc;
        }

        //now the other direction
        loc2 = seg_loc[0];
        loc = seg_loc[1];
        point2 = seg[0];
        point = seg[1];

        last_plane_loc = plane->project(point2);

        //FIXME repeat by not copying code ...
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains(cv::Point(loc3[0])))
                break;

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.01, 0.0001);

                //then refine
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.01, 0.0001);

                if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc))
                    break;

            seg2.push_back(point3);
            seg_loc2.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            plane_loc = plane->project(point3);
            if (get_block(block, plane_loc, plane_roi, block_step))
                break;

            set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
            last_plane_loc = plane_loc;
        }

        std::reverse(seg2.begin(), seg2.end());
        std::reverse(seg_loc2.begin(), seg_loc2.end());

        seg2.insert(seg2.end(), seg.begin(), seg.end());
        seg_loc2.insert(seg_loc2.end(), seg_loc.begin(), seg_loc.end());


        seg_vol_raw.push_back(seg2);
        seg_grid_raw.push_back(seg_loc2);
    }

    //split up into disconnected segments
    for(int s=0;s<seg_vol_raw.size();s++) {
        std::vector<cv::Vec3f> seg_vol_curr;
        std::vector<cv::Vec2f> seg_grid_curr;
        cv::Vec3f last = {-1,-1,-1};
        for(int n=0;n<seg_vol_raw[s].size();n++) {
                if (last[0] != -1 && cv::norm(last-seg_vol_raw[s][n]) >= 2*step) {
                seg_vol.push_back(seg_vol_curr);
                seg_grid.push_back(seg_grid_curr);
                seg_vol_curr.resize(0);
                seg_grid_curr.resize(0);
            }
            last = seg_vol_raw[s][n];
            seg_vol_curr.push_back(seg_vol_raw[s][n]);
            seg_grid_curr.push_back(seg_grid_raw[s][n]);
        }
        if (seg_vol_curr.size() >= 2) {
            seg_vol.push_back(seg_vol_curr);
            seg_grid.push_back(seg_grid_curr);
        }
    }
}
