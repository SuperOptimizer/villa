
#include "spiral_common.hpp"
#include "spiral_ceres.hpp"

#include "utils/Json.hpp"

void to_json(utils::Json& j, const SpiralPoint& p) {
    j = utils::Json::object();
    j["pos"] = utils::Json::array();
    j["pos"].push_back(p.pos[0]);
    j["pos"].push_back(p.pos[1]);
    j["pos"].push_back(p.pos[2]);
    j["winding"] = p.winding;
}

void from_json(const utils::Json& j, SpiralPoint& p) {
    std::vector<double> pos_vec = j.at("pos").get_double_array();
    if (pos_vec.size() == 3) {
        p.pos = cv::Vec3d(pos_vec[0], pos_vec[1], pos_vec[2]);
    }
    p.winding = j.at("winding").get_double();
}

void visualize_spiral(
    const std::vector<SpiralPoint>& all_points,
    const cv::Size& slice_size,
    const fs::path& output_path,
    const cv::Scalar& point_color,
    const std::vector<SheetConstraintRay>& constraint_rays,
    bool draw_influence,
    bool draw_winding_text
) {
    cv::Mat viz = cv::Mat::zeros(slice_size, CV_8UC3);
    visualize_spiral(viz, all_points, cv::Scalar(255, 255, 255), point_color, draw_winding_text);

    if (cv::imwrite(output_path.string(), viz)) {
        std::cout << "Saved spiral visualization to " << output_path << std::endl;
    } else {
        std::cerr << "Error: Failed to write spiral visualization to " << output_path << std::endl;
    }
}


#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>




void visualize_spiral(
    cv::Mat& viz,
    const std::vector<SpiralPoint>& all_points,
    const cv::Scalar& line_color,
    const cv::Scalar& point_color,
    bool draw_winding_text
) {
    if (all_points.empty()) return;

    // Draw spiral edges
    for (size_t i = 0; i < all_points.size() - 1; ++i) {
        // A simple check to handle forward and backward spirals without complex sorting
        if (std::abs(all_points[i+1].winding - all_points[i].winding) < 0.5) {
            cv::Point p1(all_points[i].pos[0], all_points[i].pos[1]);
            cv::Point p2(all_points[i+1].pos[0], all_points[i+1].pos[1]);
            cv::line(viz, p1, p2, line_color, 1, cv::LINE_AA);
        }
    }

    // Draw the points
    for (size_t i = 0; i < all_points.size(); ++i) {
        cv::Point p(all_points[i].pos[0], all_points[i].pos[1]);
        cv::circle(viz, p, 3, point_color, -1, cv::LINE_AA);
        if (draw_winding_text) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << all_points[i].winding;
            cv::putText(viz, ss.str(), p + cv::Point(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
    }
}
