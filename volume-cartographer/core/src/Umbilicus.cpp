#include "vc/core/util/Umbilicus.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <numbers>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "utils/Json.hpp"

namespace {

std::string TrimCopy(const std::string& value)
{
    auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) { return std::isspace(ch) != 0; });
    auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

} // namespace

namespace vc::core::util {

Umbilicus Umbilicus::FromFile(const std::filesystem::path& path, const cv::Vec3i& volume_shape)
{
    return Umbilicus(LoadFile(path), volume_shape);
}

const cv::Vec3f& Umbilicus::center_at(int z_index) const
{
    if (z_index < 0 || z_index >= static_cast<int>(dense_centers_.size())) {
        throw std::out_of_range("z_index outside interpolated range");
    }
    return dense_centers_[z_index];
}

cv::Vec3f Umbilicus::vector_to_umbilicus(const cv::Vec3f& point) const
{
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }
    const int index = clamp_z_index(point[2]);
    return dense_centers_[index] - point;
}

Umbilicus::Umbilicus(std::vector<cv::Vec3f> control_points, const cv::Vec3i& volume_shape)
    : volume_shape_(volume_shape), control_points_(std::move(control_points))
{
    if (volume_shape_[0] <= 0 || volume_shape_[1] <= 0 || volume_shape_[2] <= 0) {
        throw std::invalid_argument("Volume shape components must be positive");
    }
    if (control_points_.empty()) {
        throw std::invalid_argument("Umbilicus requires at least one control point");
    }

    std::sort(control_points_.begin(), control_points_.end(), [](const auto& a, const auto& b) {
        return a[2] < b[2];
    });

    interpolate_centers();
}

std::vector<cv::Vec3f> Umbilicus::LoadFile(const std::filesystem::path& path)
{
    const std::string extension = path.extension().string();
    std::string lowered_ext;
    lowered_ext.resize(extension.size());
    std::transform(extension.begin(), extension.end(), lowered_ext.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (lowered_ext == ".json") {
        return LoadJsonFile(path);
    }

    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open umbilicus text file: " + path.string());
    }
    return LoadTextFile(stream);
}

std::vector<cv::Vec3f> Umbilicus::LoadTextFile(std::istream& stream)
{
    std::vector<cv::Vec3f> points;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(stream, line)) {
        ++line_number;
        auto trimmed = TrimCopy(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        std::array<double, 3> values{};
        std::size_t value_index = 0;

        std::stringstream ss(trimmed);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = TrimCopy(token);
            if (token.empty()) {
                continue;
            }
            if (value_index >= values.size()) {
                throw std::runtime_error("Too many columns in umbilicus text line " + std::to_string(line_number));
            }
            try {
                values[value_index] = std::stod(token);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid numeric value in umbilicus text line " + std::to_string(line_number));
            }
            ++value_index;
        }

        if (value_index != values.size()) {
            throw std::runtime_error("Not enough columns in umbilicus text line " + std::to_string(line_number));
        }

        cv::Vec3f point{
            static_cast<float>(values[2]),
            static_cast<float>(values[1]),
            static_cast<float>(values[0])
        };
        points.push_back(point);
    }

    if (points.empty()) {
        throw std::runtime_error("Umbilicus text file contained no points");
    }

    return points;
}

std::vector<cv::Vec3f> Umbilicus::LoadJsonFile(const std::filesystem::path& path)
{
    utils::Json document = utils::Json::parse_file(path);

    const utils::Json* array = nullptr;
    if (document.is_array()) {
        array = &document;
    } else if (document.contains("points")) {
        const auto& candidate = document.at("points");
        if (!candidate.is_array()) {
            throw std::runtime_error("'points' member in umbilicus json must be an array");
        }
        array = &candidate;
    } else {
        throw std::runtime_error("Umbilicus json root must be an array or contain a 'points' array");
    }

    std::vector<cv::Vec3f> points;
    points.reserve(array->size());

    for (std::size_t idx = 0; idx < array->size(); ++idx) {
        const auto& entry = (*array)[idx];
        double z = 0.0;
        double y = 0.0;
        double x = 0.0;

        if (entry.is_array()) {
            if (entry.size() < 3) {
                throw std::runtime_error("Umbilicus json entry at index " + std::to_string(idx) + " expected three values");
            }
            z = entry[0].get_double();
            y = entry[1].get_double();
            x = entry[2].get_double();
        } else if (entry.is_object()) {
            if (!entry.contains("z") || !entry.contains("y") || !entry.contains("x")) {
                throw std::runtime_error("Umbilicus json entry at index " + std::to_string(idx) + " missing z/y/x keys");
            }
            z = entry.at("z").get_double();
            y = entry.at("y").get_double();
            x = entry.at("x").get_double();
        } else {
            throw std::runtime_error("Umbilicus json entry at index " + std::to_string(idx) + " has unsupported type");
        }

        points.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }

    if (points.empty()) {
        throw std::runtime_error("Umbilicus json file contained no points");
    }

    return points;
}

void Umbilicus::interpolate_centers()
{
    dense_centers_.clear();
    dense_centers_.resize(volume_shape_[0]);
    for (int z = 0; z < volume_shape_[0]; ++z) {
        dense_centers_[z] = interpolate_center(static_cast<double>(z));
    }
}

cv::Vec3f Umbilicus::interpolate_center(double z) const
{
    if (control_points_.empty()) {
        throw std::logic_error("Umbilicus interpolation requested without control points");
    }

    if (control_points_.size() == 1) {
        cv::Vec3f result = control_points_.front();
        result[2] = static_cast<float>(z);
        return result;
    }

    const double min_z = control_points_.front()[2];
    const double max_z = control_points_.back()[2];

    if (z <= min_z) {
        cv::Vec3f result = control_points_.front();
        result[2] = static_cast<float>(z);
        return result;
    }
    if (z >= max_z) {
        cv::Vec3f result = control_points_.back();
        result[2] = static_cast<float>(z);
        return result;
    }

    const float target = static_cast<float>(z);
    auto upper = std::lower_bound(control_points_.begin(), control_points_.end(), target,
                                  [](const cv::Vec3f& lhs, float value) { return lhs[2] < value; });
    if (upper == control_points_.begin()) {
        cv::Vec3f result = *upper;
        result[2] = static_cast<float>(z);
        return result;
    }

    if (upper != control_points_.end() && std::abs((*upper)[2] - target) < 1e-5f) {
        cv::Vec3f result = *upper;
        result[2] = static_cast<float>(z);
        return result;
    }

    const auto& right = (upper == control_points_.end()) ? control_points_.back() : *upper;
    const auto& left = *(upper - 1);

    const double z0 = left[2];
    const double z1 = right[2];

    if (std::abs(z1 - z0) < 1e-5) {
        cv::Vec3f result = left;
        result[2] = static_cast<float>(z);
        return result;
    }

    const double t = (z - z0) / (z1 - z0);
    const double x = left[0] + t * (right[0] - left[0]);
    const double y = left[1] + t * (right[1] - left[1]);
    return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}

int Umbilicus::clamp_z_index(double z) const
{
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }
    int index = static_cast<int>(std::lround(z));
    index = std::clamp(index, 0, static_cast<int>(dense_centers_.size() - 1));
    return index;
}

} // namespace vc::core::util
