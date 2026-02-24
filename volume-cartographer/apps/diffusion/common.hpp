#pragma once

#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

#include <filesystem>

namespace po = boost::program_options;
namespace fs = std::filesystem;

struct discrete_options;
struct continous_options;

void setup_cli_and_run(int argc, char** argv);