#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/io/PointSetIO.hpp"

#include <unordered_map>
#include <filesystem>
#include <omp.h>

#include "vc/core/types/ChunkedTensor.hpp"

using shape = z5::types::ShapeType;
using namespace xt::placeholders;
namespace fs = std::filesystem;

using json = nlohmann::json;

std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

class MeasureLife
{
public:
    MeasureLife(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~MeasureLife()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

std::string time_str()
{
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%Y%m%d%H%M%S"); // HH:MM:SS
    oss << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}

template <typename T, typename I>
float get_val(I &interp, cv::Vec3d l) {
    T v;
    interp.Evaluate(l[2], l[1], l[0], &v);
    return v;
}

int main(int argc, char *argv[])
{
    if (argc != 7 && argc != 4) {
        std::cout << "usage: " << argv[0] << " <zarr-volume> <tgt-dir> <json-params> <seed-x> <seed-y> <seed-z>" << std::endl;
        std::cout << "or:    " << argv[0] << " <zarr-volume> <tgt-dir> <json-params>" << std::endl;
        return EXIT_SUCCESS;
    }

    fs::path vol_path = argv[1];
    fs::path tgt_dir = argv[2];
    const char *params_path = argv[3];

    std::ifstream params_f(params_path);
    json params = json::parse(params_f);

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "0", json::parse(std::ifstream(vol_path/"0/.zarray")).value<>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    ChunkCache chunk_cache(1e9);


    passTroughComputor pass;
    Chunked3d<uint8_t,passTroughComputor> tensor(pass, ds.get(), &chunk_cache);
    CachedChunked3dInterpolator<uint8_t,passTroughComputor> interpolator(tensor);

    auto chunk_size = ds->chunking().blockShape();

    srand(clock());

    std::string mode;

    cv::Vec3d origin;

    std::string name_prefix = "auto_grown_";
    // std::string name_prefix = "testing_autogrow";
    int tgt_overlap_count = 10;
    float min_area_cm = 0.3;
    float step_size = 20;

    bool expansion_mode = true;

    std::unordered_map<std::string,SurfaceMeta*> partial;
    std::unordered_map<std::string,SurfaceMeta*> full;
    SurfaceMeta *src;

    //expansion mode
    int count_overlap;
    if (expansion_mode) {
        mode = "grow_random_choice";
        //got trough all exising segments (that match filter/start with auto ...)
        //list which ones do not yet less N overlapping (in symlink dir)
        //shuffle
        //iterate and for every one
            //select a random point (close to edge?)
            //check against list if other surf in bbox if we can find the point
            //if yes add symlinkg between the two segs
            //if both still have less than N then grow a seg from the seed
            //after growing, check locations on the new seg agains all existing segs

        for (const auto& entry : fs::directory_iterator(tgt_dir))
            if (fs::is_directory(entry)) {
                std::string name = entry.path().filename();
                if (name.compare(0, name_prefix.size(), name_prefix))
                    continue;

                std::cout << entry.path() << entry.path().filename() << std::endl;

                fs::path meta_fn = entry.path() / "meta.json";
                if (!fs::exists(meta_fn))
                    continue;

                std::ifstream meta_f(meta_fn);
                json meta = json::parse(meta_f);

                if (!meta.count("bbox"))
                    continue;

                if (meta.value("format","NONE") != "tifxyz")
                    continue;

                SurfaceMeta *sm = new SurfaceMeta(entry.path(), meta);
                sm->readOverlapping();

                // std::cout << "overlaps: " << sm->overlapping.size() << std::endl;

                if (sm->overlapping.size() < tgt_overlap_count)
                    partial[name] = sm;
                else
                    full[name] = sm;
            }

        std::vector<SurfaceMeta*> partial_shuffled;
        for(auto &it : partial)
            partial_shuffled.push_back(it.second);

        std::vector<SurfaceMeta*> all_shuffled;
        for(auto &it : partial)
            all_shuffled.push_back(it.second);
        for(auto &it : full)
            all_shuffled.push_back(it.second);

        std::default_random_engine rng(clock());
        std::shuffle(std::begin(all_shuffled), std::end(all_shuffled), rng);

        if (!all_shuffled.size())
            return EXIT_SUCCESS;

        for(auto &it : all_shuffled) {
            src = it;
            cv::Mat_<cv::Vec3f> points = src->surf()->rawPoints();
            int w = points.cols;
            int h = points.rows;

            // cv::Mat_<uint8_t> searchvis(points.size(), 0);

            bool found = false;
            // int fcount = 0;
            for (int r=0;r<10;r++) {
                cv::Vec2f p;
                int side = rand() % 4;
                if (side == 0)
                    p = {rand() % h, 0};
                else if (side == 1)
                    p = {0, rand() % w};
                else if (side == 2)
                    p = {rand() % h, w-1};
                else if (side == 3)
                    p = {h-1, rand() % w};

                cv::Vec2f searchdir = cv::Vec2f(h/2,w/2) - p;
                cv::normalize(searchdir, searchdir);
                found = false;
                for(int i=0;i<std::min(w/2/abs(searchdir[1]),h/2/abs(searchdir[0]));i++,p+=searchdir) {
                    found = true;
                    cv::Vec2i p_eval = p;
                    // searchvis(p_eval) = 127;
                    for(int r=0;r<5;r++) {
                        cv::Vec2i p_eval = p+r*searchdir;
                        if (points(p_eval)[0] == -1 ||get_val<double,CachedChunked3dInterpolator<uint8_t,passTroughComputor>>(interpolator, points(p_eval)) < 128) {
                            found = false;
                            break;
                        }
                        // else
                            // searchvis(p_eval) = 255;;
                    }
                    if (found) {
                        // fcount++;
                        cv::Vec2i p_eval = p+2*searchdir;
                        origin = points(p_eval);
                        break;
                    }
                }
            }

            if (!found)
                continue;

            count_overlap = 0;
            for(auto comp : all_shuffled) {
                if (comp == src)
                    continue;
                if (contains(*comp, origin, 10))
                    count_overlap++;
                if (count_overlap >= 1)
                    break;
            }
            if (count_overlap < 1)
                break;
        }

        // cv::imwrite("searchvis.tif", searchvis);

        std::cout << "found potential overlapping starting seed" << origin << "with overlap " << count_overlap << std::endl;
    }
    else {
        if (argc == 7) {
            mode = "explicit_seed";
            origin = {atof(argv[4]),atof(argv[5]),atof(argv[6])};
            double v;
            interpolator.Evaluate(origin[2], origin[1], origin[0], &v);
            std::cout << "seed location value is " << v << std::endl;
        }
        else
        {
            mode = "random_seed";
            int count = 0;
            bool succ = false;
            while(!succ) {
                origin = {128 + (rand() % (ds->shape(0)-384)), 128 + (rand() % (ds->shape(1)-384)), 128 + (rand() % (ds->shape(2)-384))};
                // origin[2] = 6500 + (rand() % 2000) - 1000;

                count++;
                auto chunk_id = chunk_size;
                chunk_id[0] = origin[2]/chunk_id[0];
                chunk_id[1] = origin[1]/chunk_id[1];
                chunk_id[2] = origin[0]/chunk_id[2];

                if (!ds->chunkExists(chunk_id))
                    continue;

                cv::Vec3d dir = {(rand() % 1024) - 512,(rand() % 1024) - 512,(rand() % 1024) - 512};
                cv::normalize(dir, dir);

                for(int i=0;i<128;i++) {
                    double v;
                    cv::Vec3d p = origin + i*dir;
                    interpolator.Evaluate(p[2], p[1], p[0], &v);
                    if (v >= 128) {
                        succ = true;
                        origin = p;
                        std::cout << "try " << count << " seed " << origin << " value is " << v << std::endl;
                        break;
                    }
                }
            }
        }
    }

    omp_set_num_threads(1);
    
    // origin = {3036.043212890625, 
    //     4807.43798828125,
    //     11265.13671875};

    QuadSurface *surf = empty_space_tracing_quad_phys(ds.get(), 1.0, &chunk_cache, origin, step_size);

    if ((*surf->meta)["area_cm"] < 0.3)
        return EXIT_SUCCESS;

    (*surf->meta)["source"] = "vc_grow_seg_from_seed";
    // (*surf->meta)["vc_gsfs_params"] = params;
    (*surf->meta)["vc_gsfs_mode"] = mode;
    (*surf->meta)["vc_gsfs_version"] = "dev";
    (*surf->meta)["seed_overlap"] = count_overlap;
    std::string uuid = name_prefix + time_str();
    fs::path seg_dir = tgt_dir / uuid;
    surf->save(seg_dir, uuid);

    SurfaceMeta current;

    if (expansion_mode) {
        current.path = seg_dir;
        current.setSurf(surf);
        current.bbox = surf->bbox();

        fs::path overlap_dir = current.path / "overlapping";
        fs::create_directory(overlap_dir);

        {std::ofstream touch(overlap_dir/src->name());}

        fs::path overlap_src = src->path / "overlapping";
        fs::create_directory(overlap_src);
        {std::ofstream touch(overlap_src/current.name());}

        for(auto &pair : full)
            if (overlap(current, *pair.second, 10)) {
                std::ofstream touch_me(overlap_dir/pair.second->name());
                fs::path overlap_other = pair.second->path / "overlapping";
                fs::create_directory(overlap_other);
                std::ofstream touch_you(overlap_other/current.name());
            }

        for(auto &pair : partial)
            if (overlap(current, *pair.second, 10)) {
                std::ofstream touch_me(overlap_dir/pair.second->name());
                fs::path overlap_other = pair.second->path / "overlapping";
                fs::create_directory(overlap_other);
                std::ofstream touch_you(overlap_other/current.name());
            }

        for (const auto& entry : fs::directory_iterator(tgt_dir))
            if (fs::is_directory(entry) && !full.count(entry.path().filename()) && !partial.count(entry.path().filename()))
            {
                std::string name = entry.path().filename();
                if (name.compare(0, name_prefix.size(), name_prefix))
                    continue;

                if (name == current.name())
                    continue;

                fs::path meta_fn = entry.path() / "meta.json";
                if (!fs::exists(meta_fn))
                    continue;

                std::ifstream meta_f(meta_fn);
                json meta = json::parse(meta_f);

                if (!meta.count("bbox"))
                    continue;

                if (meta.value("format","NONE") != "tifxyz")
                    continue;

                SurfaceMeta other = SurfaceMeta(entry.path(), meta);
                other.readOverlapping();

                if (overlap(current, other, 10)) {
                    std::ofstream touch_me(overlap_dir/other.name());
                    fs::path overlap_other = other.path / "overlapping";
                    fs::create_directory(overlap_other);
                    std::ofstream touch_you(overlap_other/current.name());
                }
            }




    }
}
