#include "vc/core/types/VolumePkg.hpp"

#include <set>
#include <utility>
#include <sys/inotify.h>
#include <unistd.h>
#include <poll.h>
#include <cerrno>
#include <cstring>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Logging.hpp"

constexpr auto CONFIG = "config.json";

VolumePkg::VolumePkg(const std::filesystem::path& fileLocation) : rootDir_{fileLocation}
{
    config_ = Metadata(fileLocation / ::CONFIG);

    std::vector<std::string> dirs = {"volumes","paths","traces","transforms","renders"};

    for (const auto& d : dirs) {
        if (not std::filesystem::exists(rootDir_ / d)) {
            std::filesystem::create_directory(rootDir_ / d);
        }

    }

    for (const auto& entry : std::filesystem::directory_iterator(rootDir_ / "volumes")) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            auto v = Volume::New(dirpath);
            volumes_.emplace(v->id(), v);
        }
    }

    auto availableDirs = getAvailableSegmentationDirectories();
    for (const auto& dirName : availableDirs) {
        loadSegmentationsFromDirectory(dirName);
    }
}

std::shared_ptr<VolumePkg> VolumePkg::New(const std::filesystem::path& fileLocation)
{
    return std::make_shared<VolumePkg>(fileLocation);
}


std::string VolumePkg::name() const
{
    auto name = config_.get<std::string>("name");
    if (name != "NULL") {
        return name;
    }

    return "UnnamedVolume";
}

int VolumePkg::version() const { return config_.get<int>("version"); }

bool VolumePkg::hasVolumes() const { return !volumes_.empty(); }

bool VolumePkg::hasVolume(const std::string& id) const
{
    return volumes_.count(id) > 0;
}

std::size_t VolumePkg::numberOfVolumes() const
{
    return volumes_.size();
}

std::vector<std::string> VolumePkg::volumeIDs() const
{
    std::vector<std::string> ids;
    for (const auto& v : volumes_) {
        ids.emplace_back(v.first);
    }
    return ids;
}

std::shared_ptr<Volume> VolumePkg::volume()
{
    if (volumes_.empty()) {
        throw std::out_of_range("No volumes in VolPkg");
    }
    return volumes_.begin()->second;
}

std::shared_ptr<Volume> VolumePkg::volume(const std::string& id)
{
    return volumes_.at(id);
}

bool VolumePkg::hasSegmentations() const
{
    return !segmentations_.empty();
}


std::shared_ptr<Segmentation> VolumePkg::segmentation(const std::string& id)
{
    return segmentations_.at(id);
}

std::vector<std::string> VolumePkg::segmentationIDs() const
{
    std::vector<std::string> ids;
    // Only return IDs from the current directory
    for (const auto& s : segmentations_) {
        auto it = segmentationDirectories_.find(s.first);
        if (it != segmentationDirectories_.end() && it->second == currentSegmentationDir_) {
            ids.emplace_back(s.first);
        }
    }
    return ids;
}


void VolumePkg::loadSegmentationsFromDirectory(const std::string& dirName)
{
    // DO NOT clear existing segmentations - we keep all directories in memory
    // Only remove segmentations from this specific directory
    std::vector<std::string> toRemove;
    for (const auto& pair : segmentationDirectories_) {
        if (pair.second == dirName) {
            toRemove.push_back(pair.first);
        }
    }
    
    // Remove old segmentations from this directory
    for (const auto& id : toRemove) {
        segmentations_.erase(id);
        segmentationDirectories_.erase(id);
    }
    
    // Check if directory exists
    const auto segDir = rootDir_ / dirName;
    if (!std::filesystem::exists(segDir)) {
        Logger()->warn("Segmentation directory '{}' does not exist", dirName);
        return;
    }
    
    // Load segmentations from the specified directory
    for (const auto& entry : std::filesystem::directory_iterator(segDir)) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            try {
                auto s = Segmentation::New(dirpath);
                segmentations_.emplace(s->id(), s);
                // Track which directory this segmentation came from
                segmentationDirectories_[s->id()] = dirName;
            }
            catch (const std::exception &exc) {
                std::cout << "WARNING: some exception occured, skipping segment dir: " << dirpath << std::endl;
                std::cerr << exc.what();
            }
        }
    }
}

void VolumePkg::setSegmentationDirectory(const std::string& dirName)
{
    if (currentSegmentationDir_ == dirName) {
        return;
    }

    bool wasWatching = watcherRunning_;
    if (wasWatching) {
        stopWatcher();
    }

    currentSegmentationDir_ = dirName;

    if (wasWatching) {
        startWatcher();
    }
}

auto VolumePkg::getSegmentationDirectory() const -> std::string
{
    return currentSegmentationDir_;
}

auto VolumePkg::getVolpkgDirectory() const -> std::string
{
    return rootDir_;
}


auto VolumePkg::getAvailableSegmentationDirectories() const -> std::vector<std::string>
{
    std::vector<std::string> dirs;
    
    // Check for common segmentation directories
    const std::vector<std::string> commonDirs = {"paths", "traces"};
    for (const auto& dir : commonDirs) {
        if (std::filesystem::exists(rootDir_ / dir) && std::filesystem::is_directory(rootDir_ / dir)) {
            dirs.push_back(dir);
        }
    }
    
    return dirs;
}

void VolumePkg::removeSegmentation(const std::string& id)
{
    // Check if segmentation exists
    auto it = segmentations_.find(id);
    if (it == segmentations_.end()) {
        throw std::runtime_error("Segmentation not found: " + id);
    }
    
    // Get the path before removing
    std::filesystem::path segPath = it->second->path();
    
    // Remove from internal map
    segmentations_.erase(it);
    
    // Delete the physical folder
    if (std::filesystem::exists(segPath)) {
        std::filesystem::remove_all(segPath);
    }
}

void VolumePkg::refreshSegmentations()
{
    const auto segDir = rootDir_ / currentSegmentationDir_;
    if (!std::filesystem::exists(segDir)) {
        Logger()->warn("Segmentation directory '{}' does not exist", currentSegmentationDir_);
        return;
    }
    
    // Build a set of current segmentation paths on disk for the current directory
    std::set<std::filesystem::path> diskPaths;
    for (const auto& entry : std::filesystem::directory_iterator(segDir)) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            diskPaths.insert(dirpath);
        }
    }
    
    // Find segmentations to remove (loaded from current directory but not on disk anymore)
    std::vector<std::string> toRemove;
    for (const auto& seg : segmentations_) {
        auto dirIt = segmentationDirectories_.find(seg.first);
        if (dirIt != segmentationDirectories_.end() && dirIt->second == currentSegmentationDir_) {
            // This segmentation belongs to the current directory
            // Check if it still exists on disk
            if (diskPaths.find(seg.second->path()) == diskPaths.end()) {
                // Not on disk anymore - mark for removal
                toRemove.push_back(seg.first);
            }
        }
    }
    
    // Remove segmentations that no longer exist
    for (const auto& id : toRemove) {
        Logger()->info("Removing segmentation '{}' - no longer exists on disk", id);
        
        // Get the path before removing the segmentation
        std::filesystem::path segPath;
        auto segIt = segmentations_.find(id);
        if (segIt != segmentations_.end()) {
            segPath = segIt->second->path();
        }
        
        // Remove from segmentations map
        segmentations_.erase(id);
        
        // Remove from directories map
        segmentationDirectories_.erase(id);
    }
    
    // Find and add new segmentations (on disk but not in memory)
    for (const auto& diskPath : diskPaths) {
        bool found = false;
        for (const auto& seg : segmentations_) {
            if (seg.second->path() == diskPath) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            try {
                auto s = Segmentation::New(diskPath);
                segmentations_.emplace(s->id(), s);
                segmentationDirectories_[s->id()] = currentSegmentationDir_;
                Logger()->info("Added new segmentation '{}'", s->id());
            }
            catch (const std::exception &exc) {
                Logger()->warn("Failed to load segment dir: {} - {}", diskPath.string(), exc.what());
            }
        }
    }
}

bool VolumePkg::isSurfaceLoaded(const std::string& id) const
{
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        return false;
    }
    return segIt->second->isSurfaceLoaded();
}

std::shared_ptr<SurfaceMeta> VolumePkg::loadSurface(const std::string& id)
{
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        throw std::runtime_error("Segmentation not found: " + id);
    }
    return segIt->second->loadSurface();
}

std::shared_ptr<SurfaceMeta> VolumePkg::getSurface(const std::string& id)
{
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        return nullptr;
    }
    return segIt->second->getSurface();
}


std::vector<std::string> VolumePkg::getLoadedSurfaceIDs() const
{
    std::vector<std::string> ids;
    for (const auto& [id, seg] : segmentations_) {
        if (seg->isSurfaceLoaded()) {
            ids.push_back(id);
        }
    }
    return ids;
}

void VolumePkg::unloadAllSurfaces()
{
    for (auto& [id, seg] : segmentations_) {
        seg->unloadSurface();
    }
}

bool VolumePkg::unloadSurface(const std::string& id)
{
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        return false;
    }
    segIt->second->unloadSurface();
    return true;
}


void VolumePkg::loadSurfacesBatch(const std::vector<std::string>& ids)
{
    std::vector<std::shared_ptr<Segmentation>> toLoad;
    for (const auto& id : ids) {
        auto segIt = segmentations_.find(id);
        if (segIt != segmentations_.end() && !segIt->second->isSurfaceLoaded() && segIt->second->canLoadSurface()) {
            toLoad.push_back(segIt->second);
        }
    }

#pragma omp parallel for schedule(dynamic,1)
    for (auto & seg : toLoad) {
        try {
            seg->loadSurface();
        } catch (const std::exception& e) {
            Logger()->error("Failed to load surface for {}: {}", seg->id(), e.what());
        }
    }
}


// Add to destructor
VolumePkg::~VolumePkg()
{
    stopWatcher();
}

void VolumePkg::enableFileWatching(bool enable)
{
    if (enable && !watcherRunning_) {
        startWatcher();
    } else if (!enable && watcherRunning_) {
        stopWatcher();
    }
}

void VolumePkg::startWatcher()
{
    if (watcherRunning_) {
        return;
    }

    // Initialize inotify
    inotifyFd_ = inotify_init1(IN_NONBLOCK);
    if (inotifyFd_ < 0) {
        Logger()->warn("Failed to initialize inotify: {}. File watching disabled.",
                      std::strerror(errno));
        return;
    }

    // Setup watches for current segmentation directory
    std::filesystem::path watchPath = rootDir_ / currentSegmentationDir_;

    if (!std::filesystem::exists(watchPath)) {
        std::filesystem::create_directories(watchPath);
    }

    // Add watches
    addWatchesRecursive(watchPath);

    // Start watch thread
    shouldStopWatcher_ = false;
    watcherRunning_ = true;
    watchThread_ = std::thread(&VolumePkg::watchLoop, this);

    Logger()->info("File watching enabled for {}", watchPath.string());
}

void VolumePkg::stopWatcher()
{
    if (!watcherRunning_) {
        return;
    }

    shouldStopWatcher_ = true;

    if (watchThread_.joinable()) {
        watchThread_.join();
    }

    // Clean up watches
    {
        std::lock_guard<std::mutex> lock(watchMutex_);
        for (const auto& [wd, path] : watchDescriptors_) {
            inotify_rm_watch(inotifyFd_, wd);
        }
        watchDescriptors_.clear();
    }

    if (inotifyFd_ >= 0) {
        close(inotifyFd_);
        inotifyFd_ = -1;
    }

    watcherRunning_ = false;
    Logger()->info("File watching disabled");
}

void VolumePkg::addWatch(const std::filesystem::path& path)
{
    if (inotifyFd_ < 0) return;

    int wd = inotify_add_watch(inotifyFd_, path.c_str(),
                               IN_CREATE | IN_DELETE | IN_MODIFY |
                               IN_MOVED_FROM | IN_MOVED_TO | IN_CLOSE_WRITE);

    if (wd < 0) {
        Logger()->warn("Failed to add watch for {}: {}",
                      path.string(), std::strerror(errno));
        return;
    }

    std::lock_guard<std::mutex> lock(watchMutex_);
    watchDescriptors_[wd] = path;
}

void VolumePkg::addWatchesRecursive(const std::filesystem::path& path)
{
    // Add watch for this directory
    addWatch(path);

    // Add watches for all subdirectories (segmentation directories)
    try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_directory()) {
                addWatch(entry.path());
                // We only go 2 levels deep
            }
        }
    } catch (const std::exception& e) {
        Logger()->error("Error adding watches: {}", e.what());
    }
}

void VolumePkg::watchLoop()
{
    constexpr size_t BUFFER_SIZE = 8192;
    alignas(struct inotify_event) char buffer[BUFFER_SIZE];

    while (!shouldStopWatcher_) {
        // Poll with timeout
        struct pollfd pfd = {
            .fd = inotifyFd_,
            .events = POLLIN,
            .revents = 0
        };

        int ret = poll(&pfd, 1, 250); // 250ms timeout

        if (ret < 0) {
            if (errno != EINTR) {
                Logger()->error("Poll error: {}", std::strerror(errno));
            }
            continue;
        } else if (ret == 0) {
            continue; // Timeout
        }

        // Read events
        ssize_t len = read(inotifyFd_, buffer, sizeof(buffer));

        if (len < 0) {
            if (errno != EAGAIN) {
                Logger()->error("Read error: {}", std::strerror(errno));
            }
            continue;
        }

        // Process events
        bool needsRefresh = false;
        const struct inotify_event* event;

        for (char* ptr = buffer; ptr < buffer + len;
             ptr += sizeof(struct inotify_event) + event->len) {

            event = reinterpret_cast<const struct inotify_event*>(ptr);

            std::filesystem::path eventPath;
            {
                std::lock_guard<std::mutex> lock(watchMutex_);
                auto it = watchDescriptors_.find(event->wd);
                if (it != watchDescriptors_.end()) {
                    eventPath = it->second;
                } else {
                    continue;
                }
            }

            // Handle new directory creation
            if ((event->mask & IN_CREATE) && (event->mask & IN_ISDIR) && event->len > 0) {
                std::filesystem::path newDir = eventPath / event->name;
                addWatch(newDir);  // Add watch for new directory
                needsRefresh = true;
            }
            // Handle directory/file deletion
            else if (event->mask & IN_DELETE) {
                needsRefresh = true;
            }
            // Handle file modifications
            else if (event->mask & (IN_MODIFY | IN_CLOSE_WRITE)) {
                needsRefresh = true;
            }
            // Handle moves
            else if (event->mask & (IN_MOVED_FROM | IN_MOVED_TO)) {
                needsRefresh = true;
            }
        }

        if (needsRefresh) {
            Logger()->debug("File system changes detected, refreshing segmentations");
            refreshSegmentations();
        }
    }
}

