#pragma once

#include <filesystem>
#include <string>

namespace vc {

enum class FilesystemType { Local, NetworkMount, Unknown };

// Detect whether a path resides on a network-mounted filesystem.
// Linux-only (uses statfs + /proc/mounts). Returns Unknown on other platforms.
FilesystemType detectFilesystemType(const std::filesystem::path& path);

// Human-readable label for the filesystem type (e.g. "NFS", "CIFS", "fuse.s3fs").
// Returns empty string if detection fails or platform unsupported.
std::string filesystemTypeLabel(const std::filesystem::path& path);

}  // namespace vc
