#pragma once

/** @file */

#include <cstdint>
#include <string>

/** Provides programmatic access to codebase information */
struct ProjectInfo {
    /** Get the library version as a Major.Minor.Patch string */
    static auto VersionString() -> std::string;
    /** Get the library name and version string */
    static auto NameAndVersion() -> std::string;
    /** Get the full hash for the current git commit */
    static auto RepositoryHash() -> std::string;
    /** Get the short hash for the current git commit */
    static auto RepositoryShortHash() -> std::string;
};
