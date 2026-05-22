// Data/IO coverage for PointCollections — the Qt-free class split out of
// VCCollection. Exercises mutation, queries, autofill, anchors, and a
// JSON save/load round-trip without any Qt.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/PointCollections.hpp"

#include <filesystem>

namespace fs = std::filesystem;

TEST_CASE("add/query points and collections")
{
    PointCollections c;
    const auto p0 = c.addPoint("alpha", {1, 2, 3});
    const auto p1 = c.addPoint("alpha", {4, 5, 6});

    const uint64_t cid = c.getCollectionId("alpha");
    CHECK(cid != 0);
    CHECK(c.getAllCollections().size() == 1);
    CHECK(c.getAllCollections().at(cid).points.size() == 2);
    CHECK(p0.id != p1.id);
    CHECK(p0.collectionId == cid);

    auto got = c.getPoint(p0.id);
    REQUIRE(got.has_value());
    CHECK(got->p == cv::Vec3f(1, 2, 3));

    CHECK(c.getPoints("alpha").size() == 2);
}

TEST_CASE("update and remove")
{
    PointCollections c;
    auto p = c.addPoint("a", {0, 0, 0});
    p.p = {9, 9, 9};
    c.updatePoint(p);
    CHECK(c.getPoint(p.id)->p == cv::Vec3f(9, 9, 9));

    c.removePoint(p.id);
    CHECK_FALSE(c.getPoint(p.id).has_value());
}

TEST_CASE("rename, clear, clearAll")
{
    PointCollections c;
    c.addPoint("a", {0, 0, 0});
    const uint64_t cid = c.getCollectionId("a");
    c.renameCollection(cid, "b");
    CHECK(c.getCollectionId("b") == cid);
    CHECK(c.getCollectionId("a") == 0);

    c.clearCollection(cid);  // removes the collection entirely
    CHECK(c.getAllCollections().count(cid) == 0);

    c.addPoint("x", {1, 1, 1});
    c.clearAll();
    CHECK(c.getAllCollections().empty());
}

TEST_CASE("tags and anchor2d + offset")
{
    PointCollections c;
    c.addPoint("a", {0, 0, 0});
    const uint64_t cid = c.getCollectionId("a");

    c.setCollectionTag(cid, "k", "v");
    CHECK(c.getCollectionTag(cid, "k") == std::optional<std::string>("v"));
    c.removeCollectionTag(cid, "k");
    CHECK_FALSE(c.getCollectionTag(cid, "k").has_value());

    c.setCollectionAnchor2d(cid, cv::Vec2f(10, 20));
    REQUIRE(c.getCollectionAnchor2d(cid).has_value());
    c.applyAnchorOffset(5, -5);
    auto a = c.getCollectionAnchor2d(cid);
    REQUIRE(a.has_value());
    CHECK((*a)[0] == doctest::Approx(15));
    CHECK((*a)[1] == doctest::Approx(15));
}

TEST_CASE("JSON round-trip preserves data")
{
    PointCollections c;
    c.addPoint("alpha", {1, 2, 3});
    c.addPoint("alpha", {4, 5, 6});
    c.addPoint("beta", {7, 8, 9});
    const uint64_t alpha = c.getCollectionId("alpha");
    c.setCollectionColor(alpha, {0.1f, 0.2f, 0.3f});
    c.setCollectionTag(alpha, "kind", "correction");

    const fs::path tmp = fs::temp_directory_path() /
        ("pc_roundtrip_" + std::to_string(::getpid()) + ".json");
    REQUIRE(c.saveToJSON(tmp.string()));

    PointCollections loaded;
    REQUIRE(loaded.loadFromJSON(tmp.string()));
    fs::remove(tmp);

    CHECK(loaded.getAllCollections().size() == c.getAllCollections().size());
    const uint64_t la = loaded.getCollectionId("alpha");
    CHECK(la != 0);
    CHECK(loaded.getPoints("alpha").size() == 2);
    CHECK(loaded.getPoints("beta").size() == 1);
    CHECK(loaded.getCollectionTag(la, "kind") == std::optional<std::string>("correction"));
}
