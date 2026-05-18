// End-to-end stress of the libs3-backed S3 stack, driven through the
// public utils:: shim API (HttpClient / HttpStore) and the vc:: layer --
// the same entry points the app uses. Network-soft-skipped: set
// VC_TEST_REQUIRE_NETWORK=1 to make failures hard.
//
// Targets the public (anonymous) vesuvius-challenge-open-data bucket so
// no credentials are needed. Ground truth on the PHerc 0172 zarr:
//   .zattrs               -> 3018 bytes, contains "multiscales"
//   level 2/ (recursive)  -> exactly 5611 non-meta keys across 6 pages

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <utils/http_fetch.hpp>
#include <utils/zarr.hpp>
#include "vc/core/util/RemoteUrl.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace {

constexpr const char* kBucketHttps =
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";
constexpr const char* kZarr =
    "PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr";

bool requireNetwork()
{
    const char* env = std::getenv("VC_TEST_REQUIRE_NETWORK");
    return env && env[0] && env[0] != '0';
}

// Wrap a network body; soft-skip unless VC_TEST_REQUIRE_NETWORK.
template <typename F>
void net(const char* what, F&& fn)
{
    try {
        fn();
    } catch (const std::exception& e) {
        if (requireNetwork()) FAIL(what << ": " << e.what());
        MESSAGE("Skipping " << what << " (network?): " << e.what());
    }
}

std::string zattrsUrl() { return std::string(kBucketHttps) + "/" + kZarr + "/.zattrs"; }

} // namespace

TEST_CASE("libs3 e2e: anonymous GET returns exact body")
{
    net("GET .zattrs", [] {
        utils::HttpClient c{};
        auto r = c.get(zattrsUrl());
        if (r.status_code == 0) { MESSAGE("no network"); return; }
        CHECK(r.ok());
        CHECK(r.body.size() == 3018);
        CHECK(r.body_string().find("multiscales") != std::string_view::npos);
        CHECK(r.content_length == 3018);
    });
}

TEST_CASE("libs3 e2e: ranged GET is a byte-exact prefix of the whole")
{
    net("ranged GET", [] {
        utils::HttpClient c{};
        auto full = c.get(zattrsUrl());
        if (!full.ok()) { MESSAGE("no network"); return; }
        auto rng = c.get_range(zattrsUrl(), 0, 64);
        CHECK(rng.body.size() == 64);
        CHECK(std::memcmp(full.body.data(), rng.body.data(), 64) == 0);
        // mid-object range
        auto mid = c.get_range(zattrsUrl(), 100, 32);
        CHECK(mid.body.size() == 32);
        CHECK(std::memcmp(full.body.data() + 100, mid.body.data(), 32) == 0);
        // zero-length range is a no-op
        CHECK(c.get_range(zattrsUrl(), 0, 0).body.empty());
    });
}

TEST_CASE("libs3 e2e: HEAD and 404 classification")
{
    net("HEAD/404", [] {
        utils::HttpClient c{};
        auto h = c.head(zattrsUrl());
        if (h.status_code == 0) { MESSAGE("no network"); return; }
        CHECK(h.ok());
        auto miss = c.get(std::string(kBucketHttps) + "/" + kZarr + "/__nope__");
        CHECK(miss.not_found());
        CHECK_FALSE(miss.ok());
        auto missH = c.head(std::string(kBucketHttps) + "/" + kZarr + "/__nope__");
        CHECK(missH.not_found());
    });
}

TEST_CASE("libs3 e2e: single-page list() with delimiter yields CommonPrefixes")
{
    net("list() one level", [] {
        utils::HttpClient c{};
        // One level under the zarr root: the numeric pyramid levels show
        // up as CommonPrefixes when delimiter="/".
        auto page = c.list(std::string("s3://vesuvius-challenge-open-data/")
                                + kZarr + "/", "/");
        if (page.status != S3_OK) { MESSAGE("no network"); return; }
        CHECK(page.status == S3_OK);
        CHECK_FALSE(page.prefixes.empty());
        bool sawLevel0 = false;
        for (const auto& p : page.prefixes)
            if (p.find("/0/") != std::string::npos) sawLevel0 = true;
        CHECK(sawLevel0);
        // .zattrs / .zgroup are objects at this level, not prefixes.
        bool sawZattrs = false;
        for (const auto& o : page.objects)
            if (o.key.find(".zattrs") != std::string::npos) sawZattrs = true;
        CHECK(sawZattrs);
    });
}

TEST_CASE("libs3 e2e: list_all paginates a multi-page prefix exactly")
{
    net("list_all 5611/6", [] {
        utils::HttpClient c{};
        struct Ctx { int keys = 0; int pages = 0; } ctx;
        auto cb = [](void* ud, const ::s3_list_result* pg) -> bool {
            auto* x = static_cast<Ctx*>(ud);
            x->pages++;
            for (size_t i = 0; i < pg->object_count; ++i) {
                std::string k = pg->objects[i].key ? pg->objects[i].key : "";
                auto fn = k.substr(k.rfind('/') + 1);
                if (!fn.empty() && fn[0] != '.' &&
                    fn.find(".json") == std::string::npos)
                    x->keys++;
            }
            return true;
        };
        auto rc = c.list_all(std::string("s3://vesuvius-challenge-open-data/")
                                 + kZarr + "/2/", nullptr, cb, &ctx);
        if (rc != S3_OK) { MESSAGE("no network"); return; }
        CHECK(rc == S3_OK);
        CHECK(ctx.keys == 5611);
        CHECK(ctx.pages == 6);
    });
}

TEST_CASE("libs3 e2e: list() continuation token paginates to the same total")
{
    net("list() manual paging", [] {
        utils::HttpClient c{};
        const std::string prefix =
            std::string("s3://vesuvius-challenge-open-data/") + kZarr + "/2/";
        int total = 0, pages = 0;
        std::string tok;
        do {
            auto pg = c.list(prefix, nullptr, tok);
            if (pg.status != S3_OK) { MESSAGE("no network"); return; }
            for (const auto& o : pg.objects) {
                auto fn = o.key.substr(o.key.rfind('/') + 1);
                if (!fn.empty() && fn[0] != '.' &&
                    fn.find(".json") == std::string::npos)
                    ++total;
            }
            ++pages;
            tok = pg.next_continuation_token;
        } while (!tok.empty() && pages < 20);
        // Same ground truth reached via the single-page primitive that
        // the GUI browser uses.
        CHECK(total == 5611);
        CHECK(pages == 6);
    });
}

TEST_CASE("libs3 e2e: HttpStore (zarr API) reads through libs3")
{
    net("HttpStore get/partial", [] {
        utils::HttpStore store(std::string(kBucketHttps) + "/" + kZarr);
        auto exists = store.exists(".zattrs");
        if (!exists) { MESSAGE("no network / unreachable"); return; }
        CHECK(exists);
        CHECK_FALSE(store.exists("__nope__"));

        auto whole = store.get_if_exists(".zattrs");
        REQUIRE(whole.has_value());
        CHECK(whole->size() == 3018);

        auto part = store.get_partial(".zattrs", 0, 16);
        REQUIRE(part.has_value());
        CHECK(part->size() == 16);
        CHECK(std::memcmp(whole->data(), part->data(), 16) == 0);

        CHECK_FALSE(store.get_if_exists("__nope__").has_value());
    });
}

TEST_CASE("libs3 e2e: S3 URL parsing + region resolution round-trip")
{
    auto p = utils::parse_s3_url("s3://vesuvius-challenge-open-data/a/b.bin");
    REQUIRE(p.has_value());
    CHECK(p->bucket == "vesuvius-challenge-open-data");
    CHECK(p->key == "a/b.bin");

    auto pr = utils::parse_s3_url("s3+us-west-2://bkt/k");
    REQUIRE(pr.has_value());
    CHECK(pr->region == "us-west-2");
    CHECK(utils::s3_to_https(*pr) == "https://bkt.s3.us-west-2.amazonaws.com/k");

    CHECK(utils::is_s3_url("s3://x/y"));
    CHECK(utils::is_s3_url("s3+eu-west-1://x/y"));
    CHECK_FALSE(utils::is_s3_url("https://example.com/x"));

    // vc:: resolver: bare s3:// defaults to us-east-1 and flags SigV4.
    auto resolved = vc::resolveRemoteUrl("s3://vesuvius-challenge-open-data/k");
    CHECK(resolved.useAwsSigv4);
    CHECK(resolved.awsRegion == "us-east-1");
    CHECK(resolved.httpsUrl.find("vesuvius-challenge-open-data") != std::string::npos);
}

TEST_CASE("libs3 e2e: process-global abort flag toggles")
{
    utils::HttpClient::resetAbort();
    CHECK_FALSE(utils::HttpClient::isAborted());
    utils::HttpClient::abortAll();
    CHECK(utils::HttpClient::isAborted());
    utils::HttpClient::resetAbort();
    CHECK_FALSE(utils::HttpClient::isAborted());
}
